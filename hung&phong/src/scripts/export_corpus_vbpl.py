"""Xuất corpus đang dùng (vbpl_v2) ra 1 thư mục để QUẢN LÝ / THEO DÕI / PHÂN TÍCH.

Tạo data/corpus_vbpl_v2/:
  - documents.parquet : sổ đăng ký từng VĂN BẢN (metadata + provenance, không kèm text)
  - articles.parquet  : từng ĐIỀU đã index (so_ky_hieu, Điều, tiêu đề, nội dung) — đây
                        chính là các đơn vị nằm trong vector DB
  - manifest.json     : provenance đầy đủ (source_url / text_hash) từng VB
  - inventory.md      : thống kê (theo loại, theo lĩnh vực, danh sách luật gốc)
  - README.md         : giải thích nguồn + cách lọc + cách tái tạo

Usage: python scripts/export_corpus_vbpl.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import os
os.environ.pop("HF_DATASETS_OFFLINE", None)
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from backend.config import get_settings
from ingest.parse_vbpl import parse_vbpl_row
from ingest.scope import in_scope_vbpl

OUT = Path(os.environ.get("CORPUS_OUT", "data/corpus_vbpl_v2"))
OUT.mkdir(parents=True, exist_ok=True)
COLS = ["doc_name", "title", "legal_type", "doc_number", "issue_date", "year",
        "issuing_authority", "source", "source_url", "legal_area", "scope",
        "markdown", "char_len", "text_hash"]

DOMAINS = [
    ("Doanh nghiệp", ["doanh nghiệp"]), ("Đầu tư", ["đầu tư"]),
    ("Lao động/BHXH", ["lao động", "việc làm", "bảo hiểm xã hội", "an toàn"]),
    ("Thuế/Hải quan", ["thuế", "hải quan", "hóa đơn"]),
    ("Kế toán/Kiểm toán", ["kế toán", "kiểm toán", "thống kê"]),
    ("Chứng khoán/Bảo hiểm", ["chứng khoán", "tín dụng", "bảo hiểm"]),
    ("Thương mại/Cạnh tranh", ["thương mại", "cạnh tranh", "trọng tài"]),
    ("Đấu thầu", ["đấu thầu"]), ("Sở hữu trí tuệ", ["sở hữu trí tuệ"]),
    ("Phá sản", ["phá sản"]), ("Đất đai/Xây dựng", ["đất đai", "xây dựng"]),
]
def domain_of(t):
    tl = t.lower()
    for n, ks in DOMAINS:
        if any(k in tl for k in ks):
            return n
    return "Khác"

s = get_settings()
print("Nạp cache vbpl-vn...", flush=True)
ds = load_dataset("tmquan/vbpl-vn", "documents", split="train", cache_dir=s.hf_cache_dir)
ds = ds.select_columns([c for c in COLS if c in ds.column_names])

import re as _re
_SK_VALID = _re.compile(r"^\d+/\d{4}/")   # số hiệu chuẩn (XX/YYYY/loại) → dedup an toàn

# ── Pha 1: thu thập + DEDUP theo số hiệu (giữ bản nhiều Điều nhất) ───────────
# tmquan có ~610 số hiệu trùng record (vd 59/2020 xuất hiện 2 lần → 436 điều).
# Giữ bản đầy đủ nhất; số hiệu không chuẩn ("Không số", "178-CP") để nguyên (VB khác nhau).
best: dict = {}     # sk -> (doc, row, n_dieu)
others: list = []   # số hiệu không chuẩn → giữ hết
seen = 0
for row in ds:
    if not in_scope_vbpl(row):
        continue
    doc = parse_vbpl_row(row)
    if doc is None or not doc.dieus:
        continue
    seen += 1
    sk = (doc.so_ky_hieu or "").strip()
    nd = sum(1 for d in doc.dieus if d.dieu_so > 0)
    if _SK_VALID.match(sk):
        cur = best.get(sk)
        if cur is None or nd > cur[2]:
            best[sk] = (doc, row, nd)
    else:
        others.append((doc, row))
    if seen % 4000 == 0:
        print(f"  ...quét {seen} VB", flush=True)

all_docs = [(d, r) for d, r, _ in best.values()] + others
# Manual override: bỏ bản tmquan của VB có trong MANUAL (.docx chính xác hơn — vd 135/2020)
from ingest.manual_docs import MANUAL as _MANUAL
all_docs = [(d, r) for d, r in all_docs if d.so_ky_hieu not in _MANUAL]
print(f"Sau dedup: {len(all_docs)} VB (từ {seen} record, bỏ {seen - len(all_docs)} bản trùng/override)", flush=True)

# ── Pha 2: build rows ────────────────────────────────────────────────────────
doc_rows, art_rows, manifest = [], [], []
by_type, by_dom = Counter(), Counter()
PRIMARY = {"Luật", "Bộ luật", "Pháp lệnh", "Văn bản hợp nhất"}
primary_laws = []
n = 0
for doc, row in all_docs:
    n += 1
    dom = domain_of(doc.title)
    by_type[doc.loai_van_ban] += 1
    by_dom[dom] += 1
    rec = {
        "so_ky_hieu": doc.so_ky_hieu, "loai_van_ban": doc.loai_van_ban,
        "title": doc.title, "nam": doc.nam, "linh_vuc": dom,
        "co_quan_ban_hanh": doc.co_quan_ban_hanh, "source_url": doc.source_url,
        "text_hash": str(row.get("text_hash") or ""), "char_len": int(row.get("char_len") or 0),
        "n_dieu": len(doc.dieus),
    }
    doc_rows.append(rec)
    manifest.append(rec)
    if doc.loai_van_ban in PRIMARY:
        primary_laws.append((doc.so_ky_hieu, doc.loai_van_ban, doc.title, doc.nam, len(doc.dieus), dom))
    for d in doc.dieus:
        art_rows.append({
            "so_ky_hieu": doc.so_ky_hieu, "loai_van_ban": doc.loai_van_ban,
            "title": doc.title, "dieu_so": d.dieu_so, "dieu_tieu_de": d.dieu_tieu_de,
            "char_len": d.char_len, "text": d.text, "source_url": doc.source_url,
        })
    if n % 2000 == 0:
        print(f"  ...{n} VB", flush=True)

# ── Bổ sung VB thủ công (.docx) mà tmquan không có/sai ───────────────────────
from ingest.manual_docs import load_manual_docs
for mdoc in load_manual_docs():
    n += 1
    dom = domain_of(mdoc.title)
    by_type[mdoc.loai_van_ban] += 1
    by_dom[dom] += 1
    rec = {
        "so_ky_hieu": mdoc.so_ky_hieu, "loai_van_ban": mdoc.loai_van_ban,
        "title": mdoc.title, "nam": mdoc.nam, "linh_vuc": dom,
        "co_quan_ban_hanh": mdoc.co_quan_ban_hanh, "source_url": "",
        "text_hash": "", "char_len": 0, "n_dieu": len(mdoc.dieus),
    }
    doc_rows.append(rec); manifest.append(rec)
    for d in mdoc.dieus:
        art_rows.append({
            "so_ky_hieu": mdoc.so_ky_hieu, "loai_van_ban": mdoc.loai_van_ban,
            "title": mdoc.title, "dieu_so": d.dieu_so, "dieu_tieu_de": d.dieu_tieu_de,
            "char_len": d.char_len, "text": d.text, "source_url": "",
        })
    print(f"  + manual: {mdoc.so_ky_hieu} ({len(mdoc.dieus)} điều)", flush=True)

pq.write_table(pa.Table.from_pylist(doc_rows), OUT / "documents.parquet")
pq.write_table(pa.Table.from_pylist(art_rows), OUT / "articles.parquet")
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

# inventory.md
inv = [f"# Kiểm kê corpus đang dùng — vbpl_v2\n",
       f"- Nguồn: `tmquan/vbpl-vn` (vbpl.vn — Bộ Tư pháp)\n",
       f"- Văn bản: **{n:,}**  ·  Điều (chunk): **{len(art_rows):,}**\n",
       "\n## Theo loại văn bản\n| Loại | Số VB |\n|---|---:|"]
for k, v in by_type.most_common():
    inv.append(f"| {k} | {v:,} |")
inv.append("\n## Theo lĩnh vực\n| Lĩnh vực | Số VB |\n|---|---:|")
for k, v in by_dom.most_common():
    inv.append(f"| {k} | {v:,} |")
inv.append(f"\n## Luật gốc (Luật/Bộ luật/Pháp lệnh/VBHN) — {len(primary_laws)} VB\n")
inv.append("| so_ky_hieu | loại | năm | điều | lĩnh vực | tên |\n|---|---|---|---:|---|---|")
for sk, lo, ti, nm, nd, dm in sorted(primary_laws, key=lambda x: (-x[4])):
    inv.append(f"| {sk} | {lo} | {nm} | {nd} | {dm} | {ti[:48]} |")
(OUT / "inventory.md").write_text("\n".join(inv) + "\n", encoding="utf-8")

# README
readme = f"""# Corpus đang dùng cho chatbot — vbpl_v2

**Nguồn DUY NHẤT:** HuggingFace `tmquan/vbpl-vn` — scrape từ **vbpl.vn** (CSDL quốc gia
về VBPL, Bộ Tư pháp). Không trộn nguồn khác.

## Đã lọc gì (từ 158.822 VB gốc → {n:,} VB)
1. Cấp **trung ương** (bỏ văn bản địa phương).
2. **Loại lõi**: Luật, Bộ luật, Pháp lệnh, Nghị định, Thông tư, TTLT, VBHN, Nghị quyết, Quyết định.
3. **Từ khoá DN/SME** (DN, thuế, lao động, đấu thầu, hải quan, hoá đơn, ...).

## File trong thư mục
- `documents.parquet` — sổ đăng ký từng VĂN BẢN (metadata + provenance, không kèm text).
- `articles.parquet`  — từng ĐIỀU đã index ({len(art_rows):,} dòng) = đơn vị trong vector DB (kèm nội dung).
- `manifest.json`     — provenance đầy đủ (so_ky_hieu, source_url vbpl.vn, text_hash) từng VB.
- `inventory.md`      — thống kê theo loại / lĩnh vực + danh sách luật gốc.

## Tái tạo
```
python scripts/export_corpus_vbpl.py          # xuất lại thư mục này
python -m ingest.run_vbpl --recreate          # ingest lại vào Qdrant (collection vbpl_v2)
```
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")

print(f"\nXONG → {OUT.resolve()}")
print(f"  documents.parquet: {n:,} VB")
print(f"  articles.parquet : {len(art_rows):,} điều")
print(f"  manifest.json, inventory.md, README.md")
