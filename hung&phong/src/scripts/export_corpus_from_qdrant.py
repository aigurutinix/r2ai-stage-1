"""Regenerate corpus export TỪ QDRANT (nguồn chân lý sau khi mở rộng v3).

export_corpus_vbpl.py cũ đọc từ HF + lọc v2 → KHÔNG gồm 119 VB thêm ở v3. Script này
scroll thẳng Qdrant (đúng những gì đang index) → documents.parquet + articles.parquet +
inventory.md phản ánh CHÍNH XÁC corpus hiện tại.

Usage: python scripts/export_corpus_from_qdrant.py
"""
import sys
from collections import Counter
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from backend.qdrant_store import QdrantStore

OUT = Path("data/corpus_vbpl_v2"); OUT.mkdir(parents=True, exist_ok=True)
KEEP = ("so_ky_hieu", "dieu_so", "dieu_tieu_de", "loai_van_ban", "title", "nam",
        "linh_vuc", "co_quan_ban_hanh", "source_url", "char_len")

store = QdrantStore()
total = store.count()
print(f"Scroll {store.collection}: {total:,} điểm...", flush=True)
rows, offset = [], None
while True:
    pts, offset = store.client.scroll(collection_name=store.collection, limit=4000,
                                      offset=offset, with_payload=True, with_vectors=False)
    for p in pts:
        pl = p.payload or {}
        rows.append({k: pl.get(k) for k in KEEP})
    if offset is None:
        break
df = pd.DataFrame(rows)
print(f"  {len(df):,} chunk. Dựng parquet...", flush=True)

# articles: 1 dòng / (số hiệu, LOẠI, điều) — số hiệu QH không unique giữa các loại VB;
# điều dài bị split nhiều chunk → gộp char_len.
art = (df.groupby(["so_ky_hieu", "loai_van_ban", "dieu_so"], dropna=False)
         .agg(title=("title", "first"),
              dieu_tieu_de=("dieu_tieu_de", "first"), char_len=("char_len", "sum"))
         .reset_index())
art = art[art["dieu_so"].notna() & (art["dieu_so"].astype("Int64") > 0)]
art.to_parquet(OUT / "articles.parquet", index=False)

# documents: 1 dòng / (số hiệu, loại) — số hiệu QH không unique giữa các loại VB
docs = (df.groupby(["so_ky_hieu", "loai_van_ban"], dropna=False)
          .agg(title=("title", "first"), nam=("nam", "first"),
               linh_vuc=("linh_vuc", "first"), co_quan_ban_hanh=("co_quan_ban_hanh", "first"),
               source_url=("source_url", "first"),
               n_dieu=("dieu_so", lambda s: s[s.notna() & (s > 0)].nunique()),
               char_len=("char_len", "sum"))
          .reset_index())
docs.to_parquet(OUT / "documents.parquet", index=False)

# inventory.md
L = [f"# Inventory corpus `{store.collection}` (regen từ Qdrant)\n",
     f"- Tổng chunk (điểm vector): **{len(df):,}**",
     f"- Tổng văn bản (số hiệu+loại): **{len(docs):,}**",
     f"- Tổng Điều: **{len(art):,}**\n", "## Theo loại văn bản"]
for t, n in Counter(docs["loai_van_ban"].dropna()).most_common():
    L.append(f"- {t}: {n}")
L.append("\n## Luật/Bộ luật nền tảng (kiểm tra bao phủ)")
prim = docs[docs["loai_van_ban"].isin(["Luật", "Bộ luật", "Pháp lệnh"])]
for kw in ["dân sự", "doanh nghiệp", "quảng cáo", "người tiêu dùng", "giao dịch điện tử",
           "công nghệ số", "lao động", "thương mại", "đầu tư", "quản lý thuế"]:
    h = prim[prim["title"].str.lower().str.contains(kw, na=False)]
    tag = " ; ".join(f"{r.so_ky_hieu}" for r in h.itertuples()) if len(h) else "—"
    L.append(f"- `{kw}`: {tag[:120]}")
(OUT / "inventory.md").write_text("\n".join(L), encoding="utf-8")
print(f"XONG: documents={len(docs):,} | articles={len(art):,} → {OUT}/")
