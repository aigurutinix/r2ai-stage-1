"""IMPROVE#3: mở rộng corpus — thêm luật nền tảng bị scope-filter loại oan.

Lọc cũ bỏ luật không chứa từ-khóa-SME trên title → mất BLDS, Quảng cáo, GDĐT,
BVQLNTD, Luật CN công nghệ số (AI)... Bộ lọc MỚI:
  - GIỮ TẤT CẢ Luật/Bộ luật/Pháp lệnh trung ương (nền tảng, ít nhiễu).
  - GIỮ Nghị định/Thông tư trung ương khớp từ khóa SME MỞ RỘNG (thêm domain mới).

Incremental: chỉ thêm doc có số ký hiệu CHƯA có trong corpus → embed + upsert,
KHÔNG recreate (tránh re-embed 8h).

Usage:
  python scripts/expand_corpus.py                 # REPORT (không ghi)
  python scripts/expand_corpus.py --apply         # embed + upsert thật
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

from backend.config import get_settings
from ingest.scope import VBPL_CORE_TYPES, SME_KEYWORDS_VBPL

PRIMARY = {"Luật", "Bộ luật", "Pháp lệnh"}
# Luật nền tảng: keyword nghiệp vụ + bổ sung domain còn thiếu (rộng tay hơn vì luật ít, ít nhiễu).
PRIMARY_KW = SME_KEYWORDS_VBPL + (
    "dân sự", "quảng cáo", "người tiêu dùng", "giao dịch điện tử", "công nghệ số",
    "trí tuệ nhân tạo", "trọng tài", "công chứng", "khuyến mại", "chữ ký số",
    "an ninh mạng", "dữ liệu cá nhân",
)
# NĐ/TT: CHỈ thêm domain mới còn thiếu (siết — tránh kéo 9000 thông tư).
DECREE_KW = (
    "quảng cáo", "người tiêu dùng", "bảo vệ quyền lợi", "giao dịch điện tử",
    "trí tuệ nhân tạo", "công nghiệp công nghệ số", "chữ ký số",
)

def in_scope_v3(row: dict) -> bool:
    if str(row.get("scope") or "").strip() != "trung_uong":
        return False
    lt = str(row.get("legal_type") or "").strip()
    blob = " ".join(str(row.get(k) or "") for k in ("title", "legal_area")).lower()
    if lt in PRIMARY:
        return any(kw in blob for kw in PRIMARY_KW)
    if lt in VBPL_CORE_TYPES:               # NĐ/TT... chỉ domain MỚI
        return any(kw in blob for kw in DECREE_KW)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    from datasets import load_dataset
    s = get_settings()

    # Dedup theo (số hiệu + LOẠI VB): số hiệu QH không unique (vd 91/2015/QH13 vừa
    # là Bộ luật Dân sự vừa là Nghị quyết) → dedup theo số hiệu đơn sẽ che mất luật.
    dfc = pd.read_parquet(ROOT / "data/corpus_vbpl_v2/documents.parquet")
    existing = set((str(r.so_ky_hieu).strip(), str(r.loai_van_ban).strip())
                   for r in dfc.itertuples() if pd.notna(r.so_ky_hieu))
    print(f"Corpus hiện tại: {len(existing)} (số hiệu, loại)", flush=True)

    print("Nạp dataset nguồn tmquan/vbpl-vn ...", flush=True)
    cols = ["doc_name","title","legal_type","doc_number","issue_date","year",
            "issuing_authority","source","source_url","legal_area","scope","markdown","char_len","text_hash"]
    ds = load_dataset("tmquan/vbpl-vn", "documents", split="train", cache_dir=s.hf_cache_dir)
    ds = ds.select_columns([c for c in cols if c in ds.column_names])

    def sky(row):
        raw = row.get("doc_number")
        v = raw[0] if isinstance(raw, list) and raw else raw
        return str(v or "").strip()

    new_rows, by_type = [], Counter()
    for row in ds:
        if not in_scope_v3(row):
            continue
        key = (sky(row), str(row.get("legal_type") or "").strip())
        if key[0] and key in existing:
            continue
        new_rows.append(row); by_type[str(row.get("legal_type"))] += 1
    print(f"\n=== DOC MỚI sẽ thêm: {len(new_rows)} ===")
    for t, n in by_type.most_common():
        print(f"  {t}: {n}")

    # xác nhận các luật nền tảng trọng yếu
    print("\n=== Luật nền tảng trọng yếu trong tập MỚI? ===")
    want = {"Bộ luật Dân sự":"dân sự","Luật Quảng cáo":"quảng cáo",
            "Luật Bảo vệ quyền lợi NTD":"bảo vệ quyền lợi người tiêu dùng",
            "Luật Giao dịch điện tử":"giao dịch điện tử","Luật CN công nghệ số":"công nghệ số"}
    for name, kw in want.items():
        hits = [r for r in new_rows if kw in str(r.get("title") or "").lower()
                and str(r.get("legal_type")) in PRIMARY]
        tag = " ; ".join(f"{r.get('doc_number')}({r.get('year')})" for r in hits[:6]) or "—"
        print(f"  {'✓' if hits else '✗'} {name}: {tag}")
    # BLDS 91/2015 đích danh (current civil code)
    blds = [r for r in new_rows if sky(r)=="91/2015/QH13"]
    print(f"  → BLDS 91/2015/QH13 trong tập mới: {'CÓ' if blds else 'KHÔNG'}")

    if not args.apply:
        print("\n(REPORT — chưa ghi. Thêm --apply để embed+upsert)")
        return

    # ---- APPLY: parse + chunk + embed + upsert ----
    from backend.embed import EmbeddingClient
    from backend.qdrant_store import QdrantStore
    from ingest.chunk import chunk_document
    from ingest.parse_vbpl import parse_vbpl_row
    from ingest.embed_load import embed_and_upsert

    chunks, manifest = [], []
    for row in new_rows:
        doc = parse_vbpl_row(row)
        if doc is None or not doc.dieus:
            continue
        manifest.append({"so_ky_hieu": doc.so_ky_hieu, "loai_van_ban": doc.loai_van_ban,
                         "title": doc.title, "nam": doc.nam, "source_url": doc.source_url,
                         "n_dieu": len(doc.dieus)})
        chunks.extend(chunk_document(doc))
    print(f"\nParse xong: {len(manifest)} VB → {len(chunks)} chunks. Embed+upsert...", flush=True)
    (ROOT / "data/vbpl_v3_added_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    store = QdrantStore()
    store.ensure_collection(recreate=False)          # KHÔNG xóa collection cũ
    n = embed_and_upsert(chunks, embedder=EmbeddingClient(), store=store)
    print(f"XONG: +{n} chunks vào {store.collection}. Nhớ rebuild BM25 + HNSW.")


if __name__ == "__main__":
    main()
