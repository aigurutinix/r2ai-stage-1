"""Ingest dataset tmquan/vbpl-vn → Qdrant. Có manifest theo dõi nguồn từng VB.

Đặt collection qua env QDRANT_COLLECTION (vd vbpl_v2) để không đụng collection cũ.

Usage:
    $env:QDRANT_COLLECTION="vbpl_v2"
    python -m ingest.run_vbpl --recreate [--no-keyword] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from backend.config import get_settings
from backend.embed import EmbeddingClient
from backend.qdrant_store import QdrantStore
from ingest.chunk import chunk_document
from ingest.embed_load import embed_and_upsert
from ingest.parse_vbpl import parse_vbpl_row
from ingest.scope import in_scope_vbpl

logger = logging.getLogger(__name__)

# Chỉ nạp cột cần — bỏ structure_json/extracted_json/file_paths_json (nặng).
_COLS = [
    "doc_name", "title", "legal_type", "doc_number", "issue_date", "year",
    "issuing_authority", "source", "source_url", "legal_area", "scope",
    "markdown", "char_len", "text_hash",
]


def run(recreate: bool, keyword_filter: bool = True, limit: int = 0) -> None:
    from datasets import load_dataset

    s = get_settings()
    logger.info("Nạp cache tmquan/vbpl-vn ...")
    ds = load_dataset("tmquan/vbpl-vn", "documents", split="train", cache_dir=s.hf_cache_dir)
    keep = [c for c in _COLS if c in ds.column_names]
    ds = ds.select_columns(keep)
    logger.info("  %d dòng · cột=%s", len(ds), keep)

    # ── Pha 1: thu thập + DEDUP theo số hiệu (giữ bản nhiều Điều nhất) ───────
    # tmquan có ~610 số hiệu trùng record → tránh embed trùng (vd 59/2020 ×2 = 436 chunk).
    import re as _re
    _SK_VALID = _re.compile(r"^\d+/\d{4}/")
    best: dict = {}     # sk -> (doc, row, n_dieu)
    others: list = []
    n_skip = 0
    for row in tqdm(ds, desc="parse vbpl", unit="doc"):
        if not in_scope_vbpl(row, keyword_filter):
            continue
        doc = parse_vbpl_row(row)
        if doc is None or not doc.dieus:
            n_skip += 1
            continue
        sk = (doc.so_ky_hieu or "").strip()
        nd = sum(1 for d in doc.dieus if d.dieu_so > 0)
        if _SK_VALID.match(sk):
            cur = best.get(sk)
            if cur is None or nd > cur[2]:
                best[sk] = (doc, row, nd)
        else:
            others.append((doc, row))
    all_docs = [(d, r) for d, r, _ in best.values()] + others
    # Manual override: bỏ bản tmquan của VB có trong MANUAL (.docx chính xác hơn)
    from ingest.manual_docs import MANUAL as _MANUAL
    all_docs = [(d, r) for d, r in all_docs if d.so_ky_hieu not in _MANUAL]
    logger.info("Sau dedup: %d VB (bỏ bản trùng/override)", len(all_docs))

    # ── Pha 2: chunk ─────────────────────────────────────────────────────────
    manifest: list[dict] = []
    chunks: list = []
    n_docs = 0
    for doc, row in all_docs:
        n_docs += 1
        manifest.append({
            "so_ky_hieu": doc.so_ky_hieu,
            "loai_van_ban": doc.loai_van_ban,
            "title": doc.title,
            "nam": doc.nam,
            "co_quan_ban_hanh": doc.co_quan_ban_hanh,
            "source": doc.nguon,
            "source_url": doc.source_url,
            "text_hash": str(row.get("text_hash") or ""),
            "char_len": int(row.get("char_len") or 0),
            "n_dieu": len(doc.dieus),
        })
        chunks.extend(chunk_document(doc))
        if limit and n_docs >= limit:
            break

    # ── Bổ sung VB thủ công (.docx) mà tmquan không có/sai (125/2020, 122/2020, 20/2026) ──
    if not limit:
        from ingest.manual_docs import load_manual_docs
        for mdoc in load_manual_docs():
            n_docs += 1
            manifest.append({
                "so_ky_hieu": mdoc.so_ky_hieu, "loai_van_ban": mdoc.loai_van_ban,
                "title": mdoc.title, "nam": mdoc.nam,
                "co_quan_ban_hanh": mdoc.co_quan_ban_hanh, "source": mdoc.nguon,
                "source_url": mdoc.source_url, "text_hash": "", "char_len": 0,
                "n_dieu": len(mdoc.dieus),
            })
            chunks.extend(chunk_document(mdoc))
            logger.info("  + manual: %s (%d điều)", mdoc.so_ky_hieu, len(mdoc.dieus))

    logger.info("Trong scope: %d docs (%d skip) → %d chunks", n_docs, n_skip, len(chunks))

    out_dir = Path(s.hf_cache_dir).resolve().parent  # ./data
    man_path = out_dir / "vbpl_v2_manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Manifest nguồn → %s (%d VB)", man_path, len(manifest))

    store = QdrantStore()
    logger.info("Collection đích: %s @ %s", store.collection, s.qdrant_url)
    store.ensure_collection(recreate=recreate)
    embedder = EmbeddingClient()
    n = embed_and_upsert(chunks, embedder=embedder, store=store)
    logger.info("XONG. %d chunks trong collection %s", n, store.collection)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--no-keyword", action="store_true", help="Bỏ lọc từ khoá (giữ mọi VB trung ương loại lõi).")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run(recreate=args.recreate, keyword_filter=not args.no_keyword, limit=args.limit)


if __name__ == "__main__":
    main()
