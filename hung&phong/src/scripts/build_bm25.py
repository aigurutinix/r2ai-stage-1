"""Dựng BM25 index từ collection Qdrant đang dùng → data/bm25_<collection>.pkl.

Usage: python scripts/build_bm25.py   (đọc QDRANT_COLLECTION từ .env)
"""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
from backend.bm25 import BM25Index
from backend.config import get_settings
from backend.qdrant_store import QdrantStore

s = get_settings()
store = QdrantStore()
total = store.count()
print(f"Collection: {store.collection} · {total:,} điểm. Đang scroll...", flush=True)

texts: list[str] = []
metas: list[dict] = []
offset = None
t0 = time.time()
KEEP = ("so_ky_hieu", "dieu_so", "loai_van_ban", "title", "nam", "source_url",
        "tinh_trang_hieu_luc", "co_quan_ban_hanh", "ngay_ban_hanh")
while True:
    points, offset = store.client.scroll(
        collection_name=store.collection, limit=2000,
        offset=offset, with_payload=True, with_vectors=False,
    )
    for p in points:
        pl = p.payload or {}
        texts.append(pl.get("text") or "")
        meta = {k: pl.get(k) for k in KEEP}
        meta["text"] = pl.get("text") or ""
        meta["_id"] = str(p.id)
        metas.append(meta)
    if len(texts) % 20000 == 0:
        print(f"  ...{len(texts):,}", flush=True)
    if offset is None:
        break

print(f"Scroll xong: {len(texts):,} điểm ({time.time()-t0:.0f}s). Build BM25...", flush=True)
t1 = time.time()
idx = BM25Index.build(texts, metas)
out = Path(s.hf_cache_dir).resolve().parent / f"bm25_{store.collection}.pkl"
idx.save(out)
print(f"XONG: build {time.time()-t1:.0f}s → {out} ({out.stat().st_size/1e6:.0f} MB)", flush=True)
