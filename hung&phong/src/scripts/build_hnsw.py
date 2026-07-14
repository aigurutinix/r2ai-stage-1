"""Dựng HNSW dense index từ collection Qdrant (scroll vector+payload). Temp/tool."""
import sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
from backend.config import get_settings
from backend.qdrant_store import QdrantStore
from backend.hnsw_index import HnswDense

s = get_settings()
store = QdrantStore()
print(f"Scroll {store.collection} (kèm vector)...", flush=True)
KEEP = ("so_ky_hieu", "dieu_so", "loai_van_ban", "title", "nam", "source_url", "text")
vectors, metas = [], []
offset = None
t0 = time.time()
while True:
    pts, offset = store.client.scroll(collection_name=store.collection, limit=2000,
                                      offset=offset, with_payload=True, with_vectors=True)
    for p in pts:
        v = p.vector
        if isinstance(v, dict):  # named vectors
            v = next(iter(v.values()))
        vectors.append(v)
        pl = p.payload or {}
        m = {k: pl.get(k) for k in KEEP}
        m["_id"] = str(p.id)
        metas.append(m)
    if len(vectors) % 40000 == 0:
        print(f"  ...{len(vectors):,}", flush=True)
    if offset is None:
        break
dim = len(vectors[0])
print(f"Scroll xong {len(vectors):,} vector dim={dim} ({time.time()-t0:.0f}s). Build HNSW...", flush=True)
t1 = time.time()
idx = HnswDense.build(vectors, metas, dim)
# hnswlib (C++) KHÔNG ghi được path Unicode → lưu vào thư mục ASCII.
out = Path("C:/Users/PHONG/vbpl_idx")
out.mkdir(parents=True, exist_ok=True)
ip, mp = out / f"hnsw_{store.collection}.bin", out / f"hnsw_{store.collection}_meta.pkl"
idx.save(ip, mp)
if not ip.exists():
    raise RuntimeError(f"save_index KHÔNG tạo được {ip} (path Unicode?)")
print(f"XONG: build {time.time()-t1:.0f}s → {ip} ({ip.stat().st_size/1e6:.0f}MB) + {mp.name}", flush=True)
