"""Re-embed corpus sang collection mới bằng model sentence-transformers (AITeamVN).

Scroll collection nguồn (vbpl_v2, vector bge-m3) → encode lại TEXT bằng model mới →
upsert sang collection đích (GIỮ NGUYÊN id + payload, chỉ đổi vector). Giữ vbpl_v2 để
rollback. BM25 tái dùng được (lexical, cùng text + cùng id).

Usage:
  python scripts/reembed_aiteam.py --src vbpl_v2 --dst vbpl_aiteam \
         --model AITeamVN/Vietnamese_Embedding_v2
"""
from __future__ import annotations
import argparse, os, sys, time
os.environ.setdefault("USE_TF", "0")
sys.stdout.reconfigure(encoding="utf-8")
from qdrant_client.http import models as qm
from backend.qdrant_store import QdrantStore

_IDX = [("doc_id", "keyword"), ("so_ky_hieu", "keyword"), ("loai_van_ban", "keyword"),
        ("co_quan_ban_hanh", "keyword"), ("tinh_trang_hieu_luc", "keyword"),
        ("linh_vuc", "keyword"), ("dieu_so", "integer"), ("ngay_ban_hanh", "keyword")]
_SCHEMA = {"keyword": qm.PayloadSchemaType.KEYWORD, "integer": qm.PayloadSchemaType.INTEGER}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="vbpl_v2")
    ap.add_argument("--dst", default="vbpl_aiteam")
    ap.add_argument("--model", default="AITeamVN/Vietnamese_Embedding_v2")
    ap.add_argument("--scroll", type=int, default=2000)
    ap.add_argument("--enc-batch", type=int, default=32)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    print(f"Load {args.model} (fp16)...", flush=True)
    m = SentenceTransformer(args.model, device="cuda")
    m = m.half()   # fp16: nửa VRAM + nhanh hơn (fp32 batch lớn gây OOM thrashing trên 3060 12GB)
    dim = m.get_sentence_embedding_dimension()
    print(f"  dim={dim}", flush=True)

    client = QdrantStore().client
    total_src = client.count(args.src, exact=True).count
    print(f"Nguồn {args.src}: {total_src:,} điểm → đích {args.dst} (dim {dim})", flush=True)

    if client.collection_exists(args.dst):
        client.delete_collection(args.dst)
    client.create_collection(args.dst,
                             vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))
    for name, t in _IDX:
        try:
            client.create_payload_index(args.dst, field_name=name, field_schema=_SCHEMA[t])
        except Exception:
            pass

    offset, total, t0 = None, 0, time.time()
    while True:
        pts, offset = client.scroll(args.src, limit=args.scroll, offset=offset,
                                    with_payload=True, with_vectors=False)
        if pts:
            texts = [(p.payload or {}).get("text") or " " for p in pts]
            vecs = m.encode(texts, batch_size=args.enc_batch, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
            points = [qm.PointStruct(id=p.id, vector=v.tolist(), payload=p.payload)
                      for p, v in zip(pts, vecs)]
            client.upsert(args.dst, points=points, wait=False)
            total += len(points)
            print(f"  {total:,}/{total_src:,} ({total/(time.time()-t0):.0f}/s)", flush=True)
        if offset is None:
            break
    print(f"XONG: {total:,} điểm → {args.dst} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
