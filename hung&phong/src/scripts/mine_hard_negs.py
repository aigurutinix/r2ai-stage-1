"""Mine HARD NEGATIVE + format dữ liệu fine-tune reranker (FlagEmbedding JSONL).

Với mỗi (query, positive=chunk Claude đã sinh query): retrieve top-K bằng chính retriever
(reranker ON) → các điều XẾP CAO mà KHÁC điều positive = hard negative (giống-mà-sai).
Loại false-negative: bỏ ứng viên CÙNG (sk,Điều) với positive.

Output JSONL: {"query","pos":[text],"neg":[text×N]} — đúng format FlagEmbedding reranker FT.

Chạy: RERANKER_MODEL=AITeamVN/Vietnamese_Reranker QDRANT_COLLECTION=vbpl_aiteam ... \
  PYTHONUTF8=1 PYTHONPATH=. python scripts/mine_hard_negs.py \
    --pairs data/_ft_queries.json --n-neg 10 --topk 30 --out data/ft_rerank_train.jsonl
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def art_key(art: str):
    p = art.split("|"); return (p[0].strip(), p[-1].strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/_ft_pairs.json", help="[{query,pos_art,pos_text}]")
    ap.add_argument("--n-neg", type=int, default=10)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--out", default="data/ft_rerank_train.jsonl")
    args = ap.parse_args()

    from backend.rag import RAGPipeline
    from tests.build_submission_v12 import hits_to_cands

    pairs = json.loads((ROOT / args.pairs).read_text(encoding="utf-8"))
    if isinstance(pairs, dict):
        pairs = pairs.get("pairs", [])

    rag = RAGPipeline()
    out_f = open(ROOT / args.out, "w", encoding="utf-8")
    t0 = time.time()
    n_ok = 0
    n_neg_total = 0
    for i, p in enumerate(pairs, 1):
        q = p["query"]
        pos_art = p["pos_art"]; pos_text = p["pos_text"]
        pos_k = art_key(pos_art)
        hits = rag.retrieve(q, top_k=args.topk)
        negs = []
        for c in hits_to_cands(hits):
            if art_key(c["art"]) == pos_k:
                continue  # chính positive → bỏ (chống false-neg)
            txt = (c.get("payload", {}) or {}).get("text") if isinstance(c, dict) else None
            # hits_to_cands không giữ text → lấy từ hit payload
            negs.append(c["art"])
            if len(negs) >= args.n_neg:
                break
        # lấy text cho negs từ payload của hits gốc
        art2text = {}
        for h in hits:
            pl = h.get("payload", {}) or {}
            sk = str(pl.get("so_ky_hieu") or ""); ds = pl.get("dieu_so")
            if sk and ds is not None:
                from tests.build_submission_v12 import _law_name
                art2text[f"{sk}|{_law_name(pl)}|Điều {ds}"] = (pl.get("text") or "")[:1400]
        neg_texts = [art2text.get(a, "") for a in negs]
        neg_texts = [t for t in neg_texts if t and len(t) > 50]
        if len(neg_texts) < 3:
            continue
        rec = {"query": q, "pos": [pos_text], "neg": neg_texts[: args.n_neg]}
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_ok += 1; n_neg_total += len(rec["neg"])
        if i % 100 == 0:
            print(f"  [{i}/{len(pairs)}] {i/(time.time()-t0):.2f} q/s · {n_ok} mẫu", flush=True)
    out_f.close()
    print(f"XONG: {n_ok} mẫu train, TB {n_neg_total/max(1,n_ok):.1f} neg/mẫu → {args.out}", flush=True)


if __name__ == "__main__":
    main()
