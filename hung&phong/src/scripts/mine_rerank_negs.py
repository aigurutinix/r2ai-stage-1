"""Mine RERANKER-HARD negative — neg = điều AITeamVN xếp CAO nhưng SAI (cái nó thật sự nhầm).

Nhanh: dense top-K (Qdrant, ~0.05s) → BATCH rerank K ứng viên 1 lần (FlagReranker) → lấy
top-xếp-cao mà != positive làm neg. Khác mine dense thường (neg chỉ giống về embedding):
neg ở đây là điểm RERANK cao = đúng cái reranker nhầm → dạy sửa trúng chỗ.

Chạy: USE_TF=0 USE_RERANKER=false HYBRID_SEARCH=false QDRANT_COLLECTION=vbpl_aiteam ... \
  PYTHONUTF8=1 PYTHONPATH=. python scripts/mine_rerank_negs.py \
    --pairs data/_ft_pairs.json --n-neg 10 --topk 50 --out data/ft_rerank_train.jsonl
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
    ap.add_argument("--pairs", default="data/_ft_pairs.json")
    ap.add_argument("--n-neg", type=int, default=10)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--rr-model", default="AITeamVN/Vietnamese_Reranker")
    ap.add_argument("--out", default="data/ft_rerank_train.jsonl")
    args = ap.parse_args()

    from backend.rag import RAGPipeline
    from tests.build_submission_v12 import hits_to_cands, _law_name
    from FlagEmbedding import FlagReranker

    pairs = json.loads((ROOT / args.pairs).read_text(encoding="utf-8"))
    rag = RAGPipeline()
    rag.settings.use_reranker = False  # dense thô nhanh; ta tự rerank
    rr = FlagReranker(args.rr_model, use_fp16=True)

    out_f = open(ROOT / args.out, "w", encoding="utf-8")
    t0 = time.time(); n_ok = 0
    for i, p in enumerate(pairs, 1):
        q = p["query"]; pos_k = art_key(p["pos_art"])
        hits = rag.retrieve(q, top_k=args.topk)
        # text + art từng ứng viên
        items = []
        for h in hits:
            pl = h.get("payload", {}) or {}
            sk = str(pl.get("so_ky_hieu") or ""); ds = pl.get("dieu_so")
            txt = (pl.get("text") or "")[:1400]
            if sk and ds is not None and len(txt) > 50:
                items.append((f"{sk}|{_law_name(pl)}|Điều {ds}", txt))
        if not items:
            continue
        scores = rr.compute_score([[q, t] for _, t in items], normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        order = sorted(range(len(items)), key=lambda j: scores[j], reverse=True)
        negs = []
        for j in order:                       # top RERANK trước = neg khó nhất
            if art_key(items[j][0]) == pos_k:
                continue
            negs.append(items[j][1])
            if len(negs) >= args.n_neg:
                break
        if len(negs) < 3:
            continue
        out_f.write(json.dumps({"query": q, "pos": [p["pos_text"]], "neg": negs}, ensure_ascii=False) + "\n")
        n_ok += 1
        if i % 150 == 0:
            print(f"  [{i}/{len(pairs)}] {i/(time.time()-t0):.2f} q/s · {n_ok} mẫu", flush=True)
    out_f.close()
    print(f"XONG: {n_ok} mẫu (reranker-hard neg) ({time.time()-t0:.0f}s) → {args.out}", flush=True)


if __name__ == "__main__":
    main()
