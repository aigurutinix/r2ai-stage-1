"""v4 retrieval: lưu top-N ứng viên KÈM điểm reranker để chọn THÍCH ỨNG (offline).
Giữ answer của v2. Dùng HNSW (nhanh) + hybrid + rerank.

Chạy: HYBRID_SEARCH=true USE_RERANKER=true USE_HNSW=true USE_TF=0
Usage: python -m tests.build_submission_v4 --v2 data/submission_v2.json
       --out data/submission_v4_scored.json --fetch 15
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from backend.rag import RAGPipeline
from tests.build_submission import _law_name

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger("sub_v4")
logger.setLevel(logging.INFO)
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="data/submission_v2.json")
    ap.add_argument("--out", default="data/submission_v4_scored.json")
    ap.add_argument("--fetch", type=int, default=15)
    args = ap.parse_args()

    v2 = json.loads((ROOT / args.v2).read_text(encoding="utf-8"))
    rag = RAGPipeline()
    s = rag.settings
    logger.info("hnsw=%s hybrid=%s reranker=%s | %d câu", s.use_hnsw, s.hybrid_search,
                s.use_reranker, len(v2))

    out: list[dict] = []
    out_path = ROOT / args.out
    t_all = time.time()
    for i, r in enumerate(v2, 1):
        hits = rag.retrieve(r["question"], top_k=args.fetch)
        cands = []
        for h in hits:
            p = h.get("payload", {})
            sk, ds = p.get("so_ky_hieu"), p.get("dieu_so")
            if not sk or ds is None or int(ds) <= 0:
                continue
            name = _law_name(p)
            cands.append({
                "art": f"{sk}|{name}|Điều {ds}", "doc": f"{sk}|{name}",
                "rr": round(float(h.get("rerank_score", 0.0)), 4),
                "adj": round(float(h.get("adj_score", 0.0)), 4),
                "nam": p.get("nam"), "loai": p.get("loai_van_ban"),
            })
        out.append({"id": r["id"], "question": r["question"], "answer": r["answer"], "candidates": cands})
        if i % 20 == 0 or i == len(v2):
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        if i % 200 == 0 or i == len(v2):
            logger.info("[%d/%d] %.1f câu/s", i, len(v2), i / (time.time() - t_all))

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("XONG %d câu (%.0fs) → %s", len(out), time.time() - t_all, out_path)


if __name__ == "__main__":
    main()
