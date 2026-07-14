"""Submission v3: GIỮ answer của v2, re-derive relevant_docs/articles từ hybrid+rerank
retrieval top-k (broaden để tăng recall). Retrieval-only, KHÔNG chạy LLM.

Chạy với: HYBRID_SEARCH=true USE_RERANKER=true USE_TF=0
Usage: python -m tests.build_submission_v3 --v2 data/submission_v2.json --out data/submission_v3.json --top-k 10
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
logger = logging.getLogger("sub_v3")
logger.setLevel(logging.INFO)
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="data/submission_v2.json")
    ap.add_argument("--out", default="data/submission_v3.json")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    v2 = json.loads((ROOT / args.v2).read_text(encoding="utf-8"))
    rag = RAGPipeline()
    logger.info("hybrid=%s reranker=%s | %d câu", rag.settings.hybrid_search,
                rag.settings.use_reranker, len(v2))

    out: list[dict] = []
    out_path = ROOT / args.out
    for i, r in enumerate(v2, 1):
        t0 = time.perf_counter()
        hits = rag.retrieve(r["question"], top_k=args.top_k)
        docs, arts = [], []
        seen_d, seen_a = set(), set()
        for h in hits:
            p = h.get("payload", {})
            sk, ds = p.get("so_ky_hieu"), p.get("dieu_so")
            if not sk:
                continue
            name = _law_name(p)
            d = f"{sk}|{name}"
            if d not in seen_d:
                seen_d.add(d); docs.append(d)
            if ds is not None and int(ds) > 0:
                a = f"{sk}|{name}|Điều {ds}"
                if a not in seen_a:
                    seen_a.add(a); arts.append(a)
        out.append({
            "id": r["id"], "question": r["question"], "answer": r["answer"],
            "relevant_docs": docs, "relevant_articles": arts,
        })
        if i % 5 == 0 or i == len(v2):
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        if i % 50 == 0 or i == len(v2):
            logger.info("[%d/%d] arts=%d %.1fs", i, len(v2), len(arts), time.perf_counter() - t0)

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    n_empty = sum(1 for r in out if not r["relevant_articles"])
    logger.info("XONG %d câu → %s | rỗng articles: %d", len(out), out_path, n_empty)


if __name__ == "__main__":
    main()
