#!/usr/bin/env python3
"""Hybrid rerank: score RRF chunks with original question + sub-queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rerank_retrieval import (
    DEFAULT_PRIMARY_WEIGHT,
    DEFAULT_SUB_WEIGHT,
    load_reranker,
    rerank_chunks_hybrid,
)
from subquery_loader import load_subquery_index, rerank_primary_and_subs

DEFAULT_RETRIEVED = ROOT / "test" / "R2AIStage1_retrieved_rrf.json"
DEFAULT_SUBQUERIES = ROOT / "test" / "R2AIStage1_subqueries (1).json"
DEFAULT_OUTPUT = ROOT / "test" / "R2AIStage1_reranked.json"
DEFAULT_RERANK_MODEL = ROOT / "models" / "Vietnamese_Reranker"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid rerank on RRF chunks using original question + sub-queries"
    )
    p.add_argument(
        "--retrieved-input",
        type=Path,
        default=DEFAULT_RETRIEVED,
        help="JSON with id, question, chunks (post-RRF, pre-rerank)",
    )
    p.add_argument(
        "--subqueries",
        type=Path,
        default=DEFAULT_SUBQUERIES,
        help="Sub-query JSON (R2AIStage1_subqueries (1).json)",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--rerank-model", type=Path, default=DEFAULT_RERANK_MODEL)
    p.add_argument("--device-rerank", default="cuda")
    p.add_argument("--rerank-batch", type=int, default=8)
    p.add_argument("--top-k", type=int, default=4, help="Final chunks after rerank")
    p.add_argument(
        "--rrf-top-k",
        type=int,
        default=15,
        help="Max RRF chunks to rerank per question",
    )
    p.add_argument("--primary-weight", type=float, default=DEFAULT_PRIMARY_WEIGHT)
    p.add_argument("--sub-weight", type=float, default=DEFAULT_SUB_WEIGHT)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-id", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    retrieved = json.loads(args.retrieved_input.read_text(encoding="utf-8"))
    sub_index = load_subquery_index(args.subqueries)

    items = [r for r in retrieved if r["id"] >= args.start_id]
    if args.limit is not None:
        items = items[: args.limit]

    print(f"Loading reranker: {args.rerank_model}")
    reranker = load_reranker(args.rerank_model, args.device_rerank)

    results: list[dict] = []
    for item in items:
        qid = item["id"]
        chunks = list(item.get("chunks") or [])[: args.rrf_top_k]
        question = str(item.get("question", "")).strip()
        spec = sub_index.get(qid)
        primary, subs = rerank_primary_and_subs(spec, question)

        reranked = rerank_chunks_hybrid(
            primary,
            subs,
            chunks,
            reranker,
            top_k=args.top_k,
            batch_size=args.rerank_batch,
            primary_weight=args.primary_weight,
            sub_weight=args.sub_weight,
        )
        results.append(
            {
                "id": qid,
                "question": question,
                "primary_query": primary,
                "sub_queries": subs or None,
                "chunks": reranked,
            }
        )
        if len(results) % 50 == 0:
            print(f"  reranked {len(results)}/{len(items)} questions")

    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    multi = sum(1 for r in results if r.get("sub_queries"))
    print(
        f"Done — {len(results)} questions ({multi} with sub-queries) → {args.output}\n"
        f"Formula: {args.primary_weight} * original + {args.sub_weight} * mean(sub_scores)"
    )


if __name__ == "__main__":
    main()
