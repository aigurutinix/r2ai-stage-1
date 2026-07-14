#!/usr/bin/env python3
"""Generate sub-query decomposition file for R2AIStage1 questions (rule-based, no LLM)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qdrant_config import get_embed_model_path
from query_decompose import (
    DEFAULT_THRESHOLD_1,
    DEFAULT_THRESHOLD_2,
    load_embed_tokenizer,
    resolve_queries,
)

DEFAULT_QUESTIONS = ROOT / "test" / "R2AIStage1DATA.json"
DEFAULT_OUTPUT = ROOT / "test" / "R2AIStage1_subqueries.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate sub-query JSON (rule-based)")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--embed-model", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-id", type=int, default=1)
    p.add_argument("--token-threshold-1", type=int, default=DEFAULT_THRESHOLD_1)
    p.add_argument("--token-threshold-2", type=int, default=DEFAULT_THRESHOLD_2)
    return p.parse_args()


def load_questions(path: Path, start_id: int, limit: int | None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = [q for q in data if q["id"] >= start_id]
    if limit is not None:
        items = items[:limit]
    return items


def build_record(q: dict, sub_queries: list[str]) -> dict:
    return {
        "id": q["id"],
        "question": q["question"],
        "n_subqueries": len(sub_queries),
        "sub_queries": sub_queries if len(sub_queries) > 1 else None,
        "queries": sub_queries,
    }


def main() -> None:
    args = parse_args()
    embed_model = args.embed_model or get_embed_model_path()
    questions = load_questions(args.questions, args.start_id, args.limit)
    embed_tokenizer = load_embed_tokenizer(embed_model)

    records = []
    multi = 0
    for q in questions:
        queries = resolve_queries(
            q["question"],
            embed_tokenizer,
            threshold_1=args.token_threshold_1,
            threshold_2=args.token_threshold_2,
        )
        if len(queries) > 1:
            multi += 1
        records.append(build_record(q, queries))

    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done — {len(records)} records ({multi} split) → {args.output}")


if __name__ == "__main__":
    main()
