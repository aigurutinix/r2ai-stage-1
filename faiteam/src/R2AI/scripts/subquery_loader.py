"""Load precomputed sub-query specs for retrieval and hybrid rerank."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubquerySpec:
    original_question: str
    queries: list[str]
    num_queries: int


def load_subquery_index(path: Path | None) -> dict[int, SubquerySpec]:
    if path is None or not path.is_file():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    index: dict[int, SubquerySpec] = {}
    for item in data:
        qid = item["id"]
        original = str(
            item.get("original_question") or item.get("question") or ""
        ).strip()
        queries = [str(q).strip() for q in item.get("queries") or [] if str(q).strip()]
        if not original and queries:
            original = queries[0]
        if not queries and original:
            queries = [original]
        num = int(item.get("num_queries") or len(queries) or 1)
        index[qid] = SubquerySpec(
            original_question=original,
            queries=queries,
            num_queries=num,
        )
    return index


def retrieval_queries(spec: SubquerySpec | None, fallback_question: str) -> list[str]:
    """Queries used for dense/BM25/RRF retrieval."""
    if spec is None:
        return [fallback_question]
    if spec.queries:
        return list(spec.queries)
    return [spec.original_question or fallback_question]


def rerank_primary_and_subs(
    spec: SubquerySpec | None,
    fallback_question: str,
) -> tuple[str, list[str]]:
    """Primary = original question; sub-queries = generated queries when num_queries > 1."""
    if spec is None:
        return fallback_question, []
    primary = spec.original_question or fallback_question
    if spec.num_queries > 1:
        return primary, list(spec.queries)
    return primary, []


def subquery_map_from_index(index: dict[int, SubquerySpec]) -> dict[int, list[str]]:
    return {qid: retrieval_queries(spec, spec.original_question) for qid, spec in index.items()}
