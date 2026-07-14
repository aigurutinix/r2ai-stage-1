"""Cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder

from qdrant_config import chunk_text, normalize_chunk_payload

MAX_RERANK_LENGTH = 2304
DEFAULT_PRIMARY_WEIGHT = 0.7
DEFAULT_SUB_WEIGHT = 0.3


def load_reranker(model_path: str | Path, device: str = "cuda") -> CrossEncoder:
    return CrossEncoder(str(model_path), max_length=MAX_RERANK_LENGTH, device=device)


def dense_hits_to_chunks(hits) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        payload = normalize_chunk_payload(hit.payload or {})
        payload["point_id"] = str(hit.id)
        chunks.append(
            {
                "rank": rank,
                "score": float(hit.score),
                "rrf_score": None,
                "dense_score": float(hit.score),
                "bm25_score": None,
                "rerank_score": None,
                "point_id": str(hit.id),
                "document_id": str(payload.get("document_id", "")),
                "document_number": payload.get("document_number", ""),
                "document_title": payload.get("document_title", ""),
                "legal_type": payload.get("legal_type", ""),
                "chunk_id": str(payload.get("chunk_id", "")),
                "article_no": payload.get("article_no", ""),
                "node_label": payload.get("node_label", ""),
                "source_article_no_candidates": payload.get("source_article_no_candidates") or [],
                "source_url": payload.get("source_url", ""),
                "retrieval_text": chunk_text(payload),
                "content_text": payload.get("content_text", ""),
            }
        )
    return chunks


def _predict_scores(
    reranker: CrossEncoder,
    queries: list[str],
    chunks: list[dict[str, Any]],
    *,
    batch_size: int,
) -> list[list[float]]:
    """Return score matrix [chunk_idx][query_idx]."""
    if not queries or not chunks:
        return []

    pairs: list[tuple[str, str]] = []
    index_map: list[tuple[int, int]] = []
    for ci, chunk in enumerate(chunks):
        text = chunk_text(chunk)
        for qi, query in enumerate(queries):
            pairs.append((query, text))
            index_map.append((ci, qi))

    raw = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    matrix = [[0.0] * len(queries) for _ in chunks]
    for (ci, qi), score in zip(index_map, raw):
        matrix[ci][qi] = float(score)
    return matrix


def _attach_rerank_fields(
    chunk: dict[str, Any],
    *,
    rank: int,
    final_score: float,
    primary_score: float,
    sub_mean: float,
    sub_scores: list[float],
) -> dict[str, Any]:
    out = dict(chunk)
    if out.get("rrf_score") is None and out.get("bm25_score") is not None:
        out["rrf_score"] = out.get("score")
    out["rerank_primary_score"] = float(primary_score)
    out["rerank_sub_scores"] = [float(s) for s in sub_scores]
    out["rerank_sub_mean"] = float(sub_mean)
    out["rerank_score"] = float(final_score)
    out["score"] = float(final_score)
    out["rank"] = rank
    return out


def rerank_chunks_hybrid(
    primary_query: str,
    sub_queries: list[str],
    chunks: list[dict[str, Any]],
    reranker: CrossEncoder,
    *,
    top_k: int,
    batch_size: int = 32,
    primary_weight: float = DEFAULT_PRIMARY_WEIGHT,
    sub_weight: float = DEFAULT_SUB_WEIGHT,
) -> list[dict[str, Any]]:
    """Score chunks with primary query + sub-queries.

    final = primary_weight * score(primary, chunk) + sub_weight * mean(score(sub_i, chunk))
    When there are no sub-queries, final = score(primary, chunk).
    """
    if not chunks:
        return []

    sub_queries = [q.strip() for q in sub_queries if q and q.strip()]
    queries = [primary_query.strip()] + sub_queries
    matrix = _predict_scores(reranker, queries, chunks, batch_size=batch_size)

    ranked: list[tuple[dict[str, Any], float, float, float, list[float]]] = []
    for chunk, scores in zip(chunks, matrix):
        primary_score = scores[0]
        sub_scores = scores[1:]
        sub_mean = sum(sub_scores) / len(sub_scores) if sub_scores else 0.0
        if sub_queries:
            final = primary_weight * primary_score + sub_weight * sub_mean
        else:
            final = primary_score
        ranked.append((chunk, final, primary_score, sub_mean, sub_scores))

    ranked.sort(key=lambda x: x[1], reverse=True)
    result: list[dict[str, Any]] = []
    for rank, (chunk, final, primary_score, sub_mean, sub_scores) in enumerate(
        ranked[:top_k], start=1
    ):
        result.append(
            _attach_rerank_fields(
                chunk,
                rank=rank,
                final_score=final,
                primary_score=primary_score,
                sub_mean=sub_mean,
                sub_scores=sub_scores,
            )
        )
    return result


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    reranker: CrossEncoder,
    *,
    top_k: int,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    pairs = [(query, chunk_text(c)) for c in chunks]
    scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    ranked = sorted(zip(chunks, scores), key=lambda x: float(x[1]), reverse=True)[:top_k]

    result: list[dict[str, Any]] = []
    for rank, (chunk, score) in enumerate(ranked, start=1):
        result.append(
            _attach_rerank_fields(
                chunk,
                rank=rank,
                final_score=float(score),
                primary_score=float(score),
                sub_mean=0.0,
                sub_scores=[],
            )
        )
    return result
