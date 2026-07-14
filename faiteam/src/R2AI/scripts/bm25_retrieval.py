"""BM25 sparse retrieval and hybrid fusion with dense (vector) search."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from qdrant_config import chunk_text, normalize_chunk_payload

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

DEFAULT_WEIGHT_RRF_DENSE = 0.4
DEFAULT_WEIGHT_RRF_BM25 = 0.6


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if len(t) > 1]


def chunk_key(payload: dict[str, Any]) -> str:
    doc_id = str(payload.get("document_id", payload.get("doc_id", "")))
    chunk_id = str(payload.get("chunk_id", ""))
    if doc_id or chunk_id:
        return f"{doc_id}:{chunk_id}"
    return str(payload.get("point_id", ""))


def payload_from_qdrant(point) -> dict[str, Any]:
    payload = normalize_chunk_payload(point.payload or {})
    payload["point_id"] = str(point.id)
    return payload


def scroll_all_chunks(client: QdrantClient, collection: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = payload_from_qdrant(point)
            text = chunk_text(payload).strip()
            if text:
                chunks.append(payload)
        if offset is None:
            break
    return chunks


def load_or_build_bm25_index(
    client: QdrantClient,
    collection: str,
    cache_path: Path | None,
) -> tuple[BM25Okapi, list[dict[str, Any]]]:
    if cache_path and cache_path.is_file():
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        if cached.get("collection") == collection:
            print(f"  BM25 cache: {cache_path} ({len(cached['corpus'])} chunks)")
            return cached["bm25"], cached["corpus"]

    print(f"  Building BM25 index from Qdrant collection '{collection}'...")
    corpus = scroll_all_chunks(client, collection)
    if not corpus:
        raise RuntimeError(f"Không có chunk trong collection '{collection}'")

    tokenized = [tokenize(chunk_text(c)) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    print(f"  BM25 index: {len(corpus)} documents")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump({"collection": collection, "corpus": corpus, "bm25": bm25}, f)
        print(f"  Saved BM25 cache: {cache_path}")

    return bm25, corpus


def bm25_top_k(
    bm25: BM25Okapi,
    corpus: list[dict[str, Any]],
    query: str,
    top_k: int,
) -> list[tuple[dict[str, Any], float]]:
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(corpus[i], float(score)) for i, score in ranked if score > 0]


def weighted_reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float | None]]],
    *,
    weights: list[float] | None = None,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked lists with optional per-list weights (weightRRF)."""
    fused: dict[str, float] = {}
    for idx, ranked in enumerate(ranked_lists):
        weight = 1.0 if weights is None else weights[idx]
        for rank, (key, _) in enumerate(ranked, start=1):
            fused[key] = fused.get(key, 0.0) + weight / (rrf_k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float | None]]],
    *,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Equal-weight RRF for merging same-type ranked lists (e.g. sub-queries)."""
    return weighted_reciprocal_rank_fusion(ranked_lists, rrf_k=rrf_k)


def chunk_record(
    payload: dict[str, Any],
    *,
    rank: int,
    score: float,
    dense_score: float | None = None,
    bm25_score: float | None = None,
    rerank_score: float | None = None,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": score,
        "rrf_score": score,
        "dense_score": dense_score,
        "bm25_score": bm25_score,
        "rerank_score": rerank_score,
        "point_id": str(payload.get("point_id", "")),
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


def chunk_key_from_record(chunk: dict[str, Any]) -> str:
    doc_id = str(chunk.get("document_id", chunk.get("doc_id", "")))
    chunk_id = str(chunk.get("chunk_id", ""))
    if doc_id or chunk_id:
        return f"{doc_id}:{chunk_id}"
    point_id = str(chunk.get("point_id", ""))
    if point_id:
        return point_id
    return f"{chunk.get('document_number', '')}:{chunk.get('source_url', '')}:{chunk.get('article_no', '')}"


def merge_hybrid_results(
    chunk_lists: list[list[dict[str, Any]]],
    *,
    rrf_k: int = 60,
    top_k: int,
) -> list[dict[str, Any]]:
    """RRF-merge multiple hybrid retrieval result lists into one ranked list."""
    if not chunk_lists:
        return []
    if len(chunk_lists) == 1:
        return chunk_lists[0][:top_k]

    ranked_lists: list[list[tuple[str, float | None]]] = []
    chunk_by_key: dict[str, dict[str, Any]] = {}

    for chunks in chunk_lists:
        ranked: list[tuple[str, float | None]] = []
        for chunk in chunks:
            key = chunk_key_from_record(chunk)
            chunk_by_key[key] = chunk
            ranked.append((key, chunk.get("score")))
        if ranked:
            ranked_lists.append(ranked)

    if not ranked_lists:
        return []

    fused = reciprocal_rank_fusion(ranked_lists, rrf_k=rrf_k)[:top_k]
    merged: list[dict[str, Any]] = []
    for rank, (key, fused_score) in enumerate(fused, start=1):
        source = chunk_by_key[key]
        merged.append(
            chunk_record(
                source,
                rank=rank,
                score=fused_score,
                dense_score=source.get("dense_score"),
                bm25_score=source.get("bm25_score"),
                rerank_score=source.get("rerank_score"),
            )
        )
    return merged


def hybrid_retrieve_one(
    query: str,
    *,
    dense_hits: list,
    top_k: int,
    pool_size: int,
    rrf_k: int,
    dense_weight: float = DEFAULT_WEIGHT_RRF_DENSE,
    bm25_weight: float = DEFAULT_WEIGHT_RRF_BM25,
    bm25: BM25Okapi | None = None,
    corpus: list[dict[str, Any]] | None = None,
    sparse_hits: list | None = None,
) -> list[dict[str, Any]]:
    dense_ranked: list[tuple[str, float | None]] = []
    payload_by_key: dict[str, dict[str, Any]] = {}
    dense_score_by_key: dict[str, float] = {}

    for hit in dense_hits[:pool_size]:
        payload = normalize_chunk_payload(hit.payload or {})
        payload["point_id"] = str(hit.id)
        key = chunk_key(payload)
        payload_by_key[key] = payload
        dense_score_by_key[key] = float(hit.score)
        dense_ranked.append((key, hit.score))

    bm25_ranked: list[tuple[str, float | None]] = []
    bm25_score_by_key: dict[str, float] = {}

    if sparse_hits is not None:
        for hit in sparse_hits[:pool_size]:
            payload = normalize_chunk_payload(hit.payload or {})
            payload["point_id"] = str(hit.id)
            key = chunk_key(payload)
            payload_by_key[key] = payload
            score = float(hit.score)
            bm25_score_by_key[key] = score
            bm25_ranked.append((key, score))
    elif bm25 is not None and corpus is not None:
        bm25_hits = bm25_top_k(bm25, corpus, query, pool_size)
        for payload, score in bm25_hits:
            key = chunk_key(payload)
            payload_by_key[key] = payload
            bm25_score_by_key[key] = score
            bm25_ranked.append((key, score))
    else:
        raise ValueError("Cần sparse_hits (Qdrant BM25) hoặc bm25+corpus (local index)")

    ranked_lists = [dense_ranked]
    fusion_weights = [dense_weight]
    if bm25_ranked:
        ranked_lists.append(bm25_ranked)
        fusion_weights.append(bm25_weight)
    fused = weighted_reciprocal_rank_fusion(
        ranked_lists,
        weights=fusion_weights,
        rrf_k=rrf_k,
    )[:top_k]

    chunks: list[dict[str, Any]] = []
    for rank, (key, fused_score) in enumerate(fused, start=1):
        payload = payload_by_key[key]
        chunks.append(
            chunk_record(
                payload,
                rank=rank,
                score=fused_score,
                dense_score=dense_score_by_key.get(key),
                bm25_score=bm25_score_by_key.get(key),
            )
        )
    return chunks
