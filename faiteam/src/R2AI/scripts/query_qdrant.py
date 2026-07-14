#!/usr/bin/env python3
"""Test semantic search on the legal_documents Qdrant collection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qdrant_config import (
    ROOT,
    add_qdrant_args,
    chunk_text,
    get_collection_name,
    get_embed_model_path,
    get_sparse_vector_name,
    init_qdrant_from_args,
    make_qdrant_client,
    query_dense,
    query_sparse_bm25,
)
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bm25_retrieval import hybrid_retrieve_one, load_or_build_bm25_index
from rerank_retrieval import dense_hits_to_chunks, load_reranker, rerank_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query in Vietnamese")
    add_qdrant_args(parser)
    parser.add_argument("--limit", type=int, default=30, help="Final chunks after rerank")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Embedding model path (default: EMBED_MODEL_PATH or ROAD2AI/models/Vietnamese_Embedding)",
    )
    parser.add_argument(
        "--use-bm25",
        action="store_true",
        help="Hybrid retrieval: dense + BM25 (weightRRF: dense=0.4, BM25=0.6)",
    )
    parser.add_argument(
        "--bm25-cache",
        type=Path,
        default=ROOT / "output" / "bm25_corpus.pkl",
    )
    parser.add_argument("--retrieve-pool", type=int, default=50)
    parser.add_argument("--rrf-top-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--rerank-model",
        type=Path,
        default=ROOT / "models" / "Vietnamese_Reranker",
    )
    parser.add_argument("--device-rerank", default="cuda")
    parser.add_argument("--rerank-batch", type=int, default=32)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()
    init_qdrant_from_args(args)
    collection = get_collection_name(args.collection)

    client = make_qdrant_client(args.qdrant_url, args.qdrant_api_key)
    pool_size = args.retrieve_pool
    use_rerank = not args.no_rerank
    fusion_top_k = args.rrf_top_k if (args.use_bm25 and use_rerank) else args.limit
    ann_limit = pool_size if (args.use_bm25 or use_rerank) else args.limit

    model_path = get_embed_model_path(args.model_path)
    model = SentenceTransformer(str(model_path))
    model.max_seq_length = 2048
    vector = model.encode([args.query], normalize_embeddings=True)[0].tolist()
    hits = query_dense(
        client,
        collection,
        vector,
        limit=ann_limit,
        vector_name=args.vector_name,
    )

    if args.use_bm25:
        sparse_name = get_sparse_vector_name(getattr(args, "sparse_vector_name", None))
        sparse_hits = None
        bm25 = corpus = None
        if sparse_name:
            sparse_res = query_sparse_bm25(
                client,
                collection,
                args.query,
                limit=pool_size,
                sparse_vector_name=sparse_name,
            )
            sparse_hits = sparse_res.points
            mode = f"hybrid (Qdrant {sparse_name})"
        else:
            bm25, corpus = load_or_build_bm25_index(client, collection, args.bm25_cache)
            mode = "hybrid (local BM25)"
        chunks = hybrid_retrieve_one(
            args.query,
            dense_hits=hits.points,
            bm25=bm25,
            corpus=corpus,
            sparse_hits=sparse_hits,
            top_k=fusion_top_k,
            pool_size=pool_size,
            rrf_k=args.rrf_k,
        )
    else:
        chunks = dense_hits_to_chunks(hits.points)
        mode = "dense"

    if use_rerank:
        reranker = load_reranker(args.rerank_model, args.device_rerank)
        chunks = rerank_chunks(
            args.query,
            chunks,
            reranker,
            top_k=args.limit,
            batch_size=args.rerank_batch,
        )
        mode += "+rerank"

    else:
        chunks = chunks[: args.limit]

    print(f"Query ({mode}): {args.query}\n")
    for c in chunks:
        print(
            f"--- #{c['rank']} score={c['score']:.4f} "
            f"rerank={c.get('rerank_score')} rrf={c.get('rrf_score')} "
            f"dense={c.get('dense_score')} bm25={c.get('bm25_score')} ---"
        )
        print(f"  {c.get('legal_type', '')} {c.get('document_number', '')}")
        print(f"  {c.get('document_title', '')}")
        print(f"  {c.get('article_no', '')} {c.get('chunk_id', '')}")
        text = chunk_text(c)
        print(f"  {text[:300]}{'...' if len(text) > 300 else ''}")
        print()


if __name__ == "__main__":
    main()
