"""Shared Qdrant client configuration for local and Cloud deployments."""

from __future__ import annotations

import os
from pathlib import Path

from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "legal_documents"
DEFAULT_EMBED_MODEL = ROOT / "models" / "Vietnamese_Embedding"
VECTOR_SIZE = 1024
_ENV_LOADED = False


def resolve_env_path(env_file: str | Path | None = None) -> Path | None:
    if env_file is not None:
        path = Path(env_file)
        return path if path.is_file() else None
    override = os.environ.get("QDRANT_ENV_FILE")
    if override:
        path = Path(override)
        return path if path.is_file() else None
    default = ROOT / ".env"
    return default if default.is_file() else None


def load_env(env_file: str | Path | None = None, *, force: bool = False) -> None:
    """Load ROAD2AI/.env (or QDRANT_ENV_FILE) if python-dotenv is available."""
    global _ENV_LOADED
    if _ENV_LOADED and not force:
        return
    env_path = resolve_env_path(env_file)
    if not env_path:
        _ENV_LOADED = True
        return
    try:
        from dotenv import load_dotenv

        explicit = env_file is not None or os.environ.get("QDRANT_ENV_FILE")
        load_dotenv(env_path, override=bool(explicit))
    except ImportError:
        pass
    _ENV_LOADED = True


def get_qdrant_url(url: str | None = None) -> str:
    load_env()
    return url or os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)


def get_qdrant_api_key(api_key: str | None = None) -> str | None:
    load_env()
    return api_key or os.environ.get("QDRANT_API_KEY") or None


def get_collection_name(collection: str | None = None) -> str:
    load_env()
    return collection or os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION)


def get_vector_name(vector_name: str | None = None) -> str | None:
    """Named vector for multi-vector collections (e.g. Qdrant Cloud 'dense')."""
    load_env()
    resolved = vector_name or os.environ.get("QDRANT_VECTOR_NAME")
    return resolved or None


def get_sparse_vector_name(sparse_vector_name: str | None = None) -> str | None:
    """Named sparse vector for Qdrant Cloud BM25 (e.g. 'bm25')."""
    load_env()
    resolved = sparse_vector_name or os.environ.get("QDRANT_SPARSE_VECTOR_NAME")
    return resolved or None


def get_embed_model_path(path: Path | None = None) -> Path:
    load_env()
    if path is not None:
        p = Path(path)
        if p.is_dir() or str(path) != str(DEFAULT_EMBED_MODEL):
            return p
    env_path = os.environ.get("EMBED_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_dir() or "/" in env_path or env_path.startswith("BAAI"):
            return p
    legacy = ROOT / "models" / "Vietnamese_Embedding_v2"
    if legacy.is_dir():
        return legacy
    return DEFAULT_EMBED_MODEL


def get_rerank_model_path(path: Path | None = None) -> Path:
    load_env()
    default = ROOT / "models" / "Vietnamese_Reranker"
    if path is not None:
        p = Path(path)
        if p.is_dir() or str(path) != str(default):
            return p
    env_path = os.environ.get("RERANK_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_dir() or "/" in env_path or env_path.startswith("BAAI"):
            return p
    if default.is_dir():
        return default
    return Path("BAAI/bge-reranker-v2-m3")


def chunk_text(payload: dict | None) -> str:
    """Return the text field used for retrieval/reranking."""
    p = payload or {}
    return str(
        p.get("retrieval_text")
        or p.get("content_text")
        or p.get("text")
        or ""
    )


def normalize_chunk_payload(payload: dict | None) -> dict:
    """Normalize legacy local payloads to the Qdrant Cloud payload field names."""
    p = dict(payload or {})

    doc_id = p.get("document_id", p.get("doc_id", ""))
    if doc_id not in (None, ""):
        p.setdefault("document_id", doc_id)
    p.setdefault("document_number", str(p.get("law_code") or ""))
    p.setdefault("document_title", str(p.get("law_title") or ""))
    p.setdefault("legal_type", str(p.get("law_type") or ""))
    p.setdefault("article_no", str(p.get("article_number") or p.get("node_label") or ""))
    p.setdefault("retrieval_text", chunk_text(p))
    p.setdefault("source_url", str(p.get("url") or ""))
    p.setdefault("chunk_id", str(p.get("chunk_id") or p.get("file_name") or ""))
    return p


def query_dense(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    *,
    limit: int,
    vector_name: str | None = None,
    **kwargs,
):
    using = get_vector_name(vector_name)
    if using:
        return client.query_points(
            collection_name=collection,
            query=vector,
            using=using,
            limit=limit,
            **kwargs,
        )
    return client.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        **kwargs,
    )


def query_sparse_bm25(
    client: QdrantClient,
    collection: str,
    query_text: str,
    *,
    limit: int,
    sparse_vector_name: str | None = None,
    **kwargs,
):
    """BM25 sparse search via Qdrant Cloud (pre-indexed sparse vector)."""
    from qdrant_client import models

    using = get_sparse_vector_name(sparse_vector_name)
    if not using:
        raise ValueError("QDRANT_SPARSE_VECTOR_NAME chưa cấu hình (vd. bm25)")
    return client.query_points(
        collection_name=collection,
        query=models.Document(text=query_text, model="Qdrant/bm25"),
        using=using,
        limit=limit,
        **kwargs,
    )


def make_qdrant_client(
    url: str | None = None,
    api_key: str | None = None,
    *,
    timeout: float | None = None,
) -> QdrantClient:
    resolved_url = get_qdrant_url(url)
    resolved_key = get_qdrant_api_key(api_key)
    if timeout is None:
        timeout = 120.0 if resolved_key else 30.0
    if resolved_key:
        return QdrantClient(url=resolved_url, api_key=resolved_key, timeout=timeout)
    return QdrantClient(url=resolved_url, timeout=timeout)


def add_qdrant_args(parser) -> None:
    """Register common Qdrant CLI arguments on an argparse parser."""
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Load Qdrant settings from this .env file (default: ROAD2AI/.env or QDRANT_ENV_FILE)",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help=f"Qdrant server URL (default: env QDRANT_URL or {DEFAULT_QDRANT_URL})",
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=None,
        help="Qdrant API key (default: env QDRANT_API_KEY; not needed for local Qdrant)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=f"Qdrant collection name (default: env QDRANT_COLLECTION or {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--vector-name",
        default=None,
        help="Named dense vector (default: env QDRANT_VECTOR_NAME; omit for local legal_documents)",
    )
    parser.add_argument(
        "--sparse-vector-name",
        default=None,
        help="Named sparse BM25 vector on Qdrant Cloud (default: env QDRANT_SPARSE_VECTOR_NAME)",
    )


def init_qdrant_from_args(args) -> None:
    """Load env file from argparse namespace before resolving Qdrant settings."""
    env_file = getattr(args, "env_file", None)
    if env_file is not None:
        load_env(env_file, force=True)
