"""Tự động dựng Chroma từ PostgreSQL khi backend khởi động."""
from __future__ import annotations

import fcntl
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.config import Settings
from src.schemas.legal import LegalArticle
from src.services.embeddings.client import get_embeddings_client
from src.services.vector_store.base import VectorStoreRegistry, vector_store_registry
from src.services.vector_store.chroma import ChromaLegalStore, safe_collection_name
from src.services.vector_store.hybrid import HybridLegalStore
from src.services.vector_store.in_memory import InMemoryLegalStore, tokenizer_signature

logger = logging.getLogger("uvicorn.error")
_MANIFEST_NAME = "legal_index_manifest.json"
_BM25_CACHE_NAME = "legal_bm25_cache.json"
_LOCK_NAME = ".legal_index.lock"


def quote_identifier(value: str) -> str:
    """Quote table/column đã được validate bởi cấu hình Pydantic."""

    return ".".join(f'"{part}"' for part in value.split("."))


async def initialize_legal_index(
    settings: Settings,
    registry: VectorStoreRegistry = vector_store_registry,
) -> None:
    """Đồng bộ PostgreSQL -> Chroma/BM25 cache rồi đăng ký runtime stores.

    Chroma và BM25 đều có manifest riêng trong ``persist_directory``. Backend chỉ
    build lại khi nguồn PostgreSQL, embedding config hoặc cấu hình BM25 liên quan
    thay đổi. Khi manifest hợp lệ, BM25 được nạp từ corpus đã tokenize sẵn để
    tránh chạy tokenizer tiếng Việt ở mỗi lần startup.
    """

    postgres = settings.legal_assistant.postgres
    logger.info("[index] Bắt đầu chuẩn bị retrieval index")
    if not postgres.enabled:
        logger.info("Bỏ qua auto-index vì legal_assistant.postgres.enabled=false")
        return

    persist_directory = settings.legal_assistant.vector_store.persist_directory
    persist_directory.mkdir(parents=True, exist_ok=True)
    lock_path = persist_directory / _LOCK_NAME

    # File lock ngăn nhiều uvicorn worker cùng embedding lại một bộ dữ liệu.
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        logger.info("[index] Đang đọc legal records từ PostgreSQL")
        articles_by_space = await _fetch_articles(settings)
        if not articles_by_space:
            return
        total_records = sum(len(items) for items in articles_by_space.values())
        logger.info(
            "[index] Đã đọc %s record thuộc %s search space",
            total_records,
            len(articles_by_space),
        )
        expected_manifest = _manifest(settings, articles_by_space)
        vector = settings.legal_assistant.vector_store
        if vector.mode in {"chroma", "hybrid"}:
            manifest_path = persist_directory / _MANIFEST_NAME
            current_manifest = _read_manifest(manifest_path)
            rebuild = current_manifest != expected_manifest or not _collections_are_complete(
                settings,
                expected_manifest["search_spaces"],
            )
            if rebuild:
                logger.info("[index] Chroma chưa hợp lệ, bắt đầu embedding và rebuild")
                _rebuild_chroma(settings, articles_by_space)
                manifest_path.write_text(
                    json.dumps(expected_manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info("Đã build Chroma: %s records", expected_manifest["total_records"])
            else:
                logger.info("Chroma index hợp lệ, bỏ qua embedding lại")
        else:
            logger.info("[index] mode=bm25, bỏ qua Chroma/embedding")

        logger.info("[index] Bắt đầu đăng ký runtime stores")
        _register_runtime_stores(settings, registry, articles_by_space, expected_manifest)
        logger.info("[index] Hoàn tất chuẩn bị retrieval index")


async def _fetch_articles(settings: Settings) -> dict[str, list[LegalArticle]]:
    """Đọc toàn bộ record luật từ PostgreSQL vào một search space duy nhất."""

    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Cần cài asyncpg để tự động build Chroma từ PostgreSQL") from exc

    postgres = settings.legal_assistant.postgres
    conn = await asyncpg.connect(postgres.database_url)
    try:
        available_columns = await _get_columns(conn, postgres.table_name)
        required = {
            "id",
            "law_id",
            "law_name",
            "doc_type",
            "article",
            "article_title",
            "content",
            "author",
        }
        missing = required - available_columns
        if missing:
            raise RuntimeError(f"Bảng PostgreSQL thiếu cột bắt buộc: {sorted(missing)}")

        optional_selects = [
            quote_identifier(name) if name in available_columns else f"NULL AS {quote_identifier(name)}"
            for name in ("chapter", "extra")
        ]
        sql = f"""
            SELECT id, law_id, law_name, doc_type, article, article_title,
                   content, author, {', '.join(optional_selects)}
            FROM {quote_identifier(postgres.table_name)}
            ORDER BY id
        """
        rows = await conn.fetch(sql)
    finally:
        await conn.close()

    if not rows:
        logger.warning("PostgreSQL chưa có legal record; backend khởi động với index rỗng")
        return {}

    articles: list[LegalArticle] = []
    for row in rows:
        data = dict(row)
        data["id"] = str(data["id"])
        data["extra"] = _normalize_extra(data.get("extra"))
        articles.append(LegalArticle.model_validate(data))
    return {"default": articles}


async def _get_columns(conn: Any, table_name: str) -> set[str]:
    """Lấy danh sách cột để hỗ trợ schema không có chapter/extra."""

    parts = table_name.split(".")
    schema, table = (parts[0], parts[1]) if len(parts) == 2 else ("public", parts[0])
    rows = await conn.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = $1 AND table_name = $2
        """,
        schema,
        table,
    )
    return {str(row["column_name"]) for row in rows}


def _normalize_extra(value: Any) -> set[str]:
    """Chuẩn hóa extra từ JSON/array PostgreSQL về set string."""

    if value is None:
        return set()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {value} if value else set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if item}
    return set()


def _manifest(settings: Settings, grouped: dict[str, list[LegalArticle]]) -> dict[str, Any]:
    """Tạo dấu vân tay đủ để không trộn vector từ hai embedding model."""

    postgres = settings.legal_assistant.postgres
    parsed = urlparse(postgres.database_url)
    return {
        "version": 1,
        "postgres": {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip("/"),
            "table": postgres.table_name,
        },
        "embedding": {
            "base_url": settings.embeddings.base_url.rstrip("/"),
            "model": settings.embeddings.model,
        },
        "vector_text": "law_name\\narticle_title:content",
        "search_spaces": {name: len(items) for name, items in sorted(grouped.items())},
        "total_records": sum(len(items) for items in grouped.values()),
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    """Đọc manifest cũ; file lỗi được xem như chưa từng build."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _collections_are_complete(settings: Settings, search_spaces: dict[str, int]) -> bool:
    """Đối chiếu số vector trong Chroma với manifest PostgreSQL."""

    try:
        import chromadb

        vector = settings.legal_assistant.vector_store
        client = chromadb.PersistentClient(path=str(vector.persist_directory))
        for search_space, expected_count in search_spaces.items():
            name = safe_collection_name(f"{vector.default_collection}_{search_space}")
            if client.get_collection(name).count() != expected_count:
                return False
        return True
    except Exception:
        return False


def _rebuild_chroma(settings: Settings, grouped: dict[str, list[LegalArticle]]) -> None:
    """Xóa collection legal cũ rồi embed/upsert một search space theo batch."""

    import chromadb

    vector = settings.legal_assistant.vector_store
    postgres = settings.legal_assistant.postgres
    client = chromadb.PersistentClient(path=str(vector.persist_directory))
    prefix = safe_collection_name(vector.default_collection)
    for collection in client.list_collections():
        name = collection if isinstance(collection, str) else collection.name
        if name == prefix or name.startswith(f"{prefix}_"):
            client.delete_collection(name)

    embeddings = get_embeddings_client()
    total_spaces = len(grouped)
    processed_records = 0
    for space_index, (search_space, articles) in enumerate(grouped.items(), start=1):
        logger.info(
            "[index][Chroma %s/%s] %s: %s record",
            space_index,
            total_spaces,
            search_space,
            len(articles),
        )
        store = ChromaLegalStore(
            database=search_space,
            persist_directory=str(vector.persist_directory),
            collection_prefix=vector.default_collection,
            embeddings=embeddings,
        )
        for start in range(0, len(articles), postgres.batch_size):
            batch = articles[start : start + postgres.batch_size]
            store.add_articles(batch)
            processed_records += len(batch)
            logger.info(
                "[index][Chroma] Đã embedding %s/%s record",
                processed_records,
                sum(len(items) for items in grouped.values()),
            )


def _register_runtime_stores(
    settings: Settings,
    registry: VectorStoreRegistry,
    grouped: dict[str, list[LegalArticle]],
    index_manifest: dict[str, Any],
) -> None:
    """Đăng ký Chroma/BM25 store cho một search space trong process hiện tại."""

    vector = settings.legal_assistant.vector_store
    chroma_required = vector.mode in {"chroma", "hybrid"}
    bm25_required = vector.mode in {"bm25", "hybrid"}
    embeddings = get_embeddings_client() if chroma_required else None
    bm25_stores = _load_or_build_bm25_stores(settings, grouped, index_manifest) if bm25_required else {}

    total_spaces = len(grouped)
    for space_index, (search_space, articles) in enumerate(grouped.items(), start=1):
        logger.info(
            "[index][Runtime %s/%s] %s: đăng ký %s record",
            space_index,
            total_spaces,
            search_space,
            len(articles),
        )
        chroma_store = None
        if chroma_required:
            chroma_store = ChromaLegalStore(
                database=search_space,
                persist_directory=str(vector.persist_directory),
                collection_prefix=vector.default_collection,
                embeddings=embeddings,
            )
        if vector.mode == "chroma":
            assert chroma_store is not None
            registry.register(search_space, chroma_store)
            continue

        lexical_store = bm25_stores[search_space]
        if vector.mode == "bm25":
            registry.register(search_space, lexical_store)
        else:
            assert chroma_store is not None
            registry.register(
                search_space,
                HybridLegalStore(
                    lexical_store,
                    chroma_store,
                    rrf_k=vector.rrf_k,
                    dense_weight=vector.dense_weight,
                    bm25_weight=vector.bm25_weight,
                ),
            )


def _load_or_build_bm25_stores(
    settings: Settings,
    grouped: dict[str, list[LegalArticle]],
    index_manifest: dict[str, Any],
) -> dict[str, InMemoryLegalStore]:
    """Nạp BM25 từ cache; nếu cache lệch manifest thì tokenize và ghi lại."""

    vector = settings.legal_assistant.vector_store
    cache_path = vector.persist_directory / _BM25_CACHE_NAME
    expected_manifest = _bm25_manifest(settings, index_manifest)
    cached = _read_manifest(cache_path)
    if cached and cached.get("manifest") == expected_manifest:
        stores = _bm25_stores_from_cache(settings, grouped, cached)
        if stores is not None:
            logger.info("[index] BM25 cache hợp lệ, nạp từ %s", cache_path)
            return stores
        logger.warning("[index] BM25 cache lỗi cấu trúc, sẽ build lại")

    logger.info("[index] BM25 cache chưa hợp lệ, bắt đầu tokenize/build một lần")
    stores = _build_bm25_stores(settings, grouped)
    cache = {
        "manifest": expected_manifest,
        "search_spaces": {
            search_space: {
                "article_ids": [article.article_id for article in grouped[search_space]],
                "tokenized_corpus": store.export_tokenized_corpus(),
            }
            for search_space, store in stores.items()
        },
    }
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    logger.info("[index] Đã ghi BM25 cache: %s", cache_path)
    return stores


def _build_bm25_stores(
    settings: Settings,
    grouped: dict[str, list[LegalArticle]],
) -> dict[str, InMemoryLegalStore]:
    """Tokenize records và dựng BM25 store theo một search space."""

    vector = settings.legal_assistant.vector_store
    stores: dict[str, InMemoryLegalStore] = {}
    total_spaces = len(grouped)
    for space_index, (search_space, articles) in enumerate(grouped.items(), start=1):
        logger.info(
            "[index][BM25 %s/%s] %s: tokenize %s record",
            space_index,
            total_spaces,
            search_space,
            len(articles),
        )
        store = InMemoryLegalStore(
            database=search_space,
            tokenizer=vector.bm25_tokenizer,
            k1=vector.bm25_k1,
            b=vector.bm25_b,
            epsilon=vector.bm25_epsilon,
        )
        store.add_articles(articles)
        stores[search_space] = store
    return stores


def _bm25_stores_from_cache(
    settings: Settings,
    grouped: dict[str, list[LegalArticle]],
    cache: dict[str, Any],
) -> dict[str, InMemoryLegalStore] | None:
    """Khôi phục BM25 store từ tokenized corpus đã lưu."""

    vector = settings.legal_assistant.vector_store
    search_spaces = cache.get("search_spaces")
    if not isinstance(search_spaces, dict):
        return None

    stores: dict[str, InMemoryLegalStore] = {}
    for search_space, articles in grouped.items():
        payload = search_spaces.get(search_space)
        if not isinstance(payload, dict):
            return None
        article_ids = payload.get("article_ids")
        tokenized_corpus = payload.get("tokenized_corpus")
        if article_ids != [article.article_id for article in articles]:
            return None
        if not isinstance(tokenized_corpus, list) or len(tokenized_corpus) != len(articles):
            return None
        stores[search_space] = InMemoryLegalStore.from_indexed_articles(
            database=search_space,
            articles=articles,
            tokenized_corpus=tokenized_corpus,
            tokenizer=vector.bm25_tokenizer,
            k1=vector.bm25_k1,
            b=vector.bm25_b,
            epsilon=vector.bm25_epsilon,
        )
    return stores


def _bm25_manifest(settings: Settings, index_manifest: dict[str, Any]) -> dict[str, Any]:
    """Manifest riêng cho BM25 để không phụ thuộc vào manifest Chroma."""

    vector = settings.legal_assistant.vector_store
    return {
        "version": 1,
        "source_index": index_manifest,
        "bm25": {
            "tokenizer": vector.bm25_tokenizer,
            "tokenizer_impl": tokenizer_signature(vector.bm25_tokenizer),
            "k1": vector.bm25_k1,
            "b": vector.bm25_b,
            "epsilon": vector.bm25_epsilon,
        },
    }
