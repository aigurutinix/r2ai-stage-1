"""Vector retriever đọc Chroma index đã được backend/data system build sẵn."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from legal_mcp.config import LegalServerSettings
from legal_mcp.schemas import RetrievedCandidate, article_from_mapping

_NON_WORD_RE = re.compile(r"[^\w]+")


def safe_collection_name(value: str) -> str:
    """Đổi tên database thành tên collection hợp lệ cho Chroma."""

    max_length = 60
    name = _NON_WORD_RE.sub("_", value.lower()).strip("._-")
    if len(name) > max_length:
        suffix = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
        prefix = name[: max_length - len(suffix) - 1].strip("._-")
        name = f"{prefix}_{suffix}".strip("._-")
    if len(name) < 3:
        return "legal_articles"
    return name


class ChromaRetriever:
    """Retriever đọc Chroma persistent index và embedding query."""

    def __init__(self, settings: LegalServerSettings) -> None:
        try:
            import chromadb
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover - phụ thuộc runtime MCP
            raise RuntimeError("Cần cài chromadb và langchain-openai để search vector") from exc

        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(settings.vector_store.persist_directory))
        self.embeddings = OpenAIEmbeddings(
            model=settings.embeddings.model,
            base_url=settings.embeddings.base_url,
            api_key=settings.embeddings.api_key,
        )

    def search(self, query: str, databases: list[str], top_k: int) -> list[RetrievedCandidate]:
        """Search từng collection theo database và merge theo score."""

        embedding = self.embeddings.embed_query(query)
        candidates: list[RetrievedCandidate] = []
        per_database_k = max(top_k, 1)
        for database in databases or ["default"]:
            collection = self.client.get_or_create_collection(
                name=safe_collection_name(f"{self.settings.vector_store.collection_prefix}_{database}"),
                metadata={"hnsw:space": "cosine"},
            )
            result = collection.query(
                query_embeddings=[embedding],
                n_results=per_database_k,
                include=["metadatas", "distances"],
            )
            candidates.extend(self._to_candidates(result, database))
        candidates.sort(key=lambda item: item.score, reverse=True)
        for rank, candidate in enumerate(candidates[:top_k], start=1):
            candidate.rank = rank
        return candidates[:top_k]

    def _to_candidates(self, result: dict[str, Any], database: str) -> list[RetrievedCandidate]:
        """Chuyển output Chroma thành candidates chuẩn MCP."""

        candidates: list[RetrievedCandidate] = []
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for article_id, metadata, distance in zip(ids, metadatas, distances):
            score = 1.0 / (1.0 + float(distance or 0.0))
            data = dict(metadata or {})
            data.setdefault("id", article_id)
            data.setdefault("article_id", article_id)
            data.setdefault("database", database)
            if isinstance(data.get("extra"), str):
                data["extra"] = json.loads(data.get("extra") or "[]")
            article = article_from_mapping(data, default_database=database, score=score)
            candidates.append(RetrievedCandidate(article=article, source="vector", score=score))
        return candidates
