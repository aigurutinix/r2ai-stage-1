"""Client gọi Qwen3 reranker qua vLLM ``/v1/rerank`` endpoint."""
from __future__ import annotations

from typing import Any

import httpx

from src.config import settings
from src.schemas.legal import LegalArticle


class RerankerClient:
    """Client mỏng cho Qwen3 reranker chạy sau retrieval.

    Endpoint thực tế nhận payload dạng::

        {"model": name, "query": query, "documents": [...], "top_n": n}

    và trả danh sách kết quả có ``index`` + ``score``/``relevance_score``.
    Client giữ một ``httpx.AsyncClient`` để tái sử dụng TCP connection, tránh tạo
    connection mới cho từng request chat.
    """

    def __init__(self) -> None:
        reranker = settings.legal_assistant.reranker
        self.enabled = reranker.enabled
        self.model = reranker.model
        self.base_url = reranker.base_url.rstrip("/")
        self.endpoint = reranker.endpoint if reranker.endpoint.startswith("/") else f"/{reranker.endpoint}"
        self.api_key = reranker.api_key
        self.timeout = reranker.timeout_seconds
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )

    async def score_many(self, query: str, articles: list[LegalArticle]) -> list[float]:
        """Chấm relevance cho nhiều điều luật trong một request batch."""

        if not articles:
            return []
        payload = {
            "model": self.model,
            "query": query,
            "documents": [article.vector_text for article in articles],
            "top_n": len(articles),
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = await self._client.post(f"{self.base_url}{self.endpoint}", json=payload, headers=headers)
        response.raise_for_status()
        return _extract_scores(response.json(), expected_count=len(articles))

    async def aclose(self) -> None:
        """Đóng HTTP client khi cần shutdown thủ công."""

        await self._client.aclose()


_reranker_client: RerankerClient | None = None


def get_reranker_client() -> RerankerClient:
    """Khởi tạo reranker client một lần trong process backend."""

    global _reranker_client
    if _reranker_client is None:
        _reranker_client = RerankerClient()
    return _reranker_client


def _extract_scores(payload: Any, expected_count: int) -> list[float]:
    """Parse scores từ response ``/v1/rerank`` của vLLM reranker."""

    scores: list[float | None] = [None] * expected_count
    data = _result_items(payload)
    for position, item in enumerate(data):
        if isinstance(item, (int, float)):
            index = position
            score = float(item)
        elif isinstance(item, dict):
            raw_index = item.get("index", position)
            raw_score = item.get("score", item.get("relevance_score"))
            if not isinstance(raw_index, int) or not isinstance(raw_score, (int, float)):
                raise RuntimeError(f"Item reranker không hợp lệ: {item}")
            index = raw_index
            score = float(raw_score)
        else:
            raise RuntimeError(f"Item reranker không hợp lệ: {item}")
        if 0 <= index < expected_count:
            scores[index] = score

    missing = [index for index, score in enumerate(scores) if score is None]
    if missing:
        raise RuntimeError(f"Response reranker thiếu score cho document index: {missing}")
    return [float(score) for score in scores]


def _result_items(payload: Any) -> list[Any]:
    """Lấy list result từ các field phổ biến của endpoint rerank."""

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise RuntimeError(f"Response reranker không có list results/data: {payload}")
