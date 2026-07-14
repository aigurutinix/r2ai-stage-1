"""Schema dữ liệu trao đổi qua MCP tools."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field


class LegalArticle(BaseModel):
    """Metadata/content một điều luật trả về cho backend agent."""

    id: str
    article_id: str | None = None
    law_id: str
    law_name: str
    doc_type: str
    database: str = "default"
    chapter: str | None = None
    article: str
    article_title: str | None = None
    content: str
    author: str | None = None
    extra: list[str] = Field(default_factory=list)
    score: float | None = None

    @computed_field
    @property
    def normalized_article_id(self) -> str:
        """Id dùng cho dedup nếu article_id không có trong DB."""

        return self.article_id or self.id


class RetrievedCandidate(BaseModel):
    """Một candidate trả về từ MCP tool."""

    article: LegalArticle
    source: Literal["vector", "related"] = "vector"
    score: float = 0.0
    rank: int | None = None
    reason: str | None = None


def article_from_mapping(data: dict[str, Any], default_database: str = "default", score: float | None = None) -> LegalArticle:
    """Chuẩn hóa dict từ Chroma/PostgreSQL thành LegalArticle."""

    extra = data.get("extra") or []
    if isinstance(extra, str):
        import json

        try:
            extra = json.loads(extra)
        except json.JSONDecodeError:
            extra = [item.strip() for item in extra.split(";") if item.strip()]
    return LegalArticle(
        id=str(data.get("id") or data.get("article_id") or ""),
        article_id=str(data.get("article_id") or data.get("id") or ""),
        law_id=str(data.get("law_id") or ""),
        law_name=str(data.get("law_name") or ""),
        doc_type=str(data.get("doc_type") or ""),
        database=str(data.get("database") or default_database),
        chapter=str(data.get("chapter") or "") or None,
        article=str(data.get("article") or ""),
        article_title=str(data.get("article_title") or "") or None,
        content=str(data.get("content") or ""),
        author=str(data.get("author") or "") or None,
        extra=list(extra),
        score=score,
    )
