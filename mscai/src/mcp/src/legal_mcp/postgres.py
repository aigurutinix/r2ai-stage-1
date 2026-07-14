"""PostgreSQL lookup exact-match cho data layer, không dùng embedding."""
from __future__ import annotations

import json
from dataclasses import dataclass

from legal_mcp.config import PostgreSQLSettings
from legal_mcp.schemas import RetrievedCandidate, article_from_mapping


def quote_identifier(value: str) -> str:
    """Quote table/column identifier đã được validate trong config."""

    return ".".join(f'"{part}"' for part in value.split("."))


@dataclass(frozen=True)
class RelatedRef:
    """Reference đã parse từ chuỗi extra."""

    law_id: str
    law_name: str
    article: str
    doc_type: str | None = None


def parse_related_ref(reference: str) -> RelatedRef | None:
    """Parse ``doc_type|law_id|law_name|article`` hoặc format 3 phần cũ."""

    parts = [part.strip() for part in reference.split("|") if part.strip()]
    if len(parts) == 4:
        doc_type, law_id, law_name, article = parts
        return RelatedRef(doc_type=doc_type, law_id=law_id, law_name=law_name, article=article)
    if len(parts) == 3:
        law_id, law_name, article = parts
        return RelatedRef(law_id=law_id, law_name=law_name, article=article)
    return None


class PostgresLegalRepository:
    """Repository read-only để lấy điều luật liên quan từ PostgreSQL."""

    def __init__(self, settings: PostgreSQLSettings) -> None:
        self.settings = settings

    async def fetch_related(
        self,
        extra_refs: list[str],
        databases: list[str] | None,
        limit: int,
    ) -> list[RetrievedCandidate]:
        """Tìm các điều luật được liệt kê trong extra bằng exact match."""

        refs = [ref for ref in (parse_related_ref(item) for item in extra_refs) if ref is not None]
        if not refs:
            return []
        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover - phụ thuộc runtime MCP
            raise RuntimeError("Cần cài asyncpg để đọc PostgreSQL legal data") from exc

        conn = await asyncpg.connect(self.settings.database_url)
        try:
            has_database_column = await self._has_database_column(conn)
            rows = await conn.fetch(*self._build_query(
                refs=refs,
                databases=databases if has_database_column else None,
                limit=limit,
                use_database_column=has_database_column,
            ))
        finally:
            await conn.close()

        candidates: list[RetrievedCandidate] = []
        for rank, row in enumerate(rows, start=1):
            data = dict(row)
            if isinstance(data.get("extra"), str):
                data["extra"] = json.loads(data.get("extra") or "[]")
            article = article_from_mapping(data, score=1.0)
            candidates.append(RetrievedCandidate(article=article, source="related", score=1.0, rank=rank))
        return candidates

    async def _has_database_column(self, conn) -> bool:
        """Kiểm tra bảng có cột database hay không để filter an toàn."""

        if not self.settings.database_column:
            return False
        table = self.settings.table_name.split(".")[-1]
        row = await conn.fetchrow(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = $1 AND column_name = $2
            LIMIT 1
            """,
            table.strip('"'),
            self.settings.database_column,
        )
        return row is not None

    def _build_query(
        self,
        refs: list[RelatedRef],
        databases: list[str] | None,
        limit: int,
        use_database_column: bool,
    ):
        """Build SQL exact match theo law_id/law_name/article/doc_type."""

        table = quote_identifier(self.settings.table_name)
        database_column = (
            quote_identifier(self.settings.database_column)
            if self.settings.database_column and use_database_column
            else None
        )
        select_database = f", {database_column} AS database" if database_column else ", 'default' AS database"
        clauses: list[str] = []
        params: list[object] = []
        index = 1
        for ref in refs:
            local_clauses = [f"law_id = ${index}", f"law_name = ${index + 1}", f"article = ${index + 2}"]
            params.extend([ref.law_id, ref.law_name, ref.article])
            index += 3
            if ref.doc_type:
                local_clauses.append(f"doc_type = ${index}")
                params.append(ref.doc_type)
                index += 1
            clauses.append("(" + " AND ".join(local_clauses) + ")")
        where = " OR ".join(clauses)
        if databases and database_column:
            where = f"({where}) AND {database_column} = ANY(${index}::text[])"
            params.append(databases)
            index += 1
        params.append(limit)
        sql = f"""
            SELECT id, law_id, law_name, doc_type, chapter, article,
                   article_title, content, author, extra{select_database}
            FROM {table}
            WHERE {where}
            LIMIT ${index}
        """
        return sql, *params
