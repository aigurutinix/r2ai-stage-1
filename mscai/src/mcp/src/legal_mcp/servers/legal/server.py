"""MCP package giữ logic database/vector cho dữ liệu pháp luật."""
from __future__ import annotations

from legal_mcp.config import LegalServerSettings, ServerConfig
from legal_mcp.core.server import BaseMCPServer
from legal_mcp.postgres import PostgresLegalRepository
from legal_mcp.vector import ChromaRetriever


class LegalRetrievalServer(BaseMCPServer):
    """Server shell chỉ khởi tạo database/vector logic, chưa expose tools."""

    def __init__(self, name: str, config: ServerConfig) -> None:
        self.settings = LegalServerSettings.model_validate(config.settings)
        self._vector_retriever: ChromaRetriever | None = None
        self._postgres_repository: PostgresLegalRepository | None = None
        super().__init__(name=name, config=config)

    @property
    def vector_retriever(self) -> ChromaRetriever:
        """Lazy init Chroma retriever."""

        if self._vector_retriever is None:
            self._vector_retriever = ChromaRetriever(self.settings)
        return self._vector_retriever

    @property
    def postgres_repository(self) -> PostgresLegalRepository:
        """Lazy init PostgreSQL repository read-only."""

        if self._postgres_repository is None:
            self._postgres_repository = PostgresLegalRepository(self.settings.postgres)
        return self._postgres_repository

    def register_tools(self) -> None:
        """Không đăng ký MCP tools trong giai đoạn hiện tại."""

        return None
