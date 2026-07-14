"""Interface và registry chung cho các vector store."""
from __future__ import annotations

from typing import Protocol

from src.schemas.legal import LegalArticle, RetrievalQuery, RetrievedCandidate


class LegalVectorStore(Protocol):
    """Contract tối thiểu mà mọi retrieval backend phải implement.

    Dùng ``Protocol`` giúp code type-check được mà không ép các store phải kế
    thừa class cụ thể. Chroma store, in-memory BM25 và hybrid store chỉ cần có
    đúng hai method này.
    """

    def add_articles(self, articles: list[LegalArticle]) -> None:
        """Thêm hoặc cập nhật các record pháp luật vào store."""
        ...

    def search(self, query: RetrievalQuery) -> list[RetrievedCandidate]:
        """Trả danh sách candidate đã rank cho một retrieval query."""
        ...


class VectorStoreRegistry:
    """Registry in-process map tên search space sang store cụ thể."""

    def __init__(self) -> None:
        self._stores: dict[str, LegalVectorStore] = {}

    def register(self, name: str, store: LegalVectorStore) -> None:
        """Đăng ký store cho một search space nội bộ."""

        self._stores[name] = store

    def get(self, name: str) -> LegalVectorStore:
        """Lấy store theo tên, báo lỗi nếu chưa đăng ký."""

        if name not in self._stores:
            raise KeyError(f"Search space '{name}' is not registered")
        return self._stores[name]

    def has(self, name: str) -> bool:
        """Kiểm tra search space đã có store trong process chưa."""

        return name in self._stores

    def list_databases(self) -> list[str]:
        """Liệt kê các search space đang được mở trong process hiện tại."""

        return sorted(self._stores)

    def search(self, query: RetrievalQuery) -> list[RetrievedCandidate]:
        """Search global trên mọi search space và trả đúng top_k tốt nhất."""

        merged: list[RetrievedCandidate] = []
        for name in query.search_spaces:
            if name not in self._stores:
                continue
            merged.extend(self._stores[name].search(query))
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[: query.top_k]


# Registry dùng chung trong FastAPI process. Dữ liệu bền vững nằm ở PostgreSQL
# và vector DB; registry chỉ giữ các object store đang mở trong runtime.
vector_store_registry = VectorStoreRegistry()
