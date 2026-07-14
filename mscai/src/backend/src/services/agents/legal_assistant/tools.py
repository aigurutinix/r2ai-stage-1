"""Backend tools dùng bởi legal assistant.

File này chứa tool retrieval chạy trực tiếp trong backend. Tool sử dụng registry
Chroma/BM25 đã được nạp từ PostgreSQL khi service khởi động.
"""
from __future__ import annotations

from src.schemas.legal import RetrievalQuery, RetrievedCandidate
from src.services.vector_store import VectorStoreFactory, VectorStoreRegistry


def search_legal_articles(
    query: RetrievalQuery,
    registry: VectorStoreRegistry,
    store_factory: VectorStoreFactory,
) -> list[RetrievedCandidate]:
    """Search điều luật trong các retrieval store local.

    Hàm này là tool search chính của backend agent:
    - agent luôn truyền toàn bộ search spaces đang có, không phân loại query;
    - nếu search space chưa được mở trong process thì tạo store lazy;
    - registry merge toàn bộ candidate và trả global top_k tốt nhất.
    """

    for name in query.search_spaces:
        if not registry.has(name):
            registry.register(name, store_factory.create(name))
    return registry.search(query)
