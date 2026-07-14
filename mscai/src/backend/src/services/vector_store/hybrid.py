"""Hybrid retrieval bằng cách fusion lexical rank và vector rank."""
from __future__ import annotations

from src.schemas.legal import LegalArticle, RetrievalQuery, RetrievedCandidate
from src.services.vector_store.base import LegalVectorStore


class HybridLegalStore:
    """Kết hợp BM25-like search và vector search bằng Reciprocal Rank Fusion."""

    def __init__(
        self,
        lexical_store: LegalVectorStore,
        vector_store: LegalVectorStore,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self.lexical_store = lexical_store
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def add_articles(self, articles: list[LegalArticle]) -> None:
        """Index cùng record vào cả hai backend retrieval."""

        self.lexical_store.add_articles(articles)
        self.vector_store.add_articles(articles)

    def search(self, query: RetrievalQuery) -> list[RetrievedCandidate]:
        """Merge ranking lexical/vector thành một danh sách ổn định."""

        merged: dict[str, RetrievedCandidate] = {}
        scores: dict[str, float] = {}
        weighted_results = [
            (self.lexical_store.search(query), self.bm25_weight),
            (self.vector_store.search(query), self.dense_weight),
        ]
        for source_results, weight in weighted_results:
            for rank, candidate in enumerate(source_results, start=1):
                article_id = candidate.article.article_id
                scores[article_id] = scores.get(article_id, 0.0) + weight / (self.rrf_k + rank)
                if article_id not in merged:
                    merged[article_id] = candidate
        # RRF dùng rank nội bộ của hai nhánh sparse/dense. Khi nhiều candidate
        # có cùng RRF score, raw score được dùng làm tie-breaker để ưu tiên
        # candidate có điểm BM25 hoặc Chroma tốt hơn.
        ranked = sorted(
            merged.values(),
            key=lambda item: (scores[item.article.article_id], item.score),
            reverse=True,
        )
        for rank, candidate in enumerate(ranked[: query.top_k], start=1):
            score = scores[candidate.article.article_id]
            candidate.rank = rank
            candidate.score = score
            candidate.source = "hybrid"
            candidate.article.score = score
        return ranked[: query.top_k]
