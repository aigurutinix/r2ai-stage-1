"""BM25 lexical retrieval store dùng làm nhánh sparse của hybrid search."""
from __future__ import annotations

import math
import re

from src.schemas.legal import LegalArticle, RetrievalQuery, RetrievedCandidate

try:  # underthesea giúp tách từ tiếng Việt tốt hơn regex thường.
    from underthesea import text_normalize, word_tokenize
except ImportError:  # pragma: no cover - fallback khi môi trường chưa cài dependency
    text_normalize = None
    word_tokenize = None

try:  # rank_bm25 là implementation BM25 chuẩn, giống hướng project mẫu.
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - fallback thủ công bên dưới
    BM25Okapi = None

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str, mode: str = "auto") -> list[str]:
    """Tách token cho search keyword tiếng Việt.

    ``auto`` ưu tiên underthesea nếu có, sau đó fallback regex. Regex vẫn hữu ích
    cho số hiệu văn bản như ``41/2024/QH15`` vì giữ được các mảnh số/chữ.
    """

    normalized = text.lower()
    if mode in {"auto", "underthesea"} and text_normalize and word_tokenize:
        normalized = text_normalize(normalized)
        tokens = word_tokenize(normalized)
        return [token.strip().lower() for token in tokens if token.strip()]
    return [token.lower() for token in _TOKEN_RE.findall(normalized)]


def tokenizer_signature(mode: str = "auto") -> str:
    """Tên tokenizer thực tế để đưa vào manifest cache BM25."""

    if mode in {"auto", "underthesea"} and text_normalize and word_tokenize:
        return "underthesea"
    return "regex"


class InMemoryLegalStore:
    """BM25 store chạy trong RAM cho corpus pháp luật.

    Store này index ``LegalArticle.vector_text`` và dùng BM25Okapi khi package
    ``rank_bm25`` có sẵn. Nếu dependency chưa được cài, nó fallback sang công
    thức BM25 nhỏ gọn để service vẫn chạy được trong môi trường nhẹ.
    """

    def __init__(
        self,
        database: str = "default",
        tokenizer: str = "auto",
        k1: float = 2.0,
        b: float = 1.0,
        epsilon: float = 0.5,
    ) -> None:
        self.database = database
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self._articles: dict[str, LegalArticle] = {}
        self._article_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25 = None

    def add_articles(self, articles: list[LegalArticle]) -> None:
        """Index text chuẩn của từng record vào BM25 corpus."""

        for article in articles:
            self._articles[article.article_id] = article
        self._rebuild_index()

    @classmethod
    def from_indexed_articles(
        cls,
        database: str,
        articles: list[LegalArticle],
        tokenized_corpus: list[list[str]],
        tokenizer: str = "auto",
        k1: float = 2.0,
        b: float = 1.0,
        epsilon: float = 0.5,
    ) -> "InMemoryLegalStore":
        """Tạo BM25 store từ corpus đã tokenize sẵn.

        Cache này tránh phải chạy lại tokenizer tiếng Việt ở mỗi lần startup.
        Model BM25 vẫn được dựng từ token có sẵn để object runtime luôn sạch.
        """

        store = cls(database=database, tokenizer=tokenizer, k1=k1, b=b, epsilon=epsilon)
        store._articles = {article.article_id: article for article in articles}
        store._article_ids = [article.article_id for article in articles]
        store._tokenized_corpus = tokenized_corpus
        store._build_bm25_model()
        return store

    def export_tokenized_corpus(self) -> list[list[str]]:
        """Trả corpus đã tokenize để persist xuống cache BM25."""

        return self._tokenized_corpus

    def search(self, query: RetrievalQuery) -> list[RetrievedCandidate]:
        """Search BM25 theo toàn bộ query variants và trả top_k candidate."""

        if not self._article_ids:
            return []
        query_text = " ".join(query.all_queries)
        query_tokens = tokenize(query_text, self.tokenizer)
        if not query_tokens:
            return []

        scores = self._score(query_tokens)
        query_token_set = set(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (scores[index], self._overlap_count(index, query_token_set)),
            reverse=True,
        )[: query.top_k]
        candidates: list[RetrievedCandidate] = []
        for rank, index in enumerate(ranked_indices, start=1):
            raw_score = float(scores[index])
            # Với corpus nhỏ, BM25Okapi có thể trả 0 cho nhiều token. Vẫn giữ
            # kết quả top-rank để hybrid RRF có tín hiệu sparse như project mẫu.
            score = raw_score if raw_score > 0 else 1.0 / rank
            article = self._articles[self._article_ids[index]].model_copy(update={"score": score})
            candidates.append(RetrievedCandidate(article=article, source="bm25", score=score, rank=rank))
        return candidates

    def _rebuild_index(self) -> None:
        """Tạo lại corpus và BM25 model sau khi add/update record."""

        self._article_ids = list(self._articles)
        self._tokenized_corpus = [tokenize(index_text(self._articles[article_id]), self.tokenizer) for article_id in self._article_ids]
        self._build_bm25_model()

    def _build_bm25_model(self) -> None:
        """Dựng BM25Okapi từ corpus token hiện có."""

        if BM25Okapi is not None:
            self._bm25 = BM25Okapi(
                self._tokenized_corpus if self._tokenized_corpus else [[]],
                k1=self.k1,
                b=self.b,
                epsilon=self.epsilon,
            )
        else:
            self._bm25 = None

    def _score(self, query_tokens: list[str]) -> list[float]:
        """Tính score bằng rank_bm25 hoặc fallback BM25 thủ công."""

        if self._bm25 is not None:
            return [float(score) for score in self._bm25.get_scores(query_tokens)]
        return self._manual_bm25_scores(query_tokens)

    def _overlap_count(self, index: int, query_token_set: set[str]) -> int:
        """Đếm token trùng để phá hòa khi BM25 score bằng nhau."""

        return len(set(self._tokenized_corpus[index]) & query_token_set)

    def _manual_bm25_scores(self, query_tokens: list[str]) -> list[float]:
        """Fallback BM25 nhỏ gọn khi rank_bm25 chưa được cài."""

        total_docs = len(self._tokenized_corpus)
        doc_lengths = [len(doc) for doc in self._tokenized_corpus]
        avgdl = sum(doc_lengths) / max(total_docs, 1)
        doc_freq: dict[str, int] = {}
        for doc in self._tokenized_corpus:
            for token in set(doc):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        scores: list[float] = []
        for doc, doc_len in zip(self._tokenized_corpus, doc_lengths):
            score = 0.0
            for token in query_tokens:
                tf = doc.count(token)
                if not tf:
                    continue
                df = doc_freq.get(token, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(avgdl, 1e-9))
                score += idf * (tf * (self.k1 + 1)) / denominator
            scores.append(score)
        return scores


def index_text(article: LegalArticle) -> str:
    """Trả text chuẩn dùng chung cho lexical và vector retrieval."""

    return article.vector_text
