"""Wrapper cho embedding endpoint OpenAI-compatible."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.config import settings


class EmbeddingsClient:
    """Client embedding dùng LangChain ``OpenAIEmbeddings``.

    Endpoint có thể là server local bạn đã host, miễn là API tương thích OpenAI.
    Với base_url=http://localhost:8026/v1, LangChain sẽ gọi đúng endpoint
    POST http://localhost:8026/v1/embeddings.
    Trước khi gọi endpoint, client cắt text theo ``embeddings.max_input_tokens``
    để tránh lỗi context length khi một điều luật có ``content`` quá dài.
    """

    def __init__(self):
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover - guard khi thiếu dependency
            raise RuntimeError("Install langchain-openai to use the embeddings endpoint") from exc

        self.max_input_tokens = settings.embeddings.max_input_tokens
        self._encoding = self._load_encoding(settings.embeddings.tokenizer_encoding)
        self.embeddings = OpenAIEmbeddings(
            model=settings.embeddings.model,
            base_url=settings.embeddings.base_url,
            api_key=settings.embeddings.api_key,
            tiktoken_enabled=False,  # vLLM expects raw text, not pre-tokenized OpenAI token IDs
            check_embedding_ctx_length=False,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed nhiều record pháp luật khi build vector database."""

        return self.embeddings.embed_documents([self._truncate_text(text) for text in texts])

    def embed_query(self, text: str) -> list[float]:
        """Embed một query đã rewrite/HyDE để search vector store."""

        return self.embeddings.embed_query(self._truncate_text(text))

    def _load_encoding(self, encoding_name: str) -> Any | None:
        """Nạp tokenizer cục bộ để cắt token trước khi gửi embedding endpoint."""

        try:
            import tiktoken

            return tiktoken.get_encoding(encoding_name)
        except Exception:  # pragma: no cover - fallback khi thiếu tiktoken/encoding
            return None

    def _truncate_text(self, text: str) -> str:
        """Cắt phần cuối text nếu vượt giới hạn token embedding."""

        if not text or self.max_input_tokens <= 0:
            return text
        if self._encoding is None:
            # Fallback bảo thủ: với tiếng Việt, 1 token thường không vượt quá vài ký tự.
            # Cắt theo ký tự để tránh gửi văn bản cực dài khi tokenizer local không sẵn sàng.
            return text[: self.max_input_tokens * 2]

        tokens = self._encoding.encode(text)
        if len(tokens) <= self.max_input_tokens:
            return text
        return self._encoding.decode(tokens[: self.max_input_tokens]).rstrip()


@lru_cache(maxsize=1)
def get_embeddings_client() -> EmbeddingsClient:
    """Dùng chung một embedding client cho build index và query retrieval."""

    return EmbeddingsClient()
