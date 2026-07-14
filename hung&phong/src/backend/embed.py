"""Embedding client — 2 backend: API (OpenAI-compat/Ollama) hoặc sentence-transformers local."""
from __future__ import annotations

import logging
import os
from typing import Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(self) -> None:
        s = get_settings()
        self.backend = (s.embed_backend or "api").lower()
        self.dim = s.embed_dim
        self.batch_size = s.embed_batch_size

        if self.backend == "st":
            # Model HF chuyên tiếng Việt qua sentence-transformers (GPU local).
            os.environ.setdefault("USE_TF", "0")
            from sentence_transformers import SentenceTransformer

            self.model = s.embed_st_model
            if not self.model:
                raise RuntimeError("EMBED_ST_MODEL chưa set khi EMBED_BACKEND=st")
            logger.info("Embedding backend=st · model=%s", self.model)
            self._st = SentenceTransformer(self.model, device="cuda")
            self._st = self._st.half()   # fp16: tiết kiệm VRAM (chạy cùng reranker + LLM)
            self.dim = self._st.get_sentence_embedding_dimension()
        else:
            from openai import OpenAI

            api_key = s.resolved_embed_api_key
            if not api_key:
                raise RuntimeError(
                    "Embedding API key chưa được set: cần EMBED_API_KEY hoặc LLM_API_KEY trong .env"
                )
            self.client = OpenAI(api_key=api_key, base_url=s.resolved_embed_base_url or None, timeout=60.0)
            self.model = s.embed_model

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [t.replace("\x00", " ").strip() or " " for t in texts]
        if self.backend == "st":
            vecs = self._st.encode(
                cleaned, batch_size=self.batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
            return vecs.tolist()
        return self._embed_api(cleaned)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
    def _embed_api(self, cleaned: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=cleaned)
        return [d.embedding for d in resp.data]

    def embed_one(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]
