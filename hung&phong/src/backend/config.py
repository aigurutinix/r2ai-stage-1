"""Centralized config loaded from .env via pydantic-settings."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider (OpenAI-compatible API) — dùng cho phần sinh câu trả lời (chat).
    # Trỏ sang Ollama: LLM_BASE_URL=http://localhost:11434/v1, LLM_API_KEY=ollama
    llm_api_key: str = ""
    llm_base_url: str = ""

    # LLM
    llm_model: str = "deepseek-v3.2"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.2
    # Reasoning effort cho model "thinking" (vd Qwen3.5 qua Ollama).
    # "none" = tắt thinking để token dồn vào câu trả lời (tránh content rỗng).
    # Để trống = không gửi tham số (cho provider không hỗ trợ reasoning_effort).
    llm_reasoning_effort: str = ""

    # Embedding provider — TÁCH RIÊNG khỏi LLM để có thể chat bằng model local
    # (Ollama) trong khi embedding vẫn dùng provider mạnh (text-embedding-3-large).
    # Để trống embed_api_key/embed_base_url → tự fallback về llm_* (tương thích ngược).
    embed_api_key: str = ""
    embed_base_url: str = ""
    embed_model: str = "text-embedding-3-large"
    embed_dim: int = 3072
    # Backend embedding: "api" (OpenAI-compat / Ollama) | "st" (sentence-transformers local GPU).
    # "st" để dùng model HF chuyên tiếng Việt (vd AITeamVN/Vietnamese_Embedding_v2).
    embed_backend: str = "api"
    embed_st_model: str = ""

    @property
    def resolved_embed_api_key(self) -> str:
        return self.embed_api_key or self.llm_api_key

    @property
    def resolved_embed_base_url(self) -> str:
        return self.embed_base_url or self.llm_base_url

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "vbpl_v1"

    # Retrieval
    top_k: int = 10
    rerank_top_k: int = 5
    # Hybrid + reranker
    hybrid_search: bool = False        # bật BM25 (lexical) song song dense
    use_reranker: bool = False         # bật cross-encoder bge-reranker-v2-m3
    hybrid_fetch: int = 40             # số ứng viên mỗi retriever trước khi fuse/rerank
    bm25_index_path: str = ""          # để trống = data/bm25_<collection>.pkl
    use_hnsw: bool = False             # dense qua HNSW (nhanh) thay Qdrant brute-force
    hnsw_index_path: str = ""          # để trống = data/hnsw_<collection>.bin

    # Ingestion — nguồn DUY NHẤT: tmquan/vbpl-vn (scrape vbpl.vn). Đã bỏ th1nhng0
    # (field tinh_trang_hieu_luc của nó sai, vd Luật Đầu tư 2020 bị gán "hết hiệu lực").
    hf_dataset: str = "tmquan/vbpl-vn"
    hf_cache_dir: str = "./data/hf_cache"
    embed_batch_size: int = 64
    ingest_limit: int = 0  # 0 = all
    # Lọc corpus về đúng phạm vi cuộc thi (Luật DN & SME). Xem ingest/scope.py.
    # True → chỉ ingest văn bản lõi khớp từ khoá DN/SME (~20k docs / ~305k chunks).
    ingest_scope_filter: bool = False

    # Backend
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    cors_origins: str = "http://localhost:3000"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
