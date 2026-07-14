"""Cấu hình ứng dụng chỉ đọc từ ``backend/config.yaml``.

File này là nơi gom toàn bộ cấu hình runtime: FastAPI, LLM, embedding,
short-memory và fallback vector store local. Các class cấu hình dùng Pydantic để:

- validate kiểu dữ liệu ngay khi app khởi động;
- có default rõ ràng khi thiếu config;
- tránh truyền dict thô rời rạc khắp codebase.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Tuple, Type

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# PRJ_ROOT trỏ tới thư mục backend. Các path tương đối trong config.yaml sẽ được
# resolve dựa trên thư mục này để chạy từ đâu cũng ổn định.
PRJ_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PRJ_ROOT / "config.yaml"


class ConfigModel(BaseModel):
    """BaseModel chung cho các block config con.

    ``extra='ignore'`` giúp config.yaml có thể chứa key mới trong tương lai mà
    code cũ không bị crash. ``populate_by_name=True`` cho phép dùng cả alias
    và tên field Python khi validate.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class AppSettings(ConfigModel):
    """Cấu hình host/port mặc định cho FastAPI agent service."""

    host: str = "0.0.0.0"
    port: int = 8000


class UISettings(ConfigModel):
    """Cấu hình Next.js UI khi chạy qua compose wrapper."""

    host: str = "0.0.0.0"
    port: int = 5173


class LLMSettings(ConfigModel):
    """Cấu hình chat model OpenAI-compatible, thường trỏ tới vLLM local."""

    api_key: str = "sk-1234"
    base_url: str = "http://localhost:8001/v1"
    # AliasChoices cho phép config.yaml dùng ``default_model`` hoặc
    # ``model_name`` mà code vẫn đọc về cùng một field.
    model_name: str = Field(
        "qwen3-8b-fp8",
        validation_alias=AliasChoices("model_name", "default_model"),
    )
    temperature: float = 0.0
    max_tokens: int | None = None
    enable_thinking: bool = False


class EmbeddingsSettings(ConfigModel):
    """Cấu hình embedding endpoint OpenAI-compatible."""

    api_key: str = "sk-1234"
    base_url: str = "http://localhost:8002/v1"
    model: str = "bge-m3"
    max_input_tokens: int = Field(default=8190, ge=128)
    tokenizer_encoding: str = "cl100k_base"


class ShortMemorySettings(ConfigModel):
    """Bật/tắt LangGraph InMemorySaver cho chat theo session_id.

    Memory chỉ nằm trong RAM của backend process và mất khi service restart.
    """

    enabled: bool = True


class ChatSettings(ConfigModel):
    """Cấu hình cách endpoint chat trả kết quả cho UI/client."""

    streaming: bool = True
    token_streaming: bool = True


class CompetitionSettings(ConfigModel):
    """Bật mode chạy tập test: luôn coi query là câu hỏi luật."""

    enabled: bool = False
    max_concurrency: int = Field(default=4, ge=1)
    save_outputs: bool = True
    output_dir: Path = Path("./outputs")

    @model_validator(mode="after")
    def resolve_output_dir(self):
        """Chuẩn hóa thư mục lưu kết quả competition dưới backend."""

        if not self.output_dir.is_absolute():
            self.output_dir = PRJ_ROOT / self.output_dir
        return self


class RewriteSettings(ConfigModel):
    """Bật/tắt rewrite query và giới hạn số biến thể query retrieval."""

    enabled: bool = False
    max_variants: int = 3


class HyDESettings(ConfigModel):
    """Bật/tắt HyDE: sinh hypothetical answer để làm query dense retrieval."""

    enabled: bool = False


class RerankerSettings(ConfigModel):
    """Cấu hình reranker cross-encoder chạy sau retrieval."""

    enabled: bool = False
    api_key: str = "sk-1234"
    base_url: str = "http://localhost:8025/v1"
    model: str = "qwen3-reranker-06b"
    filter_mode: Literal["fixed", "largest_gap"] = "fixed"
    threshold: float = 0.0
    min_gap: float = Field(default=0.0, ge=0)
    min_keep: int = Field(default=1, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    endpoint: str = "/v1/rerank"


class LLMFilterSettings(ConfigModel):
    """Bật/tắt bước LLM đánh giá từng điều luật sau rerank."""

    enabled: bool = False
    max_concurrency: int = Field(default=4, ge=1)
    min_keep: int = Field(default=1, ge=0)


class PostgreSQLSettings(ConfigModel):
    """Nguồn dữ liệu luật dùng để dựng Chroma và nạp BM25 khi startup."""

    enabled: bool = True
    database_url: str = "postgresql://postgres:postgres@localhost:23432/legal_assistant"
    table_name: str = "legal_knowledge_records"
    batch_size: int = Field(default=128, ge=1)

    @field_validator("table_name")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        """Chỉ chấp nhận identifier SQL đơn giản từ file cấu hình."""

        parts = value.split(".")
        if not all(part and part.replace("_", "").isalnum() and not part[0].isdigit() for part in parts):
            raise ValueError(f"SQL identifier không hợp lệ: {value}")
        return value


class VectorStoreSettings(ConfigModel):
    """Cấu hình backend retrieval: lexical, Chroma vector hoặc hybrid."""

    mode: Literal["bm25", "chroma", "hybrid"] = "hybrid"
    persist_directory: Path = Path("./chroma_db")
    default_collection: str = "legal_articles"
    rrf_k: int = 60
    dense_weight: float = Field(default=2.0, gt=0)
    bm25_weight: float = Field(default=1.0, gt=0)
    top_k: int = 8
    bm25_tokenizer: Literal["auto", "underthesea", "regex"] = "auto"
    bm25_k1: float = 2.0
    bm25_b: float = 1.0
    bm25_epsilon: float = 0.5

    @model_validator(mode="after")
    def resolve_persist_directory(self):
        """Chuẩn hóa path Chroma sau khi Pydantic parse xong model.

        Người dùng có thể viết ``./chroma_db`` trong YAML. Validator này đổi nó
        thành absolute path dưới thư mục backend để fallback local luôn nhìn đúng
        vị trí vector index.
        """

        if not self.persist_directory.is_absolute():
            self.persist_directory = PRJ_ROOT / self.persist_directory
        return self


class LegalAssistantSettings(ConfigModel):
    """Nhóm cấu hình riêng cho agent pháp lý."""

    chat: ChatSettings = Field(default_factory=ChatSettings)
    competition: CompetitionSettings = Field(default_factory=CompetitionSettings)
    rewrite: RewriteSettings = Field(default_factory=RewriteSettings)
    hyde: HyDESettings = Field(default_factory=HyDESettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    llm_filter: LLMFilterSettings = Field(default_factory=LLMFilterSettings)
    postgres: PostgreSQLSettings = Field(default_factory=PostgreSQLSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)


class Settings(BaseSettings):
    """Root settings object được inject vào toàn bộ ứng dụng."""

    app: AppSettings = Field(default_factory=AppSettings)
    ui: UISettings = Field(default_factory=UISettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    short_memory: ShortMemorySettings = Field(default_factory=ShortMemorySettings)
    legal_assistant: LegalAssistantSettings = Field(
        default_factory=LegalAssistantSettings,
        validation_alias=AliasChoices("legal_assistant", "legal-assistant"),
    )

    model_config = SettingsConfigDict(
        yaml_file=CONFIG_FILE,
        extra="ignore",
        populate_by_name=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Quy định thứ tự đọc config.

        Không đưa env/dotenv vào danh sách nguồn để ``config.yaml`` luôn là
        nguồn cấu hình runtime duy nhất của backend.
        """

        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Trả về settings đã cache để không parse YAML nhiều lần."""

    return Settings()


# Biến tiện ích cho các module đơn giản cần đọc config trực tiếp.
settings = get_settings()
