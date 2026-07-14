"""Cấu hình MCP server theo kiểu plugin giống project mẫu, nhưng tối giản."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MCP_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = MCP_ROOT / "config.yaml"


class ServerConfig(BaseModel):
    """Cấu hình cho một MCP server plugin."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8765
    log_level: str = "INFO"
    path: str = "/mcp"
    settings: dict[str, Any] = Field(default_factory=dict)


class PostgreSQLSettings(BaseModel):
    """Cấu hình PostgreSQL nơi data system lưu legal records."""

    database_url: str = "postgresql://user:password@localhost:25432/legal_assistant"
    table_name: str = "legal_knowledge_records"
    database_column: str | None = "database"

    @field_validator("table_name", "database_column")
    @classmethod
    def validate_identifier(cls, value: str | None) -> str | None:
        """Chỉ cho phép identifier an toàn để tránh SQL injection qua config."""

        if value is None:
            return value
        parts = value.split(".")
        if not all(part.replace("_", "").isalnum() and not part[0].isdigit() for part in parts if part):
            raise ValueError(f"Identifier không hợp lệ: {value}")
        return value


class EmbeddingsSettings(BaseModel):
    """Embedding endpoint dùng cho vector search trong MCP."""

    api_key: str = "sk-1234"
    base_url: str = "http://localhost:8002/v1"
    model: str = "bge-m3"


class VectorStoreSettings(BaseModel):
    """Cấu hình vector store legal MCP đang đọc."""

    provider: Literal["chroma"] = "chroma"
    persist_directory: Path = Path("../backend/chroma_db")
    collection_prefix: str = "legal_articles"
    top_k: int = 8

    @model_validator(mode="after")
    def resolve_persist_directory(self):
        """Resolve path tương đối dựa trên thư mục mcp."""

        if not self.persist_directory.is_absolute():
            self.persist_directory = (MCP_ROOT / self.persist_directory).resolve()
        return self


class LegalServerSettings(BaseModel):
    """Settings riêng cho legal retrieval server."""

    postgres: PostgreSQLSettings = Field(default_factory=PostgreSQLSettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)


class Settings(BaseSettings):
    """Root settings cho MCP process."""

    model_config = SettingsConfigDict(
        env_prefix="LEGAL_MCP_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    debug: bool = False
    environment: str = "development"
    log_level: str = "INFO"
    servers: dict[str, ServerConfig] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_FILE) -> "Settings":
        """Load YAML config, fallback default nếu file chưa tồn tại."""

        data: dict[str, Any] = {}
        if path.exists():
            raw = yaml.safe_load(path.read_text())
            if raw:
                data = raw
        return cls(**data)


@lru_cache(maxsize=1)
def get_settings(config_path: Path = CONFIG_FILE) -> Settings:
    """Đọc và cache settings MCP."""

    return Settings.from_yaml(config_path)
