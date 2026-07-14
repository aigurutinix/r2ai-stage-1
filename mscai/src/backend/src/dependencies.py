"""Các dependency provider dùng bởi FastAPI.

FastAPI gọi các hàm trong file này thông qua ``Depends``. Việc cache bằng
``lru_cache`` giúp mỗi process chỉ tạo một LLM client và một agent, tránh khởi
tạo connection/model wrapper lặp lại cho từng request.
"""
from __future__ import annotations

from functools import lru_cache

from src.config import get_settings
from src.services.agents.factory import LEGAL_ASSISTANT_AGENT, create_agent_registry
from src.services.agents.legal_assistant import LegalAssistantAgent
from src.services.agents.registry import AgentRegistry
from src.services.llm.client import LLMClient


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Tạo LLM client dùng chung cho toàn bộ FastAPI process."""

    return LLMClient()


@lru_cache(maxsize=1)
def get_agent_registry() -> AgentRegistry:
    """Tạo registry agent một lần cho toàn bộ FastAPI process."""

    return create_agent_registry(llm=get_llm_client())


@lru_cache(maxsize=1)
def get_legal_assistant_agent() -> LegalAssistantAgent:
    """Lấy legal assistant từ registry agent mặc định."""

    return get_agent_registry().get(LEGAL_ASSISTANT_AGENT)
