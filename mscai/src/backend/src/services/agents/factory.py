"""Factory tạo và đăng ký agent cho ứng dụng.

Giữ factory riêng giúp phần FastAPI dependency không phải biết chi tiết cách
khởi tạo từng agent. Khi thêm agent mới, chỉ cần thêm hàm create và register ở
đây, router/dependency có thể lấy qua AgentRegistry.
"""
from __future__ import annotations

from src.services.agents.legal_assistant import LegalAssistantAgent
from src.services.agents.registry import AgentRegistry
from src.services.vector_store import VectorStoreRegistry, vector_store_registry

LEGAL_ASSISTANT_AGENT = "legal-assistant"


def create_legal_assistant_agent(
    registry: VectorStoreRegistry = vector_store_registry,
    llm=None,
) -> LegalAssistantAgent:
    """Tạo legal assistant, có thể inject registry/llm khi test hoặc benchmark."""

    return LegalAssistantAgent(registry=registry, llm=llm)


def create_agent_registry(llm=None) -> AgentRegistry:
    """Tạo registry mặc định của backend và đăng ký các agent đang hỗ trợ."""

    registry = AgentRegistry()
    registry.register(LEGAL_ASSISTANT_AGENT, create_legal_assistant_agent(llm=llm))
    return registry
