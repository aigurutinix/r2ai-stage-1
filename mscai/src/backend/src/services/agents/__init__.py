"""Các thành phần agent của backend."""

from src.services.agents.factory import (
    LEGAL_ASSISTANT_AGENT,
    create_agent_registry,
    create_legal_assistant_agent,
)
from src.services.agents.registry import AgentRegistry

__all__ = [
    "AgentRegistry",
    "LEGAL_ASSISTANT_AGENT",
    "create_agent_registry",
    "create_legal_assistant_agent",
]
