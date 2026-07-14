"""Context runtime truyền qua các node của agent."""
from __future__ import annotations

from pydantic import BaseModel


class AgentContext(BaseModel):
    """Các tùy chọn runtime không trộn trực tiếp vào câu hỏi người dùng."""

    session_id: str | None = None
    top_k: int = 8
