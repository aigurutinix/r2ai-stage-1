"""State chuyên biệt của legal assistant."""
from __future__ import annotations

from src.services.agents.base.state import AgentState


class LegalAssistantState(AgentState, total=False):
    """Điểm mở rộng cho state riêng của legal agent.

    Hiện tại state dùng lại toàn bộ field từ ``AgentState``. Giữ class riêng giúp
    sau này thêm field pháp lý đặc thù mà không ảnh hưởng base agent khác.
    """
