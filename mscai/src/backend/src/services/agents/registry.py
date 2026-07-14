"""Registry in-process cho các agent theo tên.

File này giữ vai trò điểm mở rộng: hiện backend chỉ chạy một legal agent,
nhưng sau này có thể đăng ký thêm agent mới mà không phải đổi toàn bộ router.
"""
from __future__ import annotations

from typing import Any


class AgentRegistry:
    """Map tên agent sang object agent đã khởi tạo trong cùng một process."""

    def __init__(self) -> None:
        self._agents: dict[str, Any] = {}

    def register(self, name: str, agent: Any) -> None:
        """Đăng ký hoặc thay thế agent theo tên."""

        self._agents[name] = agent

    def get(self, name: str) -> Any:
        """Lấy agent theo tên, báo lỗi rõ nếu agent chưa được đăng ký."""

        if name not in self._agents:
            raise KeyError(f"Agent '{name}' is not registered")
        return self._agents[name]

    def list_agents(self) -> list[str]:
        """Liệt kê tên agent đang có trong process hiện tại."""

        return sorted(self._agents)
