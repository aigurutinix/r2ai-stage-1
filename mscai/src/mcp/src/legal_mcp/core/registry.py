"""Registry cho các MCP server plugin."""
from __future__ import annotations

from typing import ClassVar


class ServerRegistry:
    """Map tên server trong config sang class server."""

    _servers: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, name: str, server_class: type) -> None:
        """Đăng ký server class theo tên."""

        if name in cls._servers:
            raise ValueError(f"Server '{name}' đã được đăng ký")
        cls._servers[name] = server_class

    @classmethod
    def get(cls, name: str) -> type:
        """Lấy server class theo tên."""

        if name not in cls._servers:
            raise ValueError(f"Server '{name}' chưa đăng ký. Available: {list(cls._servers)}")
        return cls._servers[name]

    @classmethod
    def all(cls) -> dict[str, type]:
        """Trả toàn bộ registry, chủ yếu để debug/test."""

        return dict(cls._servers)
