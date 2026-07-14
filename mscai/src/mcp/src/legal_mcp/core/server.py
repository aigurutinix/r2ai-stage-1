"""Base class cho MCP server plugin."""
from __future__ import annotations

from abc import ABC, abstractmethod

from fastmcp import FastMCP

from legal_mcp.config import ServerConfig


class BaseMCPServer(ABC):
    """Khung tối thiểu cho một MCP server."""

    def __init__(self, name: str, config: ServerConfig) -> None:
        self.name = name
        self.config = config
        self._mcp = FastMCP(name)
        self.register_tools()

    @abstractmethod
    def register_tools(self) -> None:
        """Đăng ký các MCP tools của server."""

    def get_app(self):
        """Trả ASGI app để uvicorn chạy."""

        return self._mcp.http_app(path=self.config.path)
