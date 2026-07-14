"""Factory tạo MCP servers từ config."""
from __future__ import annotations

from legal_mcp.config import Settings
from legal_mcp.core.registry import ServerRegistry


class ServerFactory:
    """Tạo các server enabled, giống pattern trong project mẫu."""

    @staticmethod
    def create_enabled_servers(settings: Settings, filter_names: list[str] | None = None) -> list:
        """Tạo server instances theo config."""

        servers = []
        for name, config in settings.servers.items():
            if filter_names and name not in filter_names:
                continue
            if not config.enabled:
                continue
            server_class = ServerRegistry.get(name)
            servers.append(server_class(name=name, config=config))
        return servers
