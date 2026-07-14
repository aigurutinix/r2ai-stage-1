"""Entry point chạy legal MCP servers theo config."""
from __future__ import annotations

import sys

from legal_mcp.config import get_settings
from legal_mcp.core import MultiServerRunner, ServerFactory


def main() -> None:
    """Load plugins, tạo servers enabled và chạy runner."""

    import legal_mcp.servers  # noqa: F401 - trigger ServerRegistry.register

    settings = get_settings()
    filter_names = sys.argv[1:] if len(sys.argv) > 1 else None
    servers = ServerFactory.create_enabled_servers(settings, filter_names=filter_names)
    if not servers:
        raise SystemExit("Không có MCP server nào được bật trong mcp/config.yaml")
    MultiServerRunner(servers).start()


if __name__ == "__main__":
    main()
