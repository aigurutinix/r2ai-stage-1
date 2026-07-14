"""Runner chạy một hoặc nhiều MCP servers bằng uvicorn."""
from __future__ import annotations

import asyncio

import uvicorn


class MultiServerRunner:
    """Chạy nhiều MCP server song song, tối giản từ project mẫu."""

    def __init__(self, servers: list) -> None:
        self.servers = servers

    async def _run_server(self, server) -> None:
        """Chạy một server bằng uvicorn."""

        config = uvicorn.Config(
            app=server.get_app(),
            host=server.config.host,
            port=server.config.port,
            log_level=server.config.log_level.lower(),
        )
        await uvicorn.Server(config).serve()

    async def run(self) -> None:
        """Chạy tất cả server enabled."""

        await asyncio.gather(*(self._run_server(server) for server in self.servers))

    def start(self) -> None:
        """Entry point đồng bộ."""

        asyncio.run(self.run())
