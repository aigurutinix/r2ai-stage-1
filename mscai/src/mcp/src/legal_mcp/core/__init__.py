"""Core hạ tầng tối giản cho legal MCP server."""
from legal_mcp.core.factory import ServerFactory
from legal_mcp.core.runner import MultiServerRunner
from legal_mcp.core.server import BaseMCPServer
from legal_mcp.core.registry import ServerRegistry

__all__ = ["BaseMCPServer", "MultiServerRunner", "ServerFactory", "ServerRegistry"]
