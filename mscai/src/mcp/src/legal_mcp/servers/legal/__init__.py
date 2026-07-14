"""Đăng ký legal retrieval MCP server."""
from legal_mcp.core.registry import ServerRegistry
from legal_mcp.servers.legal.server import LegalRetrievalServer

ServerRegistry.register("legal_retrieval", LegalRetrievalServer)
