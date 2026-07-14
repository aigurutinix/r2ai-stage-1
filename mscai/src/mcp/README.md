# Legal Data Logic

Tạm thời package `mcp/` chỉ giữ logic data/database cho legal assistant. Backend agent không gọi MCP tools trong giai đoạn này.

Các phần còn giữ:

```text
mcp/src/legal_mcp/postgres.py  # logic đọc PostgreSQL/search related bằng exact match
mcp/src/legal_mcp/vector.py    # logic đọc Chroma/vector index nếu cần dùng lại
mcp/src/legal_mcp/schemas.py   # schema dữ liệu nội bộ phía data layer
```

Luồng chat hiện tại chạy retrieval trực tiếp trong backend/local vector store theo `category`, không qua MCP tool.
