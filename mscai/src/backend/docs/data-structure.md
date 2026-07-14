# Data Structure

PostgreSQL là nguồn dữ liệu chuẩn để backend tự build Chroma khi khởi động.

## Record chuẩn

```json
{
  "id": 1,
  "law_id": "41/2024/QH15",
  "law_name": "Bộ luật Luật Bảo hiểm xã hội 2024 số áp dụng năm 2025",
  "doc_type": "Bộ luật",
  "article": "Điều 1",
  "article_title": "Phạm vi điều chỉnh",
  "content": "Luật này quy định về quyền, trách nhiệm...",
  "author": "Quốc hội"
}
```

Các cột bắt buộc:

```text
id, law_id, law_name, doc_type, article, article_title,
content, author
```

`chapter` và `extra` là tùy chọn. `extra` có thể là JSON/JSONB/array chứa các
reference dạng:

```text
doc_type|law_id|law_name|article
```

## Text được embedding

Mỗi record dùng đúng một format:

```text
{article_title}:{content}
```

Không đưa `id`, `law_id`, `doc_type`, `article`, `author` hoặc
`extra` vào vector text. Các field này vẫn được lưu trong Chroma metadata để
trích dẫn và dựng lại `LegalArticle`.

## Retrieval metadata

Dataset không còn trường `category`. Agent không gọi LLM để phân loại category, không giới hạn search theo category và không chia Chroma/BM25 theo category. Mỗi request search global `top_k` trên retrieval store đã nạp.

Build document và search query luôn sử dụng cùng `EmbeddingsClient`, cùng endpoint, model và tokenizer phía embedding server.
