# Legal Assistant System Report

## 1. Mục tiêu

Hệ thống trả lời câu hỏi pháp luật Việt Nam dựa trên legal records đã chuẩn hóa
trong PostgreSQL. Thiết kế giữ luồng đơn giản:

- PostgreSQL giữ dữ liệu gốc.
- Chroma giữ vector index persistent.
- BM25 giữ lexical index trong RAM.
- LangGraph điều phối intent, retrieval và answer.
- FastAPI cung cấp API; Next.js cung cấp giao diện chat.

Hệ thống không xử lý PDF/OCR và không có API quản trị dataset.

## 2. Sơ đồ tổng thể

```mermaid
flowchart LR
    PG[(PostgreSQL)] --> Builder[Startup Index Builder]
    Embed[Embedding Server] --> Builder
    Builder --> Chroma[(Chroma collections)]
    Builder --> BM25[BM25 in memory]

    User[User / Test set] --> API[FastAPI]
    API --> Graph[LangGraph Legal Agent]
    Graph --> LLM[LLM Server]
    Graph --> Tool[search_legal_articles]
    Tool --> BM25
    Tool --> Chroma
    BM25 --> RRF[Reciprocal Rank Fusion]
    Chroma --> RRF
    RRF --> Graph
    Graph --> API
    API --> User
```

## 3. Dữ liệu PostgreSQL

Record chuẩn:

```json
{
  "id": 1,
  "law_id": "41/2024/QH15",
  "law_name": "Bộ luật Luật Bảo hiểm xã hội 2024 số áp dụng năm 2025",
  "doc_type": "Bộ luật",
  "article": "Điều 1",
  "article_title": "Phạm vi điều chỉnh",
  "content": "...",
  "author": "Quốc hội"
}
```

`chapter` và `extra` là tùy chọn. PostgreSQL chỉ được đọc tại startup; agent
không ghi hoặc sửa dữ liệu.

## 4. Build Chroma tự động

FastAPI gọi `initialize_legal_index()` trong lifespan, trước khi mở API.

```mermaid
sequenceDiagram
    participant B as Backend startup
    participant P as PostgreSQL
    participant M as Manifest
    participant E as Embedding server
    participant C as Chroma

    B->>P: đọc records
    P-->>B: legal records
    B->>M: kiểm tra source/model/count
    alt index hợp lệ
        M-->>B: reuse
    else cần rebuild
        B->>C: xóa legal collections cũ
        loop từng retrieval store và batch
            B->>E: embed vector_text
            E-->>B: vectors
            B->>C: upsert vectors + metadata
        end
        B->>M: lưu manifest mới
    end
    B->>B: nạp BM25 vào RAM
```

Text embedding duy nhất:

```text
{article_title}:{content}
```

Manifest `chroma_db/legal_index_manifest.json` lưu:

- PostgreSQL host, port, database, table và batch size;
- embedding endpoint và model;
- format vector text;
- số record của từng retrieval store.

Vì vậy lần chạy sau không embedding lại nếu index còn hợp lệ. Khi port/source,
số record hoặc embedding model thay đổi, hệ thống rebuild để bảo đảm nhất quán.
File lock ngăn nhiều uvicorn worker cùng build một lúc.

## 5. Cùng model và tokenizer

`get_embeddings_client()` cache đúng một `EmbeddingsClient` trong mỗi process.
Build document và embed query đều gửi raw text tới cùng:

```yaml
embeddings:
  base_url: http://localhost:8013/v1
  model: bge-m3
```

Tokenizer được thực thi bởi cùng embedding server/model. Không có tokenizer
embedding thứ hai trong backend. `bm25_tokenizer` chỉ phục vụ nhánh keyword.

## 6. Global retrieval và hybrid search

Agent không còn dùng LLM để phân loại category. Khi query cần retrieval, backend luôn search global `top_k` trên retrieval store đã được nạp khi startup.

```text
query -> Chroma/BM25 store -> global rank -> top_k candidates -> reranker filter -> optional LLM filter -> final context
```

Khi `mode: hybrid`, startup nạp cùng legal records vào BM25. Search chạy song song hai nhánh và hợp nhất rank bằng RRF:

```text
query -> BM25 rank ----+
                       +-> RRF -> top_k candidates
query -> Chroma rank --+
```

Dataset/PostgreSQL không còn dùng trường `category`; Chroma/BM25 được build thành một retrieval store global.

## 6.1. Reranker

Reranker là một service scoring, không phải agent riêng. Sau retrieval, backend gọi
Qwen3 reranker với `query=retrieval_question`, `documents=[law_name\\narticle_title:content, ...]` và `top_n=len(documents)`. Sau khi có score, backend lọc theo `legal_assistant.reranker.filter_mode`:

- `fixed`: dùng threshold tĩnh, giữ score lớn hơn hoặc bằng `threshold`.
- `largest_gap`: tìm khoảng cách score lớn nhất giữa hai kết quả liền kề trong danh sách đã sort giảm dần, giữ phần phía trên khoảng cách đó. `min_gap` là gap tối thiểu để được phép cắt, `min_keep` là số điều luật tối thiểu luôn giữ.

Score reranker có thể âm nên threshold không bị giới hạn dương.

## 6.2. LLM filter sau rerank

`legal_assistant.llm_filter` là bước tùy chọn sau reranker. Backend gửi từng điều luật đã qua rerank cho LLM và dùng prompt riêng theo nguồn query: rewritten query dùng prompt đánh giá quan hệ trực tiếp, HyDE answer dùng prompt đánh giá tương đồng ngữ nghĩa pháp lý. Kết quả vẫn chỉ là `PASS` hoặc `DROP`. `PASS` được giữ trong context cuối, `DROP` bị loại. Bước này không tạo agent mới và không dùng registry vì nó chỉ là quality gate nội bộ của legal workflow. Nếu endpoint LLM lỗi, item lỗi được giữ lại để tránh mất căn cứ; nếu LLM loại hết, `min_keep` giữ lại top N sau rerank.

## 7. Workflow chat

```mermaid
flowchart TD
    Q[User question] --> Intent[Analyze intent]
    Intent -->|SKIP| General[General LLM answer]
    Intent -->|NEXT| Prepare[none / rewrite / hyde]
    Prepare --> Search[Global top-k retrieval]
    Search --> Context[Legal context + metadata]
    Context --> Answer[Grounded legal answer]
    General --> Output[API response]
    Answer --> Output
```

- `none`: search bằng query gốc.
- `rewrite`: LLM làm rõ query pháp luật trước search.
- `hyde`: LLM sinh câu trả lời giả định ngắn rồi search bằng đoạn đó.
- Retrieval luôn lấy global `vector_store.top_k` candidates trên toàn bộ stores đã nạp.

## 8. Short-term memory

LangGraph `InMemorySaver` giữ messages theo `session_id`. Memory chỉ tồn tại khi
backend process đang chạy; restart backend sẽ giải phóng toàn bộ hội thoại.
PostgreSQL và Chroma không lưu lịch sử chat.

## 9. Cấu hình chính

```yaml
legal_assistant:
  postgres:
    enabled: true
    database_url: postgresql://postgres:postgres@localhost:23432/legal_assistant
    table_name: legal_knowledge_records
    batch_size: 128
  rewrite:
    enabled: true
    max_variants: 3
  hyde:
    enabled: false
  vector_store:
    mode: hybrid
    persist_directory: ./chroma_db
    default_collection: legal_articles
```

## 10. Phân chia trách nhiệm

```text
PostgreSQL: dữ liệu luật chuẩn
Index builder: validate, build/reuse Chroma, preload BM25
Backend tool: global top-k retrieval
Agent: hiểu query và tổng hợp câu trả lời
LLM: intent, rewrite/HyDE, answer
UI: quản lý trải nghiệm từng đoạn chat
```

Cấu trúc này giữ data ingestion tách khỏi request chat, nhưng vẫn tự động chuẩn
bị retrieval index khi service khởi động.
