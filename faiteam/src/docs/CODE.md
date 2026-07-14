# Mô tả mã nguồn — R2AI Stage 1

Tài liệu mô tả kiến trúc mã nguồn, các module, script, dependencies và luồng xử lý của hệ thống R2AI.

---

## 1. Tổng quan kiến trúc

R2AI là hệ thống **RAG (Retrieval-Augmented Generation)** monolith Python, orchestrated bởi một script chính. Kiến trúc theo pipeline tuần tự:

```
R2AIStage1DATA.json
        │
        ▼
┌───────────────────┐
│ Sub-query planning│  query_decompose.py / subquery_loader.py
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Hybrid retrieval  │  bm25_retrieval.py + qdrant_config.py
│ (Dense + BM25)    │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Cross-encoder     │  rerank_retrieval.py
│ rerank            │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Filter valid      │  rag_answer_stage1.py
│ chunks            │
└─────────┬─────────┘
          ▼
    ┌─────┴─────┐
    │           │
 citations   Full LLM
  only       generation
    │           │
    └─────┬─────┘
          ▼
   JSON output
```

**Script orchestrator:** `scripts/rag_answer_stage1.py` (~1.700 dòng)

---

## 2. Cấu trúc thư mục mã nguồn

```
.                              # Thư mục gốc repo
├── README.md
├── docs/                        # Tài liệu (DATASET, MODELS, CODE, PIPELINE_OVERVIEW)
└── R2AI/                        # Mã nguồn pipeline
    ├── .env.example
    ├── .gitignore
    ├── scripts/                 # Pipeline và utilities
    │   ├── rag_answer_stage1.py     # ★ Entry point chính
    │   ├── qdrant_config.py         # Config Qdrant + model resolver
    │   ├── bm25_retrieval.py        # BM25 index + RRF fusion
    │   ├── rerank_retrieval.py      # Cross-encoder rerank
    │   ├── query_decompose.py       # Rule-based sub-query split
    │   ├── subquery_loader.py       # Load precomputed sub-queries
    │   ├── query_qdrant.py          # CLI test single query
    │   ├── rerank_hybrid.py         # Standalone rerank utility
    │   ├── generate_subqueries.py   # Precompute sub-query JSON
    │   ├── ingest_parquet_to_qdrant.py
    │   ├── ingest_docx_to_qdrant.py
    │   ├── text_document_builder.py
    │   ├── document_filters.py
    │   ├── effectiveness_sources.py
    │   ├── vbpl_client.py
    │   ├── build_effectiveness_parquet.py
    │   ├── export_filtered_metadata.py
    │   ├── check_expired_law_codes.py
    │   ├── cleanup_invalid_docx.py
    │   ├── cleanup_non_chl_docx.py
    │   ├── restore_deleted_docx.py
    │   ├── qdrant_ingest.py
    │   └── requirements-rag.txt
    │
    ├── Extractor_chunk/             # Module: text chunking
    ├── Reader/                      # Module: DOCX/PDF reading
    ├── Construct_Tree/              # Module: legal document tree
    │
    ├── models/                      # Model checkpoints (gitignored)
    ├── test/                        # Benchmark I/O (gitignored)
    └── output/                      # Cache trung gian (gitignored)
```

---

## 3. Module pipeline chính

### 3.1. `rag_answer_stage1.py`

**Vai trò:** Orchestrator toàn bộ pipeline Stage 1.

**Class chính:**

| Class / Function | Vai trò |
|------------------|---------|
| `RetrievalEngine` | Embed query, hybrid retrieve, rerank, filter |
| `load_questions()` | Đọc `R2AIStage1DATA.json` |
| `load_llm()` | Load Vi-Qwen2 với 4-bit/bf16 |
| `extract_citations()` | Trích `relevant_docs`, `relevant_articles` từ chunks |
| `generate_answer()` | Gọi LLM với RAG prompt |
| `FilterChunkStats` | Thống kê lọc chunk không hợp lệ |

**Luồng `_retrieval_queries()` (sub-query priority):**

1. Cache file `R2AIStage1_subqueries (1).json`
2. Rule-based runtime (`query_decompose.py`)
3. Subquery index (`subquery_loader.py`)
4. Fallback: câu hỏi gốc

**Regex trích xuất trích dẫn:**

- `_CODE_RE`: mã văn bản (`04/2017/QH14`, `123/NĐ-CP`, …)
- `_ARTICLE_RE`: số điều (`Điều 12`, `Điều 5a`)

### 3.2. `qdrant_config.py`

**Vai trò:** Cấu hình tập trung cho Qdrant và model paths.

| Function | Mô tả |
|----------|-------|
| `load_env()` | Load `.env` via python-dotenv |
| `make_qdrant_client()` | Tạo `QdrantClient` (local/cloud) |
| `get_embed_model_path()` | Resolve embedding model path |
| `get_rerank_model_path()` | Resolve reranker path |
| `query_dense()` | ANN search dense vector |
| `query_sparse_bm25()` | Sparse BM25 search (Cloud) |
| `normalize_chunk_payload()` | Chuẩn hoá payload Qdrant → dict thống nhất |
| `chunk_text()` | Lấy text từ payload (retrieval_text / content_text) |

**Constants:**

- `VECTOR_SIZE = 1024`
- `DEFAULT_COLLECTION = "legal_documents"` (local)
- Production: `vld_business_law_v2` (via `.env`)

### 3.3. `bm25_retrieval.py`

**Vai trò:** Hybrid retrieval — dense + BM25 + RRF.

| Function | Mô tả |
|----------|-------|
| `scroll_all_chunks()` | Scroll toàn bộ corpus từ Qdrant |
| `load_or_build_bm25_index()` | Build/cache BM25 tại `output/bm25_corpus.pkl` |
| `hybrid_retrieve_one()` | Retrieve 1 query: dense + BM25 → RRF |
| `merge_hybrid_results()` | Merge kết quả multi sub-query |

**Default weights:**

```python
DEFAULT_WEIGHT_RRF_DENSE = 0.4
DEFAULT_WEIGHT_RRF_BM25 = 0.6
DEFAULT_RRF_K = 60
```

**Hành vi theo môi trường:**

- **Qdrant Cloud:** BM25 qua `query_sparse_bm25()` (pre-indexed)
- **Qdrant local:** Build BM25 từ scroll corpus, cache pickle

### 3.4. `rerank_retrieval.py`

**Vai trò:** Cross-encoder reranking.

| Function | Mô tả |
|----------|-------|
| `load_reranker()` | Load `CrossEncoder` |
| `rerank_chunks_hybrid()` | Score với primary + sub-query hybrid |
| `dense_hits_to_chunks()` | Convert Qdrant hits → chunk dicts |

**Constants:**

```python
DEFAULT_PRIMARY_WEIGHT = 0.7
DEFAULT_SUB_WEIGHT = 0.3
MAX_RERANK_LENGTH = 2304
```

### 3.5. `query_decompose.py`

**Vai trò:** Phân tách câu hỏi phức tạp thành sub-queries (rule-based).

| Function | Mô tả |
|----------|-------|
| `plan_queries()` | Quyết định số sub-query theo token count |
| `batch_decompose_questions()` | Batch decompose |
| `resolve_queries()` | Áp dụng pattern rules |

**Token thresholds:**

```python
DEFAULT_THRESHOLD_1 = 30   # < 30 tokens → 1 query
DEFAULT_THRESHOLD_2 = 58   # 30–58 → 2 queries; ≥ 58 → 3 queries
```

**Patterns:** câu hỏi kép ("... như thế nào và ... ra sao?"), mệnh đề phụ sau dấu phẩy, v.v.

### 3.6. `subquery_loader.py`

**Vai trò:** Load và index file sub-query precomputed.

| Class / Function | Mô tả |
|------------------|-------|
| `SubquerySpec` | Dataclass: id, queries, policy |
| `load_subquery_index()` | Parse JSON → dict[id → SubquerySpec] |
| `retrieval_queries()` | Lấy query list cho 1 question id |
| `rerank_primary_and_subs()` | Hybrid rerank scoring helper |

---

## 4. Script ingest dữ liệu

Dùng khi tái tạo Qdrant index từ corpus gốc (không bắt buộc cho benchmark nếu dùng Qdrant Cloud).

**Nguồn corpus:** [vohuutridung/vietnamese-legal-documents](https://huggingface.co/datasets/vohuutridung/vietnamese-legal-documents) trên Hugging Face.

| Script | Input | Output |
|--------|-------|--------|
| `ingest_parquet_to_qdrant.py` | `full.parquet` | Qdrant points + vectors |
| `ingest_docx_to_qdrant.py` | Thư mục DOCX | Qdrant points |
| `text_document_builder.py` | Parquet row | Reader Document + tree |
| `document_filters.py` | Metadata | Lọc theo keyword + hiệu lực |
| `build_effectiveness_parquet.py` | vbpl API | `effectiveness.parquet` |
| `qdrant_ingest.py` | Shared helpers | Upsert batch, embed |

**Bộ lọc mặc định** (`document_filters.py`):

- **25 keyword tiêu đề** (logic OR): `công ty`, `thuế`, `doanh nghiệp`, `xử lý`, `lao động`, `đăng ký`, `hợp đồng`, `hồ sơ`, `nhân viên`, `cơ quan`, `kinh doanh`, `yêu cầu`, `quy định`, `nội dung`, `điều kiện`, `thời hạn`, `thông tin`, `hỗ trợ`, `trách nhiệm`, `nghĩa vụ`, `thông báo`, `hàng hóa`, `khách hàng`, `quỹ`, `hóa đơn`
- Cutoff hiệu lực: `2026-03-01`
- Loại bỏ danh mục văn bản hết hiệu lực
- Override keyword: `--keywords-file path/to/keywords.txt`

---

## 5. Module ingest (local)

Scripts tự thêm `Extractor_chunk/`, `Reader/`, `Construct_Tree/` vào `sys.path`. Chỉ cần thiết cho **ingest**, không cần cho chạy benchmark citations.

### 5.1. Extractor_chunk

| Thuộc tính | Giá trị |
|------------|---------|
| Path | `Extractor_chunk/` |
| Python | ≥ 3.9 |
| Dependencies | `tiktoken`, `semantic-text-splitter` |
| Vai trò | Chia văn bản thành chunk theo cấu trúc cây |

### 5.2. Reader

| Thuộc tính | Giá trị |
|------------|---------|
| Path | `Reader/` |
| Dependencies | `python-docx`, `PyMuPDF`, `requests`, `docx2python` |
| Vai trò | Đọc file DOCX/PDF → Document object |
| Phụ thuộc | Construct_Tree |

### 5.3. Construct_Tree

| Thuộc tính | Giá trị |
|------------|---------|
| Path | `Construct_Tree/` |
| Dependencies | `openai`, `requests`, `tiktoken`, `dotenv`, `html-to-markdown[lxml]` |
| Vai trò | Xây cây cấu trúc văn bản pháp luật (Chương → Điều → Khoản) |

---

## 6. Dependencies đầy đủ

### 6.1. Pipeline RAG (`scripts/requirements-rag.txt`)

```
qdrant-client
sentence-transformers
transformers
torch
rank-bm25
sentencepiece
bitsandbytes
accelerate
python-dotenv
pandas
pyarrow
```

### 6.2. Gói nội bộ (khi ingest)

| Package | Dependencies |
|---------|-------------|
| Extractor_chunk | tiktoken, semantic-text-splitter |
| Reader | python-docx, PyMuPDF, requests, docx2python |
| Construct_Tree | openai, requests, tiktoken, dotenv, html-to-markdown[lxml] |

### 6.3. Cài đặt đầy đủ

```bash
pip install -r scripts/requirements-rag.txt
```

---

## 7. File cấu hình triển khai

| File | Mục đích | Commit Git |
|------|----------|------------|
| `.env.example` | Mẫu biến môi trường | Có |
| `.env` | Config thực (Qdrant key, model paths) | Không (gitignored) |
| `.gitignore` | Loại trừ secrets, models, test, output | Có |
| `scripts/qdrant_config.py` | Logic resolve config | Có |
| `scripts/document_filters.py` | Filter rules ingest | Có |
| `models/*/config.json` | HuggingFace model config | Không (gitignored) |

**Biến môi trường quan trọng:**

```env
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION
QDRANT_VECTOR_NAME          # Cloud: dense
QDRANT_SPARSE_VECTOR_NAME   # Cloud: bm25
EMBED_MODEL_PATH
RERANK_MODEL_PATH
```

Override file env: `--env-file .env.cloud` hoặc `QDRANT_ENV_FILE=.env.cloud`.

---

## 8. Luồng xử lý chi tiết

### 8.1. Retrieval (per question)

```
1. Resolve sub-queries (cache → rules → fallback)
2. For each query:
   a. Encode query → dense vector (Vietnamese_Embedding_v2)
   b. Dense ANN search → top 30
   c. BM25 search → top 30
   d. Weighted RRF (0.4 dense + 0.6 BM25) → pool 50
3. If multi sub-query: RRF merge lists
4. Cross-encoder rerank top candidates
   - Score = 0.7 × primary + 0.3 × mean(sub_scores)
5. Take top-K (default 8)
6. Filter: reject chunks missing law code/title
```

### 8.2. Citations extraction

```
1. Take top --llm-top-k chunks (default 2)
2. For each chunk:
   - Extract document_number, document_title
   - Extract article_no (Điều X, Phụ lục, ...)
3. Deduplicate → relevant_docs[], relevant_articles[]
4. Write JSON output
```

### 8.3. LLM generation (full pipeline)

```
1. Build context string from top chunks (max 3500 chars)
2. Format Vi-Qwen2 RAG prompt
3. Generate with temperature=0.1, max_new_tokens=512
4. Extract citations from same chunks
5. Write answer + citations to JSON
```

---

## 9. Các chế độ chạy (CLI)

| Mode | Flags | Mô tả |
|------|-------|-------|
| Citations only | `--citations-only` | Benchmark Stage 1, không LLM |
| Full pipeline | (default) | Retrieve + LLM answer |
| Retrieve only | `--retrieve-only` | Cache chunks, không generate |
| From cache | `--skip-retrieve` | Đọc retrieved cache |
| Resume | `--skip-answered` | Bỏ qua IDs đã có trong output |
| Slice | `--start-id`, `--limit` | Chạy subset |

**Disable components:**

| Flag | Effect |
|------|--------|
| `--no-bm25` | Chỉ dense search |
| `--no-rerank` | Bỏ cross-encoder |
| `--no-subquery` | Không phân tách sub-query |
| `--no-4bit` | LLM bf16 thay vì 4-bit |

---

## 10. Script tiện ích

| Script | Mục đích |
|--------|----------|
| `query_qdrant.py` | Test 1 câu hỏi, in top chunks |
| `generate_subqueries.py` | Tạo file sub-query JSON |
| `rerank_hybrid.py` | Rerank standalone trên cache RRF |
| `check_expired_law_codes.py` | Audit mã văn bản hết hiệu lực trong Qdrant |
| `export_filtered_metadata.py` | Export metadata đã lọc |
| `cleanup_*.py` | Dọn DOCX crawl (cần VbplCrawler) |

---

## 11. Output và cache files

| Path | Tạo bởi | Mục đích |
|------|---------|----------|
| `test/R2AIStage1_citations*.json` | `--citations-only` | Kết quả benchmark |
| `test/R2AIStage1_answers.json` | full pipeline | Answer + citations |
| `test/R2AIStage1_retrieved.json` | `--retrieve-only` | Cache retrieve |
| `output/bm25_corpus.pkl` | auto (local Qdrant) | BM25 index cache |

---

## 12. Mở rộng và tuỳ chỉnh

**Thay đổi trọng số RRF:** sửa constants trong `bm25_retrieval.py`.

**Thay đổi rerank hybrid weight:** sửa `DEFAULT_PRIMARY_WEIGHT`, `DEFAULT_SUB_WEIGHT` trong `rerank_retrieval.py`.

**Thêm sub-query pattern:** mở rộng rules trong `query_decompose.py`.

**Thay LLM:** `--llm-model <path>` (model phải tương thích Qwen2 chat template).

**Thay embedding/reranker:** `--embed-model`, `--rerank-model` hoặc env vars.

---

## 13. Giới hạn đã biết

- Không có Stage 2 trong repo này
- Flag `--hybrid-rerank` tồn tại nhưng hybrid rerank luôn bật trong code
- `VbplCrawler/` và `data/` nằm ngoài repo (gitignored)
- Full pipeline cần GPU CUDA; citations-only có thể chạy CPU (chậm)
- Context LLM giới hạn 3500 chars mặc định (có thể tăng bằng `--max-context-chars`)

---

## 14. Tài liệu liên quan

- [README.md](../README.md) — Hướng dẫn cài đặt và chạy
- [DATASET.md](DATASET.md) — Mô tả dữ liệu
- [MODELS.md](MODELS.md) — Mô tả model
- [PIPELINE_OVERVIEW_29-06.md](PIPELINE_OVERVIEW_29-06.md) — Tổng quan pipeline nội bộ
