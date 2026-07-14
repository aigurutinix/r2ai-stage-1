# R2AI — Hệ thống RAG trả lời câu hỏi pháp luật Việt Nam (Stage 1)

Hệ thống **Retrieval-Augmented Generation (RAG)** trả lời câu hỏi pháp luật doanh nghiệp tiếng Việt cho benchmark **R2AI Stage 1**. Pipeline gồm: phân tách truy vấn con → truy xuất lai (dense + BM25) trên Qdrant → rerank bằng cross-encoder → trích xuất trích dẫn hoặc sinh câu trả lời bằng LLM.

---

## Mục lục

1. [Yêu cầu môi trường](#1-yêu-cầu-môi-trường)
2. [Cài đặt](#2-cài-đặt)
3. [Tải dữ liệu và model](#3-tải-dữ-liệu-và-model)
4. [Cấu hình](#4-cấu-hình)
5. [Chạy pipeline](#5-chạy-pipeline)
6. [Kết quả đầu ra](#6-kết-quả-đầu-ra)
7. [Tài liệu chi tiết](#7-tài-liệu-chi-tiết)
8. [Cấu trúc thư mục](#8-cấu-trúc-thư-mục)
9. [Khắc phục sự cố](#9-khắc-phục-sự-cố)

---

## 1. Yêu cầu môi trường

| Thành phần | Phiên bản khuyến nghị |
|------------|----------------------|
| **Hệ điều hành** | Linux (Ubuntu 22.04+) hoặc tương đương |
| **Python** | ≥ 3.9 |
| **CUDA** | 11.8+ (GPU NVIDIA, khuyến nghị ≥ 8 GB VRAM cho full pipeline) |
| **RAM** | ≥ 16 GB |
| **Ổ đĩa** | ≥ 15 GB trống (models + cache) |

**Phần mềm bắt buộc:**

- Python 3.9+
- pip
- Git
- (Tuỳ chọn) Docker — nếu chạy Qdrant local

**Phần cứng tối thiểu theo chế độ:**

| Chế độ | GPU | Ghi chú |
|--------|-----|---------|
| `--citations-only` | Khuyến nghị | Chỉ cần embed + rerank; không load LLM |
| Full pipeline | Bắt buộc | Vi-Qwen2-1.5B-RAG chạy 4-bit trên CUDA |

---

## 2. Cài đặt

### 2.1. Clone mã nguồn

```bash
git clone <URL_REPO_R2AI> R2AI
cd R2AI
```

### 2.2. Tạo môi trường ảo (khuyến nghị)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2.3. Cài dependencies chính

```bash
pip install --upgrade pip
pip install -r scripts/requirements-rag.txt
```

**Danh sách thư viện chính** (`scripts/requirements-rag.txt`):

| Package | Mục đích |
|---------|----------|
| `qdrant-client` | Kết nối vector database Qdrant |
| `sentence-transformers` | Embedding + cross-encoder reranker |
| `transformers` | LLM (Vi-Qwen2) |
| `torch` | Inference trên GPU/CPU |
| `rank-bm25` | BM25 local (khi không dùng Qdrant Cloud sparse) |
| `bitsandbytes`, `accelerate` | Quantization 4-bit cho LLM |
| `python-dotenv` | Đọc file `.env` |
| `pandas`, `pyarrow` | Đọc/ghi Parquet (ingest) |

### 2.4. Module ingest (chỉ cần khi ingest dữ liệu)

Các script ingest tự thêm `Extractor_chunk/`, `Reader/`, `Construct_Tree/` vào `sys.path` — không cần `pip install -e`.

---

## 3. Tải dữ liệu và model

### 3.1. Dữ liệu benchmark

Tải bộ dữ liệu test và file phụ trợ, giải nén vào thư mục `test/`:

| File | Mô tả |
|------|-------|
| `R2AIStage1DATA.json` | 2.000 câu hỏi benchmark (bắt buộc) |
| `R2AIStage1_subqueries (1).json` | Sub-query đã tính sẵn (tuỳ chọn, tăng tốc) |
| `R2AIStage1_retrieved.json` | Cache retrieve (tuỳ chọn, dùng với `--skip-retrieve`) |

**Link tải dữ liệu (Google Drive):**

> 🔗 **Dataset R2AI Stage 1:** [Google Drive — R2AI Dataset](https://drive.google.com/file/d/1pedkNrz2mKJ7GmCSK79COKWvT4sokRWP/view?usp=sharing)

Chi tiết cấu trúc dữ liệu: xem [docs/DATASET.md](docs/DATASET.md).

### 3.2. Model checkpoints

Tải ba model vào thư mục `models/` (hoặc để HuggingFace tự tải khi chạy lần đầu):

```bash
mkdir -p models

# Embedding (1024-dim, fine-tuned từ BGE-M3)
huggingface-cli download AITeamVN/Vietnamese_Embedding_v2 \
  --local-dir models/Vietnamese_Embedding_v2

# Cross-encoder reranker
huggingface-cli download AITeamVN/Vietnamese_Reranker \
  --local-dir models/Vietnamese_Reranker

# LLM RAG (1.5B, dùng khi chạy full pipeline)
huggingface-cli download AITeamVN/Vi-Qwen2-1.5B-RAG \
  --local-dir models/Vi-Qwen2-1.5B-RAG
```

**Link tải checkpoint (Google Drive — bản đóng gói sẵn):**

> 🔗 **Model checkpoints:** [Google Drive — R2AI Models](https://drive.google.com/drive/folders/YOUR_MODELS_FOLDER_ID)
>
> HuggingFace: [AITeamVN/Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding) · [AITeamVN/Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker) · [AITeamVN/Vi-Qwen2-1.5B-RAG](https://huggingface.co/AITeamVN/Vi-Qwen2-1.5B-RAG)

Chi tiết model: xem [docs/MODELS.md](docs/MODELS.md).

### 3.3. Vector database (Qdrant)

Corpus pháp luật doanh nghiệp được lưu trên **Qdrant Cloud**, collection `vld_business_law_v2`. Thông tin truy cập được cung cấp trong file `.env` (xem mục 4).

**Nguồn corpus:** lọc từ [vohuutridung/vietnamese-legal-documents](https://huggingface.co/datasets/vohuutridung/vietnamese-legal-documents) (518.255 văn bản từ thuvienphapluat.vn) bằng **25 keyword tiêu đề** + lọc hiệu lực (cutoff 2026-03-01). Chi tiết: [docs/DATASET.md](docs/DATASET.md).

Nếu cần tái tạo index local, xem hướng dẫn ingest trong [docs/CODE.md](docs/CODE.md).

---

## 4. Cấu hình

### 4.1. Tạo file môi trường

```bash
cp .env.example .env
```

Chỉnh sửa `.env` theo môi trường triển khai:

**Qdrant Cloud (production — khuyến nghị):**

```env
QDRANT_URL=https://YOUR-CLUSTER.region.aws.cloud.qdrant.io
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION=vld_business_law_v2
QDRANT_VECTOR_NAME=dense
QDRANT_SPARSE_VECTOR_NAME=bm25

EMBED_MODEL_PATH=models/Vietnamese_Embedding_v2
RERANK_MODEL_PATH=models/Vietnamese_Reranker
```

**Qdrant local (phát triển):**

```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=legal_documents
# Không cần QDRANT_API_KEY, QDRANT_VECTOR_NAME, QDRANT_SPARSE_VECTOR_NAME
```

### 4.2. Các file cấu hình quan trọng

| File | Vai trò |
|------|---------|
| `.env` | URL Qdrant, API key, tên collection, đường dẫn model |
| `.env.example` | Mẫu cấu hình |
| `scripts/qdrant_config.py` | Resolver cấu hình Qdrant và model |
| `scripts/document_filters.py` | Bộ lọc văn bản pháp luật (ingest) |
| `scripts/bm25_retrieval.py` | Trọng số RRF (dense 0.4, BM25 0.6) |
| `scripts/rerank_retrieval.py` | Trọng số hybrid rerank (primary 0.7, sub 0.3) |

> **Lưu ý:** Không commit file `.env` lên Git (đã được liệt kê trong `.gitignore`).

---

## 5. Chạy pipeline

Tất cả lệnh chạy từ **thư mục gốc** `R2AI/`:

```bash
cd R2AI
source .venv/bin/activate   # nếu dùng venv
```

### 5.1. Chế độ citations only (benchmark Stage 1)

Chỉ trích xuất `relevant_docs` và `relevant_articles`, không sinh câu trả lời LLM:

```bash
python scripts/rag_answer_stage1.py \
  --citations-only \
  --llm-top-k 3 \
  --output test/R2AIStage1_citations_llm_top3.json
```

### 5.2. Full pipeline (retrieve + generate)

Sinh câu trả lời đầy đủ bằng Vi-Qwen2-1.5B-RAG:

```bash
python scripts/rag_answer_stage1.py \
  --output test/R2AIStage1_answers.json
```

### 5.3. Retrieve only (cache chunks)

```bash
python scripts/rag_answer_stage1.py \
  --retrieve-only \
  --output test/R2AIStage1_retrieved.json
```

### 5.4. Citations từ cache (bỏ qua Qdrant)

```bash
python scripts/rag_answer_stage1.py \
  --citations-only \
  --skip-retrieve \
  --output test/R2AIStage1_citations.json
```

### 5.5. Resume — tiếp tục từ output cũ

```bash
python scripts/rag_answer_stage1.py \
  --citations-only \
  --skip-answered \
  --output test/R2AIStage1_citations.json
```

### 5.6. Chạy thử trên một phần dataset

```bash
python scripts/rag_answer_stage1.py \
  --citations-only \
  --start-id 1 \
  --limit 10 \
  --output test/sample_citations.json
```

### 5.7. Kiểm tra truy vấn đơn

```bash
python scripts/query_qdrant.py "Doanh nghiệp nhỏ được hưởng ưu đãi gì?" --use-bm25
```

### Tham số CLI quan trọng

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--citations-only` | tắt | Chỉ trích xuất trích dẫn, không LLM |
| `--llm-top-k` | 2 | Số chunk đưa vào trích dẫn / LLM |
| `--top-k` | 8 | Số chunk sau rerank |
| `--retrieve-pool` | 30 | Pool dense/BM25 trước RRF |
| `--rrf-top-k` | 50 | Pool sau RRF |
| `--no-bm25` | tắt | Tắt BM25, chỉ dense search |
| `--no-rerank` | tắt | Tắt cross-encoder reranker |
| `--no-subquery` | tắt | Tắt phân tách sub-query |
| `--skip-retrieve` | tắt | Đọc cache thay vì query Qdrant |
| `--skip-answered` | tắt | Bỏ qua câu đã có trong output |
| `--start-id`, `--limit` | 1, none | Giới hạn phạm vi câu hỏi |
| `--env-file` | `.env` | File cấu hình môi trường |

---

## 6. Kết quả đầu ra

### Citations only

```json
{
  "id": 1,
  "question": "Các cơ sở ươm tạo và khu làm việc chung được hưởng những chính sách hỗ trợ nào về thuế và đất đai?",
  "answer": "",
  "relevant_docs": [
    "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa 2017"
  ],
  "relevant_articles": [
    "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa 2017|Điều 12"
  ]
}
```

**Quy ước định dạng:**

- `relevant_docs`: `{mã_văn_bản}|{tên_văn_bản}`
- `relevant_articles`: `{mã_văn_bản}|{tên_văn_bản}|{Điều/Phụ lục}`

### Full pipeline

Cùng schema, trường `answer` chứa đoạn văn tiếng Việt do LLM sinh.

---

## 7. Tài liệu chi tiết

| Tài liệu | Nội dung |
|----------|----------|
| [docs/DATASET.md](docs/DATASET.md) | Nguồn dữ liệu, cấu trúc, định dạng, hướng dẫn truy cập |
| [docs/MODELS.md](docs/MODELS.md) | Model embedding, reranker, LLM; checkpoint; hướng dẫn tải |
| [docs/CODE.md](docs/CODE.md) | Kiến trúc mã nguồn, module, script, luồng xử lý |
| [docs/PIPELINE_OVERVIEW_29-06.md](docs/PIPELINE_OVERVIEW_29-06.md) | Tổng quan pipeline (nội bộ, cập nhật 29/06/2026) |

---

## 8. Cấu trúc thư mục

```
.                              # Thư mục gốc repo
├── README.md                    # Tài liệu này
├── docs/                        # Tài liệu chi tiết
│   ├── DATASET.md
│   ├── MODELS.md
│   ├── CODE.md
│   └── PIPELINE_OVERVIEW_29-06.md
└── R2AI/                        # Mã nguồn pipeline (cd vào đây khi chạy)
    ├── .env.example             # Mẫu cấu hình
    ├── .gitignore
    ├── scripts/
    │   ├── rag_answer_stage1.py     # ★ Pipeline chính
    │   ├── qdrant_config.py         # Cấu hình Qdrant + model
    │   ├── bm25_retrieval.py        # BM25 + RRF fusion
    │   ├── rerank_retrieval.py      # Cross-encoder rerank
    │   ├── query_decompose.py       # Phân tách sub-query
    │   ├── subquery_loader.py       # Load cache sub-query
    │   ├── query_qdrant.py          # Test truy vấn đơn
    │   ├── ingest_parquet_to_qdrant.py
    │   ├── ingest_docx_to_qdrant.py
    │   ├── text_document_builder.py
    │   ├── document_filters.py
    │   └── requirements-rag.txt
    ├── models/                  # Checkpoints (gitignored)
    │   ├── Vietnamese_Embedding_v2/
    │   ├── Vietnamese_Reranker/
    │   └── Vi-Qwen2-1.5B-RAG/
    ├── test/                    # Benchmark I/O (gitignored)
    │   ├── R2AIStage1DATA.json
    │   └── R2AIStage1_citations*.json
    ├── Extractor_chunk/         # Module chunking văn bản theo cây
    ├── Reader/                  # Module đọc DOCX/PDF
    └── Construct_Tree/          # Module xây cây văn bản pháp luật
```

> Các đường dẫn như `test/`, `models/`, `scripts/` trong tài liệu là **tương đối so với `R2AI/`** khi chạy pipeline.

---

## 9. Khắc phục sự cố

### Lỗi kết nối Qdrant

```
Connection refused / Unauthorized
```

- Kiểm tra `QDRANT_URL` và `QDRANT_API_KEY` trong `.env`
- Với Qdrant Cloud: đảm bảo đã set `QDRANT_VECTOR_NAME=dense` và `QDRANT_SPARSE_VECTOR_NAME=bm25`

### Out of Memory (GPU)

- Dùng `--citations-only` để bỏ qua LLM
- Giảm `--retrieve-batch`, `--gen-batch`, `--rerank-batch`
- Giảm `--llm-top-k` và `--max-context-chars`
- Thêm `--no-4bit` chỉ khi VRAM đủ lớn (>12 GB)

### Model không tìm thấy

```
OSError: ... does not appear to have a file named config.json
```

- Tải model theo mục 3.2 hoặc set `EMBED_MODEL_PATH` / `RERANK_MODEL_PATH` trỏ tới HuggingFace ID

### File benchmark không tồn tại

```
FileNotFoundError: test/R2AIStage1DATA.json
```

- Tải dataset theo mục 3.1 và đặt vào `test/`

---

## Giấy phép

Mã nguồn và model tuân theo giấy phép Apache 2.0 (trừ khi có ghi chú khác trên từng model card HuggingFace).

## Liên hệ

Đội phát triển R2AI — thay `<email>` bằng email liên hệ thực tế của nhóm.
