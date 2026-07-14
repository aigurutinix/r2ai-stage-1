# Chatbot Văn bản Pháp luật VN

Hệ thống RAG (Retrieval-Augmented Generation) tra cứu và hỏi đáp văn bản quy phạm pháp luật Việt Nam — xây dựng cho cuộc thi **Truy hồi & Hỏi đáp Văn bản Pháp luật Tiếng Việt** (xem [`project.md`](project.md)).

**Phạm vi corpus:** Luật Doanh nghiệp & văn bản liên quan SME (thuế, lao động, hợp đồng, đầu tư...).

---

## Kiến trúc tổng quan

```
Câu hỏi → [Query Analyzer] → [BM25 + Dense Retrieval] → [Reranker]
         → [RAG Pipeline] → [LLM (Qwen3.5:9b)] → Câu trả lời + Citation
```

| Thành phần | Công nghệ |
| --- | --- |
| **LLM** | Qwen3.5:9b qua Ollama (model tùy chỉnh `qwen-vbpl`) |
| **Embedding** | bge-m3 qua Ollama (1024-dim, tiếng Việt tốt) |
| **Vector store** | Qdrant embedded (`file://`, không cần Docker) |
| **Backend** | FastAPI + SSE streaming |
| **Dataset** | [tmquan/vbpl-vn](https://huggingface.co/datasets/tmquan/vbpl-vn) (~250k chunk sau lọc scope) |

---

## Yêu cầu môi trường

| Phần mềm | Phiên bản | Ghi chú |
| --- | --- | --- |
| **Python** | 3.12+ | Kiểm tra: `python --version` |
| **Ollama** | 0.4+ | [ollama.com](https://ollama.com) — chạy LLM & embedding local |
| **Git** | bất kỳ | Để clone repo |
| **Docker** (tùy chọn) | 24+ | Chỉ cần nếu muốn chạy Qdrant server riêng |
| **RAM** | ≥ 16 GB | Qwen3.5:9b Q4 cần ~6 GB VRAM/RAM; bge-m3 ~1.5 GB thêm |
| **Disk** | ≥ 20 GB | Dataset HuggingFace cache + Qdrant storage |

---

## Cấu trúc thư mục

```
.
├── .env.example            # Mẫu cấu hình — chép thành .env
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Qdrant server (tùy chọn)
├── project.md              # Thể lệ & tiêu chí cuộc thi
├── backend/
│   ├── main.py             # FastAPI app: /chat (SSE), /chat_sync, /health
│   ├── config.py           # Load .env qua pydantic-settings
│   ├── llm.py              # LLM client (OpenAI-compatible)
│   ├── embed.py            # Embedding client
│   ├── qdrant_store.py     # Qdrant wrapper
│   ├── rag.py              # RAG pipeline chính
│   ├── bm25.py             # BM25 hybrid search
│   ├── reranker.py         # Reranking
│   ├── query_analyzer.py   # Phân tích câu hỏi
│   ├── decompose.py        # Query decomposition
│   └── prompts.py          # Prompt templates (ép citation Điều X)
├── ingest/
│   ├── run_vbpl.py         # Ingestion từ tmquan/vbpl-vn (script chính)
│   ├── parse_vbpl.py       # Parse row → cấu trúc Điều/Khoản
│   ├── chunk.py            # Chunking theo Điều
│   ├── embed_load.py       # Embed + upload Qdrant
│   ├── scope.py            # Bộ lọc domain DN/SME
│   └── run.py              # Ingestion từ dataset cũ (th1nhng0, không dùng)
├── scripts/
│   ├── Modelfile.qwen-vbpl # Ollama Modelfile cho LLM
│   ├── test_llm.ps1        # Kiểm tra kết nối LLM
│   ├── test_qdrant.ps1     # Kiểm tra Qdrant
│   └── test_query.ps1      # Test query end-to-end
├── tests/
│   ├── f2_eval.py          # Tính F2 macro (tiêu chí cuộc thi)
│   ├── run_competition.py  # Chạy batch submission
│   ├── build_submission_v6.py
│   └── ...
├── docs/                   # Báo cáo đánh giá, nhật ký thay đổi
├── data/                   # Dataset cache + BM25 index (gitignored)
└── qdrant_storage/         # Qdrant data khi dùng Docker (gitignored)
```

---

## Cài đặt từ đầu (lần đầu)

### Bước 1 — Clone repo và cấu hình môi trường

```bash
git clone <repo-url>
cd Chatbot_VBPL

cp .env.example .env
# Mở .env, kiểm tra và điều chỉnh nếu cần (xem mục Cấu hình bên dưới)
```

### Bước 2 — Tạo Python virtual environment

```bash
python -m venv .venv

# Linux / macOS:
source .venv/bin/activate

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Bước 3 — Cài đặt và khởi động Ollama

Tải Ollama từ [ollama.com](https://ollama.com) và cài đặt, sau đó:

```bash
# Tải model embedding (bge-m3 ~670 MB):
ollama pull bge-m3

# Tải LLM base (Qwen3.5 9B ~5.5 GB):
ollama pull qwen3.5:9b

# Tạo model tùy chỉnh qwen-vbpl (tăng context window lên 16k, cần thiết cho RAG):
ollama create qwen-vbpl -f scripts/Modelfile.qwen-vbpl

# Kiểm tra Ollama đang chạy:
ollama list
# Kết quả phải có: bge-m3, qwen3.5:9b, qwen-vbpl
```

> **Lưu ý:** Ollama mặc định chạy tại `http://localhost:11434`. Cấu hình này đã được đặt sẵn trong `.env.example`.

### Bước 4 — Ingest dataset (chạy 1 lần, ~1–3 giờ)

Script tải dataset `tmquan/vbpl-vn` từ HuggingFace, lọc phạm vi DN/SME, parse → chunk theo Điều, embed bằng bge-m3 rồi nạp vào Qdrant.

```bash
# Đặt PYTHONUTF8 để tránh lỗi encoding tiếng Việt trên Windows:
export PYTHONUTF8=1     # Linux/macOS
# $env:PYTHONUTF8=1    # Windows PowerShell

# Chạy thử nhỏ trước (100 văn bản) để xác nhận pipeline hoạt động:
PYTHONUTF8=1 PYTHONPATH=. python -m ingest.run_vbpl --limit 100 --recreate

# Ingest toàn bộ corpus (~250k chunk):
PYTHONUTF8=1 PYTHONPATH=. python -m ingest.run_vbpl --recreate
```

**Tùy chọn `--no-keyword`:** Bỏ lọc từ khóa, giữ tất cả văn bản trung ương thuộc loại lõi (corpus lớn hơn, thời gian embed lâu hơn):

```bash
PYTHONUTF8=1 PYTHONPATH=. python -m ingest.run_vbpl --recreate --no-keyword
```

Dataset sẽ được cache tại `./data/hf_cache/` — lần chạy tiếp theo không tải lại từ internet.

### Bước 5 — Build BM25 index (tùy chọn, cải thiện hybrid search)

```bash
PYTHONUTF8=1 PYTHONPATH=. python scripts/build_bm25.py
```

### Bước 6 — Khởi động backend

```bash
# Chạy trực tiếp (khuyến nghị — tránh xung đột lock Qdrant embedded):
PYTHONUTF8=1 PYTHONPATH=. python -m backend.main

# Backend sẽ lắng nghe tại:
# http://127.0.0.1:8000
```

> **Quan trọng:** Qdrant embedded (`file://`) chỉ cho phép **1 process** truy cập cùng lúc. Không chạy song song backend và script ingest/test khác.

### Bước 7 — Kiểm tra hoạt động

```bash
# Kiểm tra health:
curl http://127.0.0.1:8000/health

# Kết quả mẫu:
# {"status":"ok","collection":"vbpl_v2","doc_chunks":247831,"llm_model":"qwen-vbpl","embed_model":"bge-m3"}

# Test query:
curl -X POST http://127.0.0.1:8000/chat_sync \
  -H "Content-Type: application/json" \
  -d '{"query": "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào theo Luật Hỗ trợ DNNVV?"}'
```

---

## Cấu hình (`.env`)

Sao chép `.env.example` thành `.env` và chỉnh sửa theo môi trường:

```bash
# ===== LLM =====
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen-vbpl          # Model tùy chỉnh với context 16k
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.0
LLM_REASONING_EFFORT=none    # Tắt reasoning chain của Qwen3.5

# ===== Embedding =====
EMBED_API_KEY=ollama
EMBED_BASE_URL=http://localhost:11434/v1
EMBED_MODEL=bge-m3
EMBED_DIM=1024               # Phải khớp với model, không tự ý thay đổi sau ingest

# ===== Qdrant =====
# Embedded (file://) — không cần Docker:
QDRANT_URL=file:///đường/dẫn/tuyệt/đối/đến/data/qdrant_local
# Hoặc Qdrant server (Docker): QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=vbpl_v2

# ===== Retrieval =====
TOP_K=10
RERANK_TOP_K=5

# ===== Ingestion =====
HF_DATASET=tmquan/vbpl-vn    # Dataset chính (không dùng th1nhng0)
HF_CACHE_DIR=./data/hf_cache
EMBED_BATCH_SIZE=64
INGEST_LIMIT=0               # 0 = toàn bộ
INGEST_SCOPE_FILTER=true     # true = chỉ giữ domain DN/SME

# ===== Backend =====
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
CORS_ORIGINS=http://localhost:3000
```

> **Lưu ý `QDRANT_URL`:** Trên Windows, đường dẫn có thể là `file:///D:/project/data/qdrant_local`. Trên Linux/macOS: `file:///home/user/project/data/qdrant_local`. Phải là đường dẫn tuyệt đối.

---

## API Endpoints

| Endpoint | Method | Mô tả |
| --- | --- | --- |
| `GET /health` | GET | Kiểm tra trạng thái, số chunk trong Qdrant |
| `POST /chat` | POST | Trả lời streaming qua SSE |
| `POST /chat_sync` | POST | Trả lời đồng bộ (JSON) |
| `POST /v1/chat/completions` | POST | OpenAI-compatible (dùng với open-webui) |

**Request body (`/chat` và `/chat_sync`):**

```json
{
  "query": "Câu hỏi pháp luật của bạn",
  "top_k": 10
}
```

---

## Chạy Qdrant qua Docker (tùy chọn)

Dùng khi cần chạy nhiều process đồng thời (backend + ingest) hoặc corpus lớn hơn:

```bash
docker compose up -d qdrant
# Dashboard: http://localhost:6333/dashboard
```

Sau đó đổi trong `.env`:

```bash
QDRANT_URL=http://localhost:6333
```

Và bật `--reload` khi chạy backend:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

---

## Đánh giá (cuộc thi)

```bash
# Chạy batch evaluation trên tập câu hỏi cuộc thi:
PYTHONUTF8=1 PYTHONPATH=. python tests/run_competition.py

# Tính F2 macro:
PYTHONUTF8=1 PYTHONPATH=. python tests/f2_eval.py

# Kết quả lưu tại: docs/EVAL.md, docs/COMPETITION_EVAL.md
```

---

## Tái hiện (Reproduce) toàn bộ hệ thống

Thứ tự tối thiểu để chạy lại từ đầu trên máy mới:

1. Clone repo + `cp .env.example .env` + chỉnh `QDRANT_URL`
2. `pip install -r requirements.txt`
3. `ollama pull bge-m3 && ollama pull qwen3.5:9b`
4. `ollama create qwen-vbpl -f scripts/Modelfile.qwen-vbpl`
5. `PYTHONUTF8=1 PYTHONPATH=. python -m ingest.run_vbpl --recreate`
6. `PYTHONUTF8=1 PYTHONPATH=. python -m backend.main`
7. `curl http://127.0.0.1:8000/health` — xác nhận `doc_chunks > 0`

---

## Ghi chú kỹ thuật

- **PYTHONUTF8=1** bắt buộc trên Windows (console cp1252 gây lỗi chữ Việt). Linux/macOS thường không cần.
- **Chunking theo Điều:** Mỗi Điều luật là một chunk — đây là đơn vị ngữ nghĩa phù hợp với cách tổ chức VBPL VN.
- **Citation bắt buộc:** Prompt ép LLM trích dẫn format `[Điều X, Khoản Y, Tên văn bản (số ký hiệu)]` — tiêu chí chấm điểm của cuộc thi dựa vào pattern `Điều X`.
- **F2 ưu tiên Recall:** `TOP_K=10` để bao phủ đủ căn cứ pháp lý, `RERANK_TOP_K=5` để lọc lại trước khi đưa vào prompt.
- **Không fine-tune:** Hệ thống RAG thuần — không có tập train/dev, phù hợp với thể lệ cuộc thi.
