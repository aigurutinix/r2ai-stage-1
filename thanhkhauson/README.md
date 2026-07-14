# R2AI2026 Legal Assistant

Hệ thống truy hồi và hỏi đáp pháp lý tiếng Việt cho bài toán R2AI2026 BUILD AI LEGAL ASSISTANT. Pipeline chính dùng hybrid retrieval trên Elasticsearch/Qdrant, rerank bằng ViRanker, sau đó sinh câu trả lời có căn cứ bằng Qwen3.5-9B qua endpoint OpenAI-compatible.

Đầu ra cuối cùng của cuộc thi là `results.json`:

```json
{
  "id": 1,
  "question": "...",
  "answer": "...",
  "relevant_docs": ["<document_number>|<doc_name>"],
  "relevant_articles": ["<document_number>|<doc_name>|<article_no>"]
}
```

Các lệnh bên dưới giả định đang chạy từ thư mục gốc dự án `Legal_assistant`.

## Cấu Trúc Thư Mục

```text
Legal_assistant/
  Agents/                 # LangGraph agent, retrieval tool, memory, logging
  data/                   # Dữ liệu giải nén ngoài repo
  dbs/start_dbs.sh        # Khởi động Elasticsearch và Qdrant trên máy cục bộ
  docs/                   # Tài liệu mô tả dữ liệu, model, checklist nộp bài
  model_cache/            # Checkpoint tải ngoài repo
  build_hybrid_index.py   # Nạp chunk vào Elasticsearch/Qdrant
  pre_tokenize.py         # Tạo trường tokenized_content_search cho BM25
  evaluation.py           # Chạy suy luận hàng loạt và tạo submission.zip
  start_eval.sh           # Phục vụ Qwen3.5-9B bằng vLLM trên port 8000
  requirements.txt
```

Không commit `data/`, `model_cache/`, file sinh khi chạy trong `dbs/`, kết quả evaluation, checkpoint hoặc zip dữ liệu. Các tệp này được tải qua link và có thể build lại.

## Môi Trường

Môi trường đã dùng để chạy:

- Linux
- Python 3.12
- CUDA GPU cho Qwen, embedding và reranker
- Java 1.8+ cho VnCoreNLP
- Elasticsearch trên máy cục bộ, port `9201`
- Qdrant trên máy cục bộ, port `6333`

Cài các thư viện phụ thuộc:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Kiểm tra Java và VnCoreNLP:

```bash
java -version
python -m pip show py-vncorenlp
```

Tạo file cấu hình local:

```bash
cp .env.example .env
```

Khi chạy evaluation hàng loạt, nên giữ:

```bash
CHECKPOINTER_BACKEND=none
LANGSMITH_TRACING=false
```

## Dữ Liệu

Nguồn dữ liệu:

- HuggingFace dataset: https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents
- Folder sử dụng: https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents/tree/main/legacy
- Hai file trong `data/raw_law/law_only/effective/` là dữ liệu crawl Bộ luật/Luật từ `vbpl.vn`, trong đó nội dung điều đã được cập nhật theo các văn bản sửa đổi.

Đặt file dữ liệu đã tiền xử lý vào thư mục gốc dự án:

```text
preprocessed_active_legal_data_20260630.zip
```

Giải nén:

```bash
unzip preprocessed_active_legal_data_20260630.zip -d .
```

Sau khi giải nén cần có 4 file:

```text
data/raw_law/law_only/others_doc/parsed_others_doc_database.jsonl
data/raw_law/law_only/others_doc/metadata_merged_active.jsonl
data/raw_law/law_only/effective/law_luoc_do_merged_dedup.jsonl
data/raw_law/law_only/effective/parsed_law_database.jsonl
```

Thống kê bản zip hiện tại:

| File | Số dòng | Ghi chú |
| --- | ---: | --- |
| `parsed_others_doc_database.jsonl` | 4,836,218 | Chunk điều/mục của văn bản khác |
| `metadata_merged_active.jsonl` | 198,674 | Chỉ còn `effect_status = "In effect"` |
| `law_luoc_do_merged_dedup.jsonl` | 300 | Metadata/lược đồ Bộ luật/Luật |
| `parsed_law_database.jsonl` | 26,591 | Chunk điều của Bộ luật/Luật |

Cấu trúc dữ liệu chi tiết nằm trong [docs/data_description.md](docs/data_description.md).

## Checkpoint Và Model

Tải checkpoint vào `model_cache/`:

```bash
mkdir -p model_cache

huggingface-cli download Qwen/Qwen3.5-9B \
  --local-dir model_cache/Qwen3.5-9B

huggingface-cli download namdp-ptit/ViRanker \
  --local-dir model_cache/ViRanker

huggingface-cli download mainguyen9/vietlegal-harrier-0.6b \
  --local-dir model_cache/vietlegal-harrier-0.6b
```

Khởi tạo VnCoreNLP nếu chưa có:

```bash
python - <<'PY'
import os
import py_vncorenlp

save_dir = "model_cache/vncorenlp"
os.makedirs(save_dir, exist_ok=True)
py_vncorenlp.download_model(save_dir=save_dir)
PY
```

Đường dẫn checkpoint mặc định:

```text
model_cache/Qwen3.5-9B/
model_cache/ViRanker/
model_cache/vietlegal-harrier-0.6b/
model_cache/vncorenlp/
```

Mô tả model chi tiết nằm trong [docs/model_description.md](docs/model_description.md).

## Khởi Động Elasticsearch Và Qdrant

Chạy service trên máy cục bộ:

```bash
bash dbs/start_dbs.sh
```

Kiểm tra:

```bash
curl http://localhost:9201
curl http://localhost:6333
```

Index/collection mặc định:

```text
Elasticsearch index: legal_chunks
Qdrant collection:  legal_chunks
```

## Build Lại Chỉ Mục

Tạo file tokenized cho BM25:

```bash
VNCORENLP_PRETOKENIZE_WORKERS=8 python pre_tokenize.py
```

Lệnh này tạo:

```text
data/raw_law/law_only/effective/parsed_law_database_tokenized.jsonl
data/raw_law/law_only/others_doc/parsed_others_doc_database_tokenized.jsonl
```

Nạp dữ liệu vào Elasticsearch/Qdrant:

```bash
INDEX_DEVICE=cuda:0 python build_hybrid_index.py
```

Nếu cần chia việc theo máy/GPU:

```bash
START_LINE=0 END_LINE=2500000 INDEX_DEVICE=cuda:0 python build_hybrid_index.py
START_LINE=2500000 INDEX_DEVICE=cuda:1 python build_hybrid_index.py
```

`build_hybrid_index.py` sẽ:

1. Nạp metadata của hai tập văn bản.
2. Đọc các file `_tokenized.jsonl`.
3. Tạo `chunk_id` deterministic.
4. Upsert text/metadata vào Elasticsearch.
5. Upsert embedding vào Qdrant.

Nếu có sẵn `embeddings_output_ids.jsonl` và thư mục `embeddings/`, script sẽ dùng embedding đã tính sẵn khi tìm thấy. Nếu không có, script tự tính embedding bằng `model_cache/vietlegal-harrier-0.6b`.

## Chạy LLM Trên Máy Cục Bộ

Phục vụ Qwen3.5-9B bằng vLLM:

```bash
bash start_eval.sh
```

Script mặc định dùng GPU `1`, port `8000`, tên model `qwen3.5-9b`. Có thể đổi GPU bằng biến môi trường:

```bash
GPU_ID=0 bash start_eval.sh
```

Kiểm tra endpoint:

```bash
curl http://localhost:8000/v1/models
```

## Chạy Suy Luận/Evaluation

Chạy thử nhanh với một số câu:

```bash
CHECKPOINTER_BACKEND=none python evaluation.py \
  --input R2AIStage1DATA.json \
  --out-dir submission_eval_smoke \
  --limit 5 \
  --workers 1 \
  --graph-mode stateless
```

Chạy toàn bộ test set:

```bash
CHECKPOINTER_BACKEND=none python evaluation.py \
  --input R2AIStage1DATA.json \
  --out-dir submission_eval \
  --workers 1 \
  --graph-mode stateless
```

Đầu ra:

```text
submission_eval/
  results.json
  submission.zip
  diagnostics.jsonl
  errors.jsonl
```

`submission.zip` được tạo phẳng, bên trong chỉ có `results.json`.

Nếu bị dừng giữa chừng, chạy lại cùng `--out-dir`; script sẽ đọc `results.json` cũ và bỏ qua các câu đã có câu trả lời hợp lệ. Dùng `--force` nếu muốn chạy lại từ đầu.

## Lệnh Thường Dùng

Chạy một khoảng ID:

```bash
CHECKPOINTER_BACKEND=none python evaluation.py \
  --start-id 1 \
  --end-id 100 \
  --out-dir submission_eval_1_100 \
  --workers 1 \
  --graph-mode stateless
```

Chạy một danh sách ID:

```bash
CHECKPOINTER_BACKEND=none python evaluation.py \
  --ids 1,2,10 \
  --out-dir submission_eval_ids \
  --workers 1 \
  --graph-mode stateless
```

Chạy với NVIDIA endpoint chỉ để chạy thử/gỡ lỗi:

```bash
NVIDIA_API_KEY=... CHECKPOINTER_BACKEND=none python evaluation.py \
  --llm-backend nvidia \
  --limit 5 \
  --out-dir submission_eval_nvidia_smoke \
  --graph-mode stateless
```

Bản nộp cuối cần tuân thủ quy định model trong [AGENTS.md](AGENTS.md).

## Cấu Hình Chính

`Agents/config.py` đọc các biến môi trường sau:

| Biến | Mặc định | Mô tả |
| --- | --- | --- |
| `LLM_MODEL` | `qwen3.5-9b` | Tên model được phục vụ qua endpoint |
| `LLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `LLM_API_KEY` | `EMPTY` | API key cho endpoint local |
| `CHECKPOINTER_BACKEND` | `postgres` | Dùng `none` khi batch evaluation |
| `EMBEDDING_MAX_CONCURRENT` | `1` | Giới hạn embedding model |
| `RERANKER_MAX_CONCURRENT` | `1` | Giới hạn reranker |
| `VNCORENLP_MAX_CONCURRENT` | `8` | Giới hạn VnCoreNLP |

## Sự Cố Thường Gặp

- `Connection refused` tới `localhost:8000`: chưa chạy `start_eval.sh` hoặc vLLM chưa sẵn sàng.
- `Connection refused` tới Elasticsearch/Qdrant: chưa chạy `bash dbs/start_dbs.sh`.
- Lỗi Java/VnCoreNLP: kiểm tra `java -version` và thư mục `model_cache/vncorenlp`.
- Lỗi Postgres checkpointer: dùng `CHECKPOINTER_BACKEND=none` cho evaluation, hoặc cài thêm `langgraph-checkpoint-postgres` và `psycopg[binary,pool]`.
- Hết VRAM khi serve Qwen: giảm `--max-model-len`, `--gpu-memory-utilization`, hoặc đổi GPU bằng `GPU_ID`.

## Tài Liệu Bổ Sung

- [AGENTS.md](AGENTS.md): mục tiêu cuộc thi, cấu trúc submission và quy định model.
- [docs/data_description.md](docs/data_description.md): mô tả chi tiết 4 file JSONL.
- [docs/model_description.md](docs/model_description.md): model card và cách tải checkpoint.
- [docs/submission_document_requirements.md](docs/submission_document_requirements.md): checklist bộ tài liệu nộp nghiệm thu.
