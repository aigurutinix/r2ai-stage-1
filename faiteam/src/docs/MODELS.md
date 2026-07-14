# Mô tả mô hình — R2AI Stage 1

Tài liệu mô tả các mô hình AI sử dụng trong pipeline R2AI Stage 1: thông tin model, phiên bản checkpoint, hướng dẫn tải và sử dụng.

---

## 1. Tổng quan kiến trúc model

Pipeline R2AI Stage 1 sử dụng **ba model neural** và **một chỉ mục sparse**:

```
Câu hỏi
   │
   ├─► [Embedding] Vietnamese_Embedding_v2  ──► Dense search (Qdrant ANN)
   │
   ├─► [BM25] Qdrant sparse vector "bm25"     ──► Sparse search
   │
   └─► RRF fusion → [Reranker] Vietnamese_Reranker → Top-K chunks
                              │
                              ├─► extract_citations (benchmark mode)
                              └─► [LLM] Vi-Qwen2-1.5B-RAG (full pipeline)
```

| Vai trò | Model | Base model | Thư viện |
|---------|-------|------------|----------|
| Dense embedding | `Vietnamese_Embedding_v2` | BAAI/bge-m3 | sentence-transformers |
| Cross-encoder rerank | `Vietnamese_Reranker` | BAAI/bge-reranker-v2-m3 | sentence-transformers |
| Text generation (RAG) | `Vi-Qwen2-1.5B-RAG` | Qwen2-1.5B-Instruct | transformers |
| Sparse retrieval | BM25 (Qdrant pre-indexed) | Qdrant/bm25 | qdrant-client |

---

## 2. Link tải checkpoint

> **Thay link Google Drive bằng link thực tế trước khi nộp sản phẩm.**

| Model | HuggingFace (chính thức) | Google Drive (bản đóng gói) |
|-------|--------------------------|------------------------------|
| Vietnamese_Embedding_v2 | [AITeamVN/Vietnamese_Embedding_v2](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2) | [Google Drive](https://drive.google.com/drive/folders/YOUR_EMBED_FOLDER_ID) |
| Vietnamese_Reranker | [AITeamVN/Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker) | [Google Drive](https://drive.google.com/drive/folders/YOUR_RERANK_FOLDER_ID) |
| Vi-Qwen2-1.5B-RAG | [AITeamVN/Vi-Qwen2-1.5B-RAG](https://huggingface.co/AITeamVN/Vi-Qwen2-1.5B-RAG) | [Google Drive](https://drive.google.com/drive/folders/YOUR_LLM_FOLDER_ID) |

**Đường dẫn local mặc định** (sau khi tải):

```
models/Vietnamese_Embedding_v2/
models/Vietnamese_Reranker/
models/Vi-Qwen2-1.5B-RAG/
```

---

## 3. Vietnamese_Embedding_v2

### 3.1. Thông tin

| Thuộc tính | Giá trị |
|------------|---------|
| **Loại** | Sentence Transformer (bi-encoder) |
| **Base model** | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| **Kiến trúc** | XLM-RoBERTa |
| **Output dimension** | 1024 |
| **Max sequence length** | 2048 tokens (runtime); train: 256 query + 2048 passage |
| **Similarity** | Dot product |
| **Ngôn ngữ** | Tiếng Việt |
| **License** | Apache 2.0 |
| **Huấn luyện** | ~1.100.000 triplets (query, positive, negative) |

### 3.2. Checkpoint

- **HuggingFace revision:** `main` (snapshot tại thời điểm tải)
- **Files chính:** `config.json`, `model.safetensors` (hoặc `pytorch_model.bin`), `tokenizer.json`, `sentence_bert_config.json`
- **Local path:** `models/Vietnamese_Embedding_v2/`

### 3.3. Tải checkpoint

```bash
# Cách 1: HuggingFace CLI
pip install huggingface_hub
huggingface-cli download AITeamVN/Vietnamese_Embedding_v2 \
  --local-dir models/Vietnamese_Embedding_v2

# Cách 2: Tải từ Google Drive, giải nén vào models/Vietnamese_Embedding_v2/
```

### 3.4. Sử dụng

**Standalone:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/Vietnamese_Embedding_v2")
model.max_seq_length = 2048
query_vec = model.encode(["Câu hỏi pháp luật?"])
doc_vec = model.encode(["Đoạn văn bản pháp luật..."])
```

**Trong pipeline:**

```bash
python scripts/rag_answer_stage1.py \
  --embed-model models/Vietnamese_Embedding_v2 \
  --device-embed cuda
```

**Fallback (không có local model):** set `EMBED_MODEL_PATH=BAAI/bge-m3` trong `.env`.

---

## 4. Vietnamese_Reranker

### 4.1. Thông tin

| Thuộc tính | Giá trị |
|------------|---------|
| **Loại** | Cross-encoder (sequence classification) |
| **Base model** | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| **Kiến trúc** | XLM-RoBERTa |
| **Max sequence length** | 2304 tokens (256 query + 2048 passage) |
| **Output** | Relevance score (logit) |
| **Ngôn ngữ** | Tiếng Việt |
| **License** | Apache 2.0 |
| **Huấn luyện** | ~1.100.000 triplets |

### 4.2. Checkpoint

- **HuggingFace:** [AITeamVN/Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker)
- **Local path:** `models/Vietnamese_Reranker/`

### 4.3. Tải checkpoint

```bash
huggingface-cli download AITeamVN/Vietnamese_Reranker \
  --local-dir models/Vietnamese_Reranker
```

### 4.4. Sử dụng trong pipeline

Reranker được load qua `scripts/rerank_retrieval.py`:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("models/Vietnamese_Reranker", max_length=2304, device="cuda")
```

**Hybrid rerank score** (khi có sub-queries):

```
final_score = 0.7 × primary_score + 0.3 × mean(sub_scores)
```

**CLI:**

```bash
python scripts/rag_answer_stage1.py \
  --rerank-model models/Vietnamese_Reranker \
  --device-rerank cuda \
  --top-k 8
```

Tắt reranker: `--no-rerank`

**Fallback:** `RERANK_MODEL_PATH=BAAI/bge-reranker-v2-m3`

---

## 5. Vi-Qwen2-1.5B-RAG

### 5.1. Thông tin

| Thuộc tính | Giá trị |
|------------|---------|
| **Loại** | Causal Language Model (RAG-specialized) |
| **Base model** | Qwen2-1.5B-Instruct |
| **Tham số** | 1.5B |
| **Context window** | 8192 tokens (model card); pipeline dùng ~3500 chars context |
| **Ngôn ngữ** | Tiếng Việt |
| **License** | Apache 2.0 |
| **Mục đích** | Sinh câu trả lời RAG từ ngữ cảnh pháp luật |

**Khả năng RAG (theo model card):**

- Trích xuất thông tin từ tài liệu nhiễu (1 positive + n negative)
- Từ chối trả lời khi không có thông tin trong context
- Tích hợp thông tin từ nhiều tài liệu
- Phân loại positive/negative context (~99% accuracy trên EvalRAGData)

### 5.2. Checkpoint

- **HuggingFace:** [AITeamVN/Vi-Qwen2-1.5B-RAG](https://huggingface.co/AITeamVN/Vi-Qwen2-1.5B-RAG)
- **Local path:** `models/Vi-Qwen2-1.5B-RAG/`
- **Files chính:** `config.json`, `model.safetensors`, `tokenizer.json`, `generation_config.json`

### 5.3. Tải checkpoint

```bash
huggingface-cli download AITeamVN/Vi-Qwen2-1.5B-RAG \
  --local-dir models/Vi-Qwen2-1.5B-RAG
```

### 5.4. Sử dụng trong pipeline

LLM **chỉ được load** khi chạy full pipeline (không dùng `--citations-only`).

**Quantization mặc định:** 4-bit trên CUDA (via `bitsandbytes`)

```bash
# Full pipeline
python scripts/rag_answer_stage1.py \
  --llm-model models/Vi-Qwen2-1.5B-RAG \
  --device-llm cuda \
  --gen-batch 2 \
  --max-new-tokens 512

# Tắt 4-bit (cần VRAM lớn hơn)
python scripts/rag_answer_stage1.py --no-4bit
```

**Tham số generation:**

| Tham số | Giá trị |
|---------|---------|
| `temperature` | 0.1 |
| `max_new_tokens` | 512 (mặc định, tuỳ `--max-new-tokens`) |
| `max_context_chars` | 3500 (mặc định) |

**Prompt template** (trong `scripts/rag_answer_stage1.py`):

- System: trợ lý tiếng Việt trung thực
- User: ngữ cảnh pháp luật + câu hỏi + yêu cầu trả lời một đoạn văn liền mạch

---

## 6. BM25 Sparse Retrieval

Không phải neural checkpoint — là chỉ mục sparse trên Qdrant Cloud.

| Thuộc tính | Giá trị |
|------------|---------|
| **Vector name** | `bm25` |
| **Backend** | Qdrant sparse vectors (pre-indexed) |
| **Local fallback** | `rank-bm25` library, cache tại `output/bm25_corpus.pkl` |

Cấu hình trong `.env`:

```env
QDRANT_SPARSE_VECTOR_NAME=bm25
```

Tắt BM25: `--no-bm25` (chỉ dense search).

---

## 7. Thứ tự ưu tiên cấu hình model

Resolver trong `scripts/qdrant_config.py`:

1. **CLI flag:** `--embed-model`, `--rerank-model`, `--llm-model`
2. **Biến môi trường:** `EMBED_MODEL_PATH`, `RERANK_MODEL_PATH`
3. **Local directory:** `models/Vietnamese_Embedding_v2`, `models/Vietnamese_Reranker`
4. **HuggingFace fallback:** `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`

LLM mặc định: `models/Vi-Qwen2-1.5B-RAG` (hardcoded, override bằng `--llm-model`).

---

## 8. Yêu cầu phần cứng theo model

| Model | VRAM ước tính | Device mặc định |
|-------|---------------|-----------------|
| Vietnamese_Embedding_v2 | ~2 GB | `--device-embed cuda` |
| Vietnamese_Reranker | ~2 GB | `--device-rerank cuda` |
| Vi-Qwen2-1.5B-RAG (4-bit) | ~2–3 GB | `--device-llm cuda` |
| Vi-Qwen2-1.5B-RAG (bf16) | ~4 GB | `--no-4bit` |
| **Tổng (full pipeline)** | **~6–8 GB** | CUDA khuyến nghị |

Chế độ `--citations-only`: không load LLM, tiết kiệm ~2–4 GB VRAM.

---

## 9. Hiệu năng tham khảo (Legal Zalo 2021)

Đánh giá trên tập Legal Zalo 2021 (model **không** train trên tập này):

| Model | Acc@1 | Acc@3 | Acc@5 | Acc@10 | MRR@10 |
|-------|-------|-------|-------|--------|--------|
| Vietnamese_Reranker | 0.7944 | 0.9324 | 0.9537 | 0.9740 | 0.8672 |
| Vietnamese_Embedding_v2 | 0.7262 | 0.8927 | 0.9268 | 0.9578 | 0.8149 |
| BGE-M3 (baseline) | 0.5682 | 0.7728 | 0.8382 | 0.8921 | 0.6822 |

---

## 10. Tham số pipeline liên quan model

| Tham số | Mặc định | Ảnh hưởng |
|---------|----------|-----------|
| `--top-k` | 8 | Số chunk sau rerank |
| `--llm-top-k` | 2 | Chunk đưa vào LLM / citations |
| `--retrieve-pool` | 30 | Pool trước RRF |
| `--rrf-top-k` | 50 | Pool sau RRF |
| `--rerank-batch` | 8 | Batch size reranker |
| `--retrieve-batch` | 4 | Batch size embedding query |
| `--gen-batch` | 2 | Batch size LLM generation |
| `--max-context-chars` | 3500 | Giới hạn context LLM |
| `--max-new-tokens` | 512 | Max tokens sinh |

**Trọng số RRF** (`scripts/bm25_retrieval.py`): dense 0.4, BM25 0.6, k=60.

**Trọng số hybrid rerank** (`scripts/rerank_retrieval.py`): primary 0.7, sub-query mean 0.3.

---

## 11. Liên hệ phát triển model

Model embedding và reranker phát triển bởi AITeamVN:

- HuggingFace: [AITeamVN](https://huggingface.co/AITeamVN)
- Email (model card): nguyennhotrung3004@gmail.com
