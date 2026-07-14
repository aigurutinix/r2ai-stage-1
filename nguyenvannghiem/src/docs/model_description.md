# Mô tả mô hình (Model Description)

Tài liệu mô tả tất cả mô hình dùng trong pipeline truy hồi điều luật R2AI Task 3.1.

> **Ràng buộc BTC**: model inference phải **< 14B parameters**. Toàn bộ pipeline bài nộp dùng:
> Vietnamese_Embedding_v2 (568M), Vietnamese_Reranker fine-tuned (568M), Qwen3-8B-AWQ (8B) —
> tất cả đều hợp lệ.

---

## 1. `AITeamVN/Vietnamese_Embedding_v2` — Dense Embedding

| Thuộc tính | Giá trị |
|------------|---------|
| Vai trò | Bi-encoder embedding cho dense retrieval |
| Kích thước | ~568M params |
| Dim | 1024 |
| Max seq | 512 (dense index chính). Biến thể 4096 đã thử nhưng thua cho LLM rerank |
| HuggingFace | `AITeamVN/Vietnamese_Embedding_v2` |
| Download | Tải từ HuggingFace Hub (tự động khi load model) |

- **Sử dụng**: encode toàn bộ ~612K chunk → FAISS index (`retrieval_index_dense/`).
  Encode query (gốc + sub-queries + HyDE doc) → search top-100.
- **L2-normalize** embedding → cosine similarity qua inner product (`faiss.IndexFlatIP`).
- **Tại sao chọn**: đã so sánh với `intfloat/multilingual-e5-base` → Vietnamese_Embedding_v2
  vượt trội (+5% recall), do được train cho Vietnamese legal domain.
- **Lưu ý**: best pipeline dùng model **GỐC** (không fine-tune). Đã thử fine-tune bi-encoder
  (CachedMNRL, 133K triplets) nhưng eval metrics giảm (catastrophic forgetting) → không dùng.

---

## 2. `AITeamVN/Vietnamese_Reranker` — Base Reranker

| Thuộc tính | Giá trị |
|------------|---------|
| Vai trò | Cross-encoder reranker (base, trước khi fine-tune) |
| Kiến trúc | XLM-RoBERTa-large (bge-reranker-v2-m3 fine-tuned) |
| Kích thước | ~568M params |
| Train data gốc | 1.1M Vietnamese triplets |
| HuggingFace | `AITeamVN/Vietnamese_Reranker` |
| Download | Tải từ HuggingFace Hub (tự động khi load model) |

- So sánh trên intersection pool: Vietnamese_Reranker **tốt nhất** so với jina-reranker và
  dense score (median rank của GT giảm từ 4 → 2 sau khi rerank).
- Là **base model** để fine-tune ra checkpoint v2 (mục 3).

---

## 3. `reranker_finetuned_v2_ck8000` — Reranker fine-tuned (BEST, dùng trong pipeline)

| Thuộc tính | Giá trị |
|------------|---------|
| Vai trò | **Reranker chính trong pipeline** (top-5 filter trước LLM) |
| Base | `AITeamVN/Vietnamese_Reranker` |
| Kiến trúc | `XLMRobertaForSequenceClassification`, hidden 1024, 24 layers, 16 heads, max_position 8194, vocab 250002 |
| Activation | Sigmoid (sentence-transformers 5.1.0) |
| Checkpoint | **checkpoint-8000** — sweet spot (~epoch 1.87) |
| Eval NDCG@10 | ≈ 0.896 (so với base 0.8056) |
| Vị trí | `models/reranker_finetuned_v2_ck8000/` |

### File trong checkpoint (đóng gói cho inference)

Trong `models/reranker_finetuned_v2_ck8000/` chỉ giữ các file cần cho inference:

```
config.json, model.safetensors (2.27 GB), sentencepiece.bpe.model,
special_tokens_map.json, tokenizer.json, tokenizer_config.json, README.md
```

> Các file train-only (`optimizer.pt` 4.5 GB, `training_args.bin`, `scheduler.pt`,
> `rng_state.pth`, `trainer_state.json`) đã được **loại bỏ** khỏi bản đóng gói — chỉ cần
> nếu muốn train tiếp. Bản inference ~2.3 GB.

### Cách train (xem `code/train_reranker_v2.py`)

- **Train data — DUY NHẤT**: `data_final_hard_negatives.jsonl` (96,456 dòng, schema
  `{query, passage, label}`). File này do `code/mine_hard_negatives_dense.py` sinh ra:
  - Lấy mẫu **10K câu hỏi** từ `data_final` theo phân bố chủ đề khớp tập test.
  - Mỗi câu → encode bằng Vietnamese_Embedding_v2 → dense search **top-20** trong corpus.
  - **Positive** (label=1) = các điều luật được cite (`article_cite`); **hard negative**
    (label=0) = các candidate **rank 5-12** sau dedup (gần đúng về mặt ngữ nghĩa nhưng sai →
    "hard"). Mỗi positive ghép ~7 hard negatives.
  - Kết quả: 9,150 query có cả pos + neg, tổng 12,057 positive + 84,399 negative.
- **Eval**: tách **500 query held-out** từ chính file hard negatives → `CrossEncoderRerankingEvaluator`
  (NDCG@10 / MRR@10 / MAP). Train trên phần còn lại.
- **Loss**: `BinaryCrossEntropyLoss` với `pos_weight` cân bằng tỉ lệ 1:7.
- **Sampler**: `GroupShuffleSampler` (nhóm 8: 1 positive + 7 hard negatives).
- **Hyperparams**: `MAX_LENGTH=2048`, `lr=2e-5`, `weight_decay=0.01`, `warmup_ratio=0.1`,
  ~2 epochs (best ở checkpoint-8000), `EarlyStoppingCallback(patience=5)`.

### So sánh checkpoint

| Checkpoint | Pool top-5 recall | F2 standalone | Ghi chú |
|------------|-------------------|---------------|---------|
| ck-6000 | 79.5% | 0.5177 | |
| **ck-8000** | 79.3% | **0.5347** | **best — dùng trong pipeline** |
| ck-18000 | 87.1% | 0.5187 | overfit |

> Cũng tồn tại `reranker_finetuned_v2_backup_ck6000/` (backup). Best là **ck8000**.

### Phục vụ inference

```bash
# vLLM pooling runner (nhanh hơn nhiều so với sentence-transformers thuần)
vllm serve models/reranker_finetuned_v2_ck8000 --runner pooling --port 8012 --api-key token-abc123
# Endpoint: POST http://localhost:8012/v1/score  {model, text_1, text_2: [...]}
```

---

## 4. `Qwen3-8B-AWQ` — LLM (decompose / HyDE / rerank cuối / sinh answer)

| Thuộc tính | Giá trị |
|------------|---------|
| Vai trò | LLM cho Query Decomposition, HyDE, chọn điều luật cuối, và sinh `answer` |
| Kích thước | 8B (lượng tử hóa AWQ) — **hợp lệ < 14B** |
| Context | 32K tokens (`--max-model-len 32768`) |
| Thinking mode | `enable_thinking` bật/tắt theo bước (xem dưới) |
| Temperature | 0 |
| Phục vụ | vLLM, OpenAI-compatible API, port **8011** |
| Download | `Qwen/Qwen3-8B-AWQ` từ HuggingFace Hub |

Gọi qua HTTP `POST http://localhost:8011/v1/chat/completions`. Vai trò theo từng bước:

| Bước | Script | enable_thinking | max_tokens | content/điều | Ghi chú |
|------|--------|-----------------|-----------|--------------|---------|
| Query decomposition | `query_decomposition.py` | True | 8000 | — | Tách câu hỏi phức → 1-3 sub-queries |
| HyDE | `hyde_retrieval.py` | — | 500 | — | Sinh hypothetical article → dense search |
| Chọn điều luật (cuối) | `run_llm_rerank_on_reranker_top5.py` | **True** | 8000 | ≤ **10,000** chars | **Quyết định Articles F2** |
| Sinh `answer` | `gen_answer.py` | **True** | 8192 | ≤ 8,000 chars | Optimized prompt + few-shot |

- **Thinking mode** ở bước chọn điều luật: bật chain-of-thought tăng precision đáng kể với
  dense candidates.
- **Prompt chọn điều luật**: đưa 5 candidate (reranker top-5, full content ≤10K chars/điều) →
  yêu cầu LLM chọn các điều luật **TRỰC TIẾP** quy định (loại match từ khóa lỏng lẻo). Trả về
  danh sách số thứ tự.
- **Prompt sinh `answer`** (`gen_answer.py`): system message tách riêng (enforce anti-hallucination:
  "TUYỆT ĐỐI KHÔNG bổ sung điều luật ngoài context") + user message chứa articles (content ≤8K chars)
  + metadata + 2 few-shot QA. `enable_thinking=True`, `max_tokens=8192`, `temperature=0`.
  Answer kết thúc bằng "Căn cứ pháp lý:" (99% answers). Post-processing: strip label "Tóm tắt:".

---

## 5. `Qwen/Qwen3-1.7B-Base` + LoRA — Classifier (post-processing filter + add)

| Thuộc tính | Giá trị |
|------------|---------|
| Vai trò | Binary classifier: (question, article) → Có/Không relevant |
| Base model | `Qwen/Qwen3-1.7B-Base` (1.7B params) |
| Fine-tune | QLoRA (4-bit, LoRA r=64, alpha=16) via Unsloth |
| Checkpoint | **checkpoint-23000** (train F1 = 1.0, overfit có chủ đích — xem giải thích bên dưới) |
| Train data | Reranker top-5 candidates (10K pairs), labels từ union 27B submission |
| Framework | Unsloth + TRL SFTTrainer |
| Vị trí | `models/qwen3-1.7b-base/` (base) + `models/legal_classifier_lora_v2_ck23000/` (LoRA adapter) |

### Cách hoạt động

Prompt-based classification: input = question + article content, output = so sánh logits token "Có" vs "Không" (không dùng generate, chỉ forward pass lấy logits).

```python
logit_yes = model_logits[last_pos, yes_token_id]
logit_no  = model_logits[last_pos, no_token_id]
P(Có) = softmax(logit_yes, logit_no)  # normalize giữa 2 tokens
```

### Tại sao overfit (ck-23000, train F1=1.0)?

Training data = chính xác reranker top-5 candidates cho 2000 câu test. Khi inference cũng chạy
trên cùng tập candidates, overfit = memorize đúng phân phối inference → kết quả tốt hơn
checkpoint sớm (ck-12000, F1=87%). So sánh Jaccard vs union 27B: ck-23000 = **0.725** vs
ck-12000 = 0.646.

### Sử dụng trong pipeline (post-processing)

Áp dụng SAU bước LLM 8B thinking chọn điều luật:
- **Filter**: bỏ articles mà classifier cho P(Có) < 0.1 → loại false positives, tăng precision
- **Add**: thêm articles từ reranker top-5 (mà LLM không chọn) có P(Có) ≥ 0.7 → bổ sung recall

### Lưu ý inference

- **KHÔNG merge được**: QLoRA adapter chỉ hoạt động đúng khi giữ riêng (4-bit base + LoRA). Merge vào base model bị hỏng do rounding errors.
- **Batched inference**: dùng forward pass + left padding, lấy logits tại vị trí cuối (KHÔNG thêm trailing space vào prompt — sẽ lệch 1 position).
- Script: `code/run_classifier_unsloth.py`

---

## 6. Bảng tổng hợp model & port

| Model | Vai trò | Params | Port (vLLM) | Hợp lệ <14B | Trong pipeline? |
|-------|---------|--------|-------------|--------------|------------------|
| Vietnamese_Embedding_v2 | Dense embed | 568M | (local, không serve) | ✅ | ✅ |
| Vietnamese_Reranker (base) | Reranker base | 568M | — | ✅ | (base để fine-tune) |
| reranker_finetuned_v2 ck8000 | Reranker fine-tuned | 568M | 8012 (pooling) | ✅ | ✅ |
| Qwen3-8B-AWQ | LLM decompose/HyDE/rerank/answer | 8B | 8011 | ✅ | ✅ |
| Qwen3-1.7B-Base + LoRA ck-23000 | Classifier filter/add | 1.7B | (unsloth, local) | ✅ | ✅ |

