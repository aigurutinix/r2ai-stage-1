# Hướng dẫn tái lập (Reproduce Guide)

Hướng dẫn tái lập bài nộp cuối `submission_3_1_combo_ck23k_f01_a07.json`.
Toàn bộ pipeline dùng model **< 14B**.

Tất cả lệnh chạy từ thư mục gốc `r2ai_submission/`. Dữ liệu lớn tải từ [NghiemAbe/r2ai-legal-data](https://huggingface.co/datasets/NghiemAbe/r2ai-legal-data) và đặt vào `data/`.

---

## 0. Chuẩn bị môi trường

```bash
# Python 3.10+ (dev dùng 3.13)
pip install -r r2ai_submission/code/requirements.txt

# GPU: NVIDIA H20 (97GB) hoặc tương đương, đủ cho:
#   - vLLM serve Qwen3-8B-AWQ
#   - vLLM serve reranker (pooling)
#   - encode embeddings (Vietnamese_Embedding_v2)
```

Tải model (xem `docs/model_description.md`):
- `AITeamVN/Vietnamese_Embedding_v2`
- `AITeamVN/Vietnamese_Reranker` (base, để fine-tune)
- Reranker fine-tuned: dùng checkpoint có sẵn `models/reranker_finetuned_v2_ck8000/`
- `Qwen3-8B-AWQ` (repo: `Qwen/Qwen3-8B-AWQ`)
- LoRA classifier base: `models/qwen3-1.7b-base/` (đã đi kèm submission)
- LoRA adapter: `models/legal_classifier_lora_v2_ck23000/` (đã đi kèm submission)

---

## 1. Build index

### 1a. Dense index (Vietnamese_Embedding_v2 + FAISS)

```bash
python3 code/retrieval_dense.py build-index
# → retrieval_index_dense/ : faiss.index, embeddings.npy, metas.pkl (~612K vectors, dim=1024)
```

### 1b. BM25S v7 index (vbpl + data_final QA + 1.27M synthetic QA, vi-tokenize)

```bash
python3 code/retrieval_bm25s.py build-index
# → retrieval_index_bm25s_v7/ : bm25s_model/, metas.pkl (~1.88M chunks)
```

> Cả hai index đã được build sẵn (symlink trong `r2ai_submission/data/`). Chỉ rebuild nếu thay
> đổi corpus. Synthetic QA có thể sinh lại bằng `code/gen_synthetic_questions.py`.

---

## 2. Phục vụ LLM Qwen3-8B-AWQ (vLLM, port 8011)

```bash
# OpenAI-compatible API
vllm serve Qwen/Qwen3-8B-AWQ \
    --served-model-name Qwen3-8B-AWQ \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.3 \
    --max-num-seqs 64 \
    --api-key token-abc123 \
    --port 8011
```

---

## 3. Query Decomposition + HyDE (sinh candidates)

```bash
# Tách câu hỏi phức → sub-queries (cache: query_decompose_cache.json)
python3 code/query_decomposition.py decompose --workers 8

# Dense retrieval với sub-queries → candidates
python3 code/query_decomposition.py retrieve --source dense --top-k 100

# HyDE: LLM sinh hypothetical article → dense search → hyde_candidates.pkl
python3 code/hyde_retrieval.py generate --port 8011
python3 code/hyde_retrieval.py retrieve --top-k 100
```

Cũng cần BM25S decompose top-150 (để tính intersection):

```bash
python3 code/query_decomposition.py retrieve --source bm25 --top-k 150
# → submission_3_1_decompose_bm25_top150.json
```

---

## 4. Phục vụ Reranker fine-tuned (vLLM pooling, port 8012) + tạo intersection scores

```bash
vllm serve models/reranker_finetuned_v2_ck8000 \
    --runner pooling \
    --port 8012 \
    --api-key token-abc123
```

Tạo intersection pool (HyDE ∩ BM25S) + chấm điểm reranker:

```bash
python3 code/rerank_intersection.py --port 8012
# → rerank_intersection_scores.pkl (sorted desc theo reranker_score)
#   Avg ~29 candidates/query, recall ceiling ~90%
```

---

## 5. Reranker top-5 → Qwen3-8B-AWQ thinking → submission base

```bash
python3 code/run_llm_rerank_on_reranker_top5.py \
    --port 8011 \
    --model Qwen3-8B-AWQ \
    --tag 8b \
    --workers 4
# → submission_3_1_llm8b_rerankerv2ck8k_top5.json   (Articles F2 = 0.6056)
```

Bước này:
1. Đọc `rerank_intersection_scores.pkl`, lấy **reranker top-5** mỗi query làm candidates.
2. Đưa 5 candidates (full content ≤10K chars) vào Qwen3-8B-AWQ với `enable_thinking=True`,
   `max_tokens=8000`, `temperature=0`.
3. LLM chọn điều luật TRỰC TIẾP liên quan → format submission (avg ~2.7 articles, 0 empty).

---

## 5b. LoRA Classifier post-processing → submission final

```bash
# (1) Chạy classifier inference trên reranker top-5 (batched logits, ~15 phút)
python3 code/run_classifier_unsloth.py \
    --checkpoint models/legal_classifier_lora_v2_ck23000 \
    --top-k 5 --batch-size 16
# → classifier_v2_probs_top5_ck23000.pkl (P(Có) cho mỗi (query, article) pair)

# (2) Combo filter + add
python3 code/ensemble_classifier_reranker.py
#   - Filter: bỏ articles từ LLM 8B có P(Có) < 0.1 (loại false positives)
#   - Add: thêm articles từ reranker top-5 có P(Có) ≥ 0.7 (bổ sung recall)
# → submission_3_1_combo_ck23k_f01_a07.json
```

**Cách hoạt động classifier**:
- Base model: `Qwen/Qwen3-1.7B-Base` (đi kèm tại `models/qwen3-1.7b-base/`)
- LoRA adapter: checkpoint-23000 (đi kèm tại `models/legal_classifier_lora_v2_ck23000/`)
- Inference: forward pass lấy logits, so sánh `logits[yes_id]` vs `logits[no_id]` (KHÔNG dùng generate)
- **Lưu ý**: KHÔNG thêm trailing space vào prompt (sẽ lệch 1 position so với training eval)
- **Lưu ý**: QLoRA adapter KHÔNG merge được — phải dùng unsloth inference (4-bit base + LoRA riêng)

**Tại sao filter + add hoạt động**:
- Classifier (logits comparison) và LLM 8B (reasoning) rất khác nhau: Jaccard overlap chỉ ~0.40
- Classifier tìm được 1657 articles mà LLM 8B không chọn, và ngược lại
- → Classifier loại đúng FP của LLM (precision +16%), thêm articles LLM bỏ sót (recall +2%)
- Kết quả: F2 tăng 0.6056 → **0.632** (+4.4%)

---

## 5c. Sinh trường `answer`

Sinh văn bản trả lời cho 5 tiêu chí chấm điểm Task 3.2 (căn cứ chính xác, chính xác nội dung,
đầy đủ, thực tiễn, rõ ràng).

```bash
# (1) Build few-shot index trên data_final questions (chỉ chạy 1 lần)
python3 code/build_fewshot_index.py
# → fewshot_index/ : questions.faiss, meta.pkl

# (2) Search + rerank → top-5 few-shot mỗi query
python3 code/run_fewshot_rerank.py --port 8011 --workers 8
# → fewshot_top5_reranked.pkl

# (3) Sinh answer: optimized prompt + few-shot → Qwen3-8B-AWQ (enable_thinking=True)
python3 code/gen_answer.py \
    --port 8011 --model Qwen3-8B-AWQ \
    --submission submission_3_1_combo_ck23k_f01_a07.json \
    --workers 4
# → submission_3_1_combo_ck23k_f01_a07_with_answer.json

# (4) Fill bù answers rỗng (articles không tìm thấy trong reranker pool)
python3 code/gen_answer_fill_empty.py \
    --port 8011 --model Qwen3-8B-AWQ \
    --file submission_3_1_combo_ck23k_f01_a07_with_answer.json \
    --workers 4

# (5) Strip label "Tóm tắt:" khỏi đầu answer
python3 code/strip_answer_labels.py \
    --file submission_3_1_combo_ck23k_f01_a07_with_answer.json
```

> **Prompt answer** (`gen_answer.py`): tách system message (anti-hallucination: "TUYỆT ĐỐI KHÔNG
> bổ sung điều luật ngoài context") + user message. Cấu trúc: phân tích chi tiết → lưu ý thực tiễn
> → kết thúc "Căn cứ pháp lý:". `max_tokens=8192` (thinking tokens tính vào).
> Flag `--reflection` tùy chọn: thêm bước tự kiểm tra trước khi trả lời (A/B test cho thấy
> không cải thiện đáng kể với Qwen3-8B).

---

## 6. Kiểm tra định dạng submission

```bash
python3 code/verify_submission.py submission_3_1_llm8b_rerankerv2ck8k_top5.json
# kiểm tra schema: id, question, answer, relevant_docs, relevant_articles
```

---

## 7. (Tùy chọn) Fine-tune lại Reranker → checkpoint-8000

Nếu cần tái tạo `reranker_finetuned_v2_ck8000/`:

```bash
# 1. Sinh train data: hard negatives từ dense retrieval trên 10K câu hỏi data_final
python3 code/mine_hard_negatives_dense.py
# → data_final_hard_negatives.jsonl  (96,456 dòng, schema {query, passage, label})

# 2. Fine-tune (CrossEncoderTrainer, BinaryCrossEntropyLoss, GroupShuffleSampler 8-way)
python3 code/train_reranker_v2.py        # hoặc bash code/run_finetune_v2.sh
# → reranker_finetuned_v2/checkpoint-8000  (sweet spot ~epoch 1.87, NDCG@10 ≈ 0.896)
```

- **Train data DUY NHẤT**: `data_final_hard_negatives.jsonl`. Eval = **500 query held-out**
  tách từ chính file này. Chi tiết: `docs/model_description.md` mục 3.
- Nếu chỉ tái lập inference, dùng checkpoint backup có sẵn — không cần train lại.

---

## Sơ đồ phụ thuộc file (data flow)

```
R2AIStage1DATA.json
  ├─(decompose)→ query_decompose_cache.json
  ├─(dense retrieve)→ [dense candidates]
  ├─(hyde)→ hyde_candidates.pkl ──────────────────────────────┐
  └─(bm25 decompose)→ submission_3_1_decompose_bm25_top150.json ┐
                                                                 ▼
                              rerank_intersection.py → rerank_intersection_scores.pkl
                                                                 ▼
                  run_llm_rerank_on_reranker_top5.py (reranker top-5 → Qwen3-8B-AWQ thinking)
                                                                 ▼
                            submission_3_1_llm8b_rerankerv2ck8k_top5.json  (F2=0.6056)
                                                                 │
                                          ┌──────────────────────┘
                                          ▼
                  run_classifier_unsloth.py (LoRA classifier, Qwen3-1.7B-Base+LoRA ck-23000)
                    → classifier_v2_probs_top5_ck23000.pkl (P(Có) per pair)
                                          ▼
                  ensemble_classifier_reranker.py (filter P<0.1 + add P≥0.7)
                                          ▼
                  submission_3_1_combo_ck23k_f01_a07.json
                                          │
                          (sinh answer đầy đủ)
   data_final ─► build_fewshot_index.py ─► fewshot_index/
                          │                                      │
                          ▼                                      ▼
        run_fewshot_rerank.py ─► fewshot_top5_reranked.pkl ─► gen_answer.py (optimized prompt)
                                                                 ▼
                                                gen_answer_fill_empty.py (fill bù)
                                                                 ▼
                                                strip_answer_labels.py (bỏ "Tóm tắt:")
                                                                 ▼
                       submission_3_1_combo_ck23k_f01_a07.json  ← BẢN NỘP CUỐI

[train reranker — offline]
   data_final ─► mine_hard_negatives_dense.py ─► data_final_hard_negatives.jsonl
                                                       ▼
                                          train_reranker_v2.py ─► checkpoint-8000

[train classifier — offline]
   data_final_hard_negatives.jsonl ─► train_legal_classifier_v2.py (Qwen3-1.7B-Base + QLoRA)
                                          ▼
                                  legal_classifier_lora_v2/checkpoint-23000
```

---

## Lưu ý quan trọng khi tái lập

- **Cache files**: nhiều bước dùng cache (`.pkl`, `.json`). Nếu thay đổi model/prompt, **xóa cache**
  tương ứng trước khi chạy lại (vd `rm -f llm_rerank_cache_*.pkl`).
- **API key**: các script hardcode `token-abc123` cho vLLM. Serve với cùng key.
- **Port**: 8011 (LLM Qwen3-8B-AWQ), 8012 (Reranker pooling). Khớp với tham số `--port`.
- **Thinking mode**: bắt buộc `enable_thinking=True` ở bước chọn điều luật (bước 5).
- **Classifier inference**: dùng unsloth (4-bit base + LoRA riêng), KHÔNG merge model. Forward pass lấy logits, KHÔNG dùng generate. KHÔNG thêm trailing space vào prompt.
- **Port**: 8011 (LLM Qwen3-8B-AWQ), 8012 (Reranker pooling). Classifier chạy local (không cần serve).

---

## 8. (Tùy chọn) Train lại LoRA Classifier

Nếu cần tái tạo `legal_classifier_lora_v2/checkpoint-23000`:

```bash
# Train data: data_final_hard_negatives.jsonl (96K pairs, có sẵn)
# Base model: Qwen/Qwen3-1.7B-Base (models/qwen3-1.7b-base/)
python3 code/train_legal_classifier_v2.py
# → legal_classifier_lora_v2/checkpoint-23000
# QLoRA 4-bit, LoRA r=64, alpha=16, 40 epochs
# Train data = reranker top-5 candidates, labels từ union 27B submission
```

- Nếu chỉ tái lập inference, dùng LoRA adapter có sẵn — không cần train lại.
