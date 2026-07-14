# R2AI Stage 1 — Task 3.1: Truy hồi Điều luật (Legal Article Retrieval)

> Hệ thống truy hồi điều luật pháp luật Việt Nam cho cuộc thi R2AI Stage 1, Task 3.1.
> Toàn bộ pipeline dùng model **< 14B** (Qwen3-8B-AWQ + Vietnamese_Embedding_v2 +
> Vietnamese_Reranker fine-tuned + LoRA classifier Qwen3-1.7B-Base).

> **Bản nộp cuối (final)**: `submission_3_1_combo_ck23k_f01_a07.json`
> — LoRA classifier checkpoint-23000, combo filter f=0.1 + add a=0.7.
> Answer sinh bằng optimized prompt (anti-hallucination, system/user split, 99% có "Căn cứ pháp lý").

### Hồ sơ nộp (theo yêu cầu BTC)

| # | Hạng mục | Vị trí trong submission |
|---|----------|--------------------------|
| 1 | File kết quả dự đoán (tối đa 5) | `submissions/`, §1.2 định dạng |
| 2 | Tài liệu dữ liệu (nguồn, cấu trúc, link) | [`docs/data_description.md`](docs/data_description.md) |
| 3 | Checkpoint model (kiến trúc, cách train, link) | [`docs/model_description.md`](docs/model_description.md), `models/` |
| 4 | Mã nguồn (config, script, dependencies) | `code/`, `code/requirements.txt` |
| 5 | README hướng dẫn tái lập từng bước | file này + [`docs/reproduce.md`](docs/reproduce.md) |

---

## 1. Tổng quan hệ thống (System Overview)

### 1.1. Bài toán

- **Đầu vào**: 2000 câu hỏi pháp luật tiếng Việt (`R2AIStage1DATA.json`), mỗi câu có `id` + `question`.
- **Đầu ra**: File JSON gồm `answer`, `relevant_docs`, `relevant_articles` cho mỗi câu hỏi.
- **Chấm điểm**: benchmark trích "Điều X" từ trường `answer` và so sánh với đáp án.
  Metrics: **Precision**, **Recall**, **F2 macro-average** (recall quan trọng gấp 4 lần precision).

### 1.2. Định dạng đầu ra

```json
{
  "id": 1,
  "question": "Các cơ sở ươm tạo ... được hưởng những chính sách hỗ trợ nào về thuế và đất đai?",
  "answer": "Theo Điều 15 Luật 61/2020/QH14: ... Căn cứ Điều 15 Nghị định 10/2024/NĐ-CP ...",
  "relevant_docs": [
    "61/2020/QH14|Luật 61/2020/QH14",
    "10/2024/NĐ-CP|Nghị định 10/2024/NĐ-CP Quy định về khu công nghệ cao"
  ],
  "relevant_articles": [
    "61/2020/QH14|Luật 61/2020/QH14|Điều 15",
    "10/2024/NĐ-CP|Nghị định 10/2024/NĐ-CP Quy định về khu công nghệ cao|Điều 15"
  ]
}
```

- `relevant_docs`: `"{law_id}|{tên văn bản}"`
- `relevant_articles`: `"{law_id}|{tên văn bản}|Điều {số}"`

---

## 2. Kiến trúc Pipeline (Architecture)

Tất cả các bước dùng model **< 14B**: Qwen3-8B-AWQ (LLM), Vietnamese_Reranker fine-tuned (568M),
Vietnamese_Embedding_v2 (568M).

```
Câu hỏi (query)
   │
   ├─► Query Decomposition (Qwen3-8B-AWQ)  ──► 1-3 sub-queries
   │
   ├─► HyDE (Qwen3-8B-AWQ) ──► hypothetical legal article ──► Dense FAISS search
   │
   ├─► Dense retrieval (Vietnamese_Embedding_v2 + FAISS, 612K vectors, top-100)
   │
   └─► BM25S v7 (1.88M chunks gồm synthetic QA, top-150)
                  │
                  ▼
   Intersection (HyDE-candidates ∩ BM25S-candidates)  ──► ~29 candidates/query
        (recall ceiling ~90%, loại ~70% noise)
                  │
                  ▼
   Vietnamese_Reranker fine-tuned (checkpoint-8000) ──► top-5 candidates
        (recall@5 ≈ 80%, median rank GT = 2)
                  │
                  ▼
   Qwen3-8B-AWQ (enable_thinking=True, max_tokens=8000, content ≤10K chars) ──► chọn điều luật
                  │
                  ▼
   Format submission (relevant_docs + relevant_articles, avg ~2.7 articles/query, 0 empty)
                  │
                  ▼
   submission_3_1_llm8b_rerankerv2ck8k_top5.json   →  Articles F2 = 0.6056
                  │
                  ▼   (post-processing bằng LoRA classifier)
   LoRA Classifier (Qwen3-1.7B-Base + LoRA, checkpoint-23000)
     - Filter: bỏ articles có P(Có) < 0.1 (loại false positives)
     - Add: thêm articles từ reranker top-5 có P(Có) ≥ 0.7 (bổ sung recall)
                  │
                  ▼
   submission_3_1_combo_ck23k_f01_a07.json   (bài nộp cuối — articles + answer)
                  │
                  ▼   (sinh trường `answer`)
   Answer generation (Qwen3-8B-AWQ, enable_thinking=True, optimized prompt)
     - System prompt tách riêng: anti-hallucination, chỉ cite từ context
     - Few-shot từ data_final, max_tokens=8192
     - Kết thúc bằng "Căn cứ pháp lý:" (99% answers)
```

> **Hai tầng đầu ra**: (1) pipeline truy hồi sinh `relevant_articles` / `relevant_docs`;
> (2) bước sinh `answer` (`gen_answer.py`) điền trường văn bản trả lời dựa trên các điều luật
> đã chọn + few-shot. Prompt answer tách system/user message, enforce anti-hallucination
> (chỉ cite từ context), kết thúc bằng "Căn cứ pháp lý:". Xem §6.

### 2.1. Vì sao thiết kế nhiều tầng (multi-stage)?

Insight cốt lõi: **Reranker chỉ hiệu quả khi input sạch (ít noise).**

- Pool lớn (HyDE ~97 candidates/query): reranker *tệ hơn* dense score.
- Pool nhỏ (Intersection ~29 candidates/query): reranker *vượt trội* — median rank GT từ 4 → 2.

Do đó pipeline giảm noise dần qua từng tầng: retrieval rộng → intersection (lọc ~70% noise) →
reranker (top-5) → LLM chọn cuối. Mỗi tầng làm sạch input cho tầng sau.

---

## 3. Dữ liệu (Data)

Chi tiết đầy đủ trong [`docs/data_description.md`](docs/data_description.md).

| Nguồn | Mô tả | Quy mô |
|-------|-------|--------|
| `vbpl_dataset/chunks/` | Văn bản pháp luật đã chunk theo Điều | ~40,435 văn bản, ~461K điều |
| `vbpl_dataset/metadata/` | Metadata: tên, số hiệu, hiệu lực | ~42,232 văn bản |
| `data_final/` | QA có `article_cite` (ground truth tự xây) | 163,677 câu, 155,867 có cite |
| `synthetic_qa/` | QA tổng hợp (LLM sinh) làm "cầu nối ngôn ngữ" cho BM25S | ~1.27M QA (sau dedup) |
| `data_final_hard_negatives.jsonl` | **Train data reranker** | 96,456 dòng |
| `R2AIStage1DATA.json` | Tập test R2AI (chỉ question) | 2,000 câu |

**Nguồn thu thập**: vbpl_dataset từ vbpl.vn + vanban.chinhphu.vn + luatvietnam; data_final QA từ
vbpl.vn, chinhsachonline.vn, vksndtc.

**Data coverage**: 99.6% GT documents và 97.0% GT (doc_id, Điều) có trong corpus.
Bottleneck là **ranking quality**, không phải coverage.

---

## 4. Mô hình (Models)

Chi tiết đầy đủ trong [`docs/model_description.md`](docs/model_description.md). Tất cả < 14B.

| Model | Vai trò | Kích thước |
|-------|---------|-----------|
| `AITeamVN/Vietnamese_Embedding_v2` | Dense embedding (1024-dim) | 568M |
| `AITeamVN/Vietnamese_Reranker` | Base reranker (bge-reranker-v2-m3) | 568M |
| `reranker_finetuned_v2_ck8000` | Reranker fine-tuned (checkpoint-8000) | 568M |
| `Qwen3-8B-AWQ` | LLM decompose / HyDE / rerank / sinh answer | 8B |
| `Qwen/Qwen3-1.7B-Base` + LoRA ck-23000 | Classifier filter/add post-processing | 1.7B |

---

## 5. Cấu trúc thư mục submission

```
r2ai_submission/
├── README.md                  ← file này
├── code/                      ← script pipeline (xem §6) + code/vbpl_dataset/ (crawl + chunk)
├── data/                      ← symlink dữ liệu/index lớn + copy cache nhỏ (xem data/README.md)
├── models/
│   ├── reranker_finetuned_v2_ck8000/   ← checkpoint inference (đã bỏ optimizer.pt)
│   ├── qwen3-1.7b-base/               ← base model cho LoRA classifier (Qwen/Qwen3-1.7B-Base)
│   └── legal_classifier_lora_v2_ck23000/ ← LoRA adapter classifier (checkpoint-23000, final)
├── submissions/               ← file kết quả nộp
│   └── submission_3_1_combo_ck23k_f01_a07.json  ← BẢN NỘP CUỐI (articles + answer)
└── docs/
    ├── data_description.md
    ├── model_description.md
    ├── reproduce.md
    └── TBD.md                  ← danh sách mục cần user xác nhận
```

### Các script chính trong `code/`

| Script | Vai trò trong pipeline |
|--------|------------------------|
| `query_decomposition.py` | Tách câu hỏi phức → 1-3 sub-queries (Qwen3-8B-AWQ) |
| `hyde_retrieval.py` | HyDE: LLM sinh hypothetical article → dense search → `hyde_candidates.pkl` |
| `retrieval_dense.py` | Build + truy hồi dense (Vietnamese_Embedding_v2 + FAISS) |
| `retrieval_bm25s.py` | Build + truy hồi BM25S v7 (vi-tokenize + synthetic QA) |
| `rerank_intersection.py` | Tạo intersection pool (HyDE ∩ BM25S) + chấm Vietnamese_Reranker → `rerank_intersection_scores.pkl` |
| `run_llm_rerank_on_reranker_top5.py` | **Bước cuối: reranker top-5 → Qwen3-8B-AWQ thinking → submission** |
| `train_legal_classifier_v2.py` | Train LoRA classifier (Qwen3-1.7B-Base) trên hard negatives |
| `run_classifier_unsloth.py` | Inference classifier: batched logits P(Có) vs P(Không) |
| `ensemble_classifier_reranker.py` | Combo filter + add: bỏ FP + thêm high-confidence articles |
| `retrieval_llm_rerank.py` | LLM rerank gốc (chứa `make_doc_name`, `format_submission`, `get_text` dùng chung) |
| `retrieval_vn_rerank.py` | Two-stage BM25S → Vietnamese_Reranker (hybrid fusion) |
| `mine_hard_negatives_dense.py` | Sinh `data_final_hard_negatives.jsonl` (train data reranker) |
| `train_reranker_v2.py` | Fine-tune Vietnamese_Reranker trên hard negatives → checkpoint-8000 |
| `build_fewshot_index.py` | Build FAISS index trên data_final questions → `fewshot_index/` |
| `run_fewshot_rerank.py` | Search + rerank few-shot → `fewshot_top5_reranked.pkl` |
| `gen_answer.py` | **Sinh trường `answer`**: optimized prompt (anti-hallucination) + few-shot → `*_with_answer.json` |
| `gen_answer_fill_empty.py` | Fill bù answers rỗng (do articles không tìm thấy trong pool) |
| `strip_answer_labels.py` | Strip label "Tóm tắt:" khỏi đầu answer (post-processing) |
| `gen_synthetic_questions.py` | Sinh synthetic QA từ chunks VBPL (cầu nối ngôn ngữ cho BM25S) |
| `verify_submission.py` | Kiểm tra định dạng + thống kê submission |
| `clean_text.py` | Làm sạch text crawl (data_final) |
| `code/vbpl_dataset/{build,legal_chunker,export_chunks}.py` | Crawl + chunk corpus VBPL |

---

## 6. Tái lập kết quả (Reproduce)

Xem chi tiết trong [`docs/reproduce.md`](docs/reproduce.md). Tóm tắt:

```bash
# 1. Build index
python3 code/retrieval_dense.py build-index      # → retrieval_index_dense/
python3 code/retrieval_bm25s.py build-index      # → retrieval_index_bm25s_v7/

# 2. Serve Qwen3-8B-AWQ qua vLLM (port 8011) — xem docs/reproduce.md

# 3. Query decomposition + HyDE
python3 code/query_decomposition.py decompose --workers 8
python3 code/hyde_retrieval.py generate --port 8011
python3 code/hyde_retrieval.py retrieve --top-k 100      # → hyde_candidates.pkl

# 4. Serve reranker fine-tuned (pooling, port 8012) + tạo intersection scores
python3 code/rerank_intersection.py --port 8012          # → rerank_intersection_scores.pkl

# 5. Chọn điều luật: reranker top-5 → Qwen3-8B-AWQ thinking → submission (quyết định F2)
python3 code/run_llm_rerank_on_reranker_top5.py --port 8011 --model Qwen3-8B-AWQ --tag 8b
#   → submission_3_1_llm8b_rerankerv2ck8k_top5.json   (Articles F2 = 0.6056)

# 6. (Tùy chọn) Sinh trường `answer` đầy đủ
python3 code/build_fewshot_index.py
python3 code/run_fewshot_rerank.py --port 8011 --workers 8
python3 code/gen_answer.py --port 8011 --model Qwen3-8B-AWQ \
    --submission submission_3_1_llm8b_rerankerv2ck8k_top5.json --workers 4
#   → submission_3_1_llm8b_rerankerv2ck8k_top5_with_answer.json
```

> **Lưu ý chấm điểm**: trường `answer` ở bước 5 đã là chuỗi liệt kê điều luật dạng
> "Theo Điều X [tên văn bản]: ..." — benchmark trích "Điều X" để tính **Articles F2 = 0.6056**.
> Bước 6 thay `answer` bằng văn bản trả lời đầy đủ (cho các metric chất lượng câu trả lời).

---

## 7. Yêu cầu môi trường (Environment & Dependencies)

- **Python**: 3.13 (dev), tương thích 3.10+
- **GPU**: NVIDIA H20 (97 GB VRAM) — dùng cho vLLM serve LLM + embedding/reranker
- **Hệ điều hành**: Linux

### Pip packages (xem `code/requirements.txt`)

```
numpy, scipy, requests, tqdm
torch, transformers, sentence-transformers
faiss-cpu (hoặc faiss-gpu)
bm25s, underthesea          # BM25S + Vietnamese word segmentation
datasets
openai                      # client gọi vLLM (OpenAI-compatible API)
vllm                        # serve LLM (port 8011) + reranker pooling (port 8012)
```

> `vllm` là **runtime bắt buộc** để serve Qwen3-8B-AWQ (port 8011) và reranker (port 8012).
> Các script gọi qua HTTP `requests` đến endpoint OpenAI-compatible.

---

## 8. Các hướng đã thử & thất bại (tóm tắt)

Để tránh lặp lại, các hướng sau đã được kiểm chứng là **không cải thiện**:

- Query expansion (multi-query RRF), Hybrid score fusion (BM25 + Dense)
- Synthetic QA cho dense index (chỉ hữu ích cho BM25S keyword matching)
- Cross-encoder rerankers generic (jina-reranker-v3, zerank-2): phá F2
- E5-base multilingual (thua Vietnamese_Embedding_v2)
- Answer-first retrieval, Citation following, lọc theo hiệu lực

Bài nộp cuối: pipeline 8B + LoRA classifier 1.7B ck-23000 (`submission_3_1_combo_ck23k_f01_a07.json`). Tất cả model < 14B.

---

## 9. Mục cần xác nhận

Xem [`docs/TBD.md`](docs/TBD.md) — trạng thái các mục cần xác nhận (hầu hết đã hoàn tất).
