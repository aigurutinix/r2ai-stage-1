# AI Legal Assistant — Truy hồi & Hỏi đáp Văn bản Pháp luật Tiếng Việt

Hệ thống AI **truy hồi (IR) & hỏi đáp (QA)** văn bản pháp luật tiếng Việt cho doanh nghiệp **SME**, xây dựng cho cuộc thi **ROAD TO AI 2026 (R2AI)**.

Pipeline xử lý hàng loạt **offline**: nhận 2000 câu hỏi pháp lý → truy hồi đúng **Điều luật** liên quan → (tùy chọn) sinh câu trả lời có dẫn nguồn → xuất `results.json` để nộp lên leaderboard.

> **Kết quả chính thức đã nộp:** `ARTICLES_F2_MACRO = 0.5766`
> (metric xếp hạng = F2 macro trên cặp **(mã văn bản, số Điều)**, ưu tiên recall: `F2 = 5PR / (4P + R)`).

| Chỉ số | Giá trị |
|---|---|
| **ARTICLES_F2_MACRO** (xếp hạng) | **0.5766** |
| ARTICLES_PRECISION | 0.504 |
| ARTICLES_RECALL | 0.6397 |
| DOCS_F2 (phụ) | 0.607 |
| DOCS_RECALL | 0.6867 |
| QA (4 tiêu chí, LLM-judge) | 0.0 — chưa promote tuần (không bắt buộc cho điểm IR) |
| Tham chiếu top leaderboard | ~0.69 |

> Leaderboard: <http://leaderboard.aiguru.com.vn/> · Nộp **offline** file `results.json` (2000 câu), nén `submission.zip` (zip phẳng). Đóng cổng **30/06/2026**.
> Đề bài chi tiết: [docs/competition-overview.md](docs/competition-overview.md) · Kiến trúc: [docs/pipeline-architecture.md](docs/pipeline-architecture.md)

> **Trung thực (quan trọng):** `0.5766` đạt được bằng reranker **off-the-shelf** `AITeamVN/Vietnamese_Reranker` ở `RERANK_MAX=512`, trên corpus HF 93K — **KHÔNG** dùng checkpoint tự train. Checkpoint dùng cho điểm này = **3 model gốc public HF** (xem §6). Fine-tune (§8) là hướng cải tiến đã build pipeline nhưng **TÙY CHỌN** (chưa chắc đã train/đo). `gold_dev.json` là gold **synthetic** (chỉ so sánh tương đối).

---

## 1. Tóm tắt sản phẩm

- **Đầu vào:** 2000 câu hỏi pháp lý tiếng Việt — `data/stage1_questions.json` (mỗi phần tử `{id, question}`).
- **Đầu ra:** `results.json` (UTF-8) = list các bản ghi:
  ```json
  {"id": 1, "question": "...", "answer": "...",
   "relevant_docs": ["45/2019/QH14|Bộ luật Lao động"],
   "relevant_articles": ["45/2019/QH14|Bộ luật Lao động|Điều 90"]}
  ```
  nén `submission.zip` (zip **phẳng**: `results.json` ở gốc; tên file bắt buộc `results.json`).
- **Chấm điểm:** metric xếp hạng = **ARTICLES_F2_MACRO** (khớp **chính xác** trên `(mã văn bản, số Điều)`). DOCS_F2 phụ. QA chấm thủ công/LLM-judge chỉ trên bài được promote hằng tuần.
- **Ràng buộc thi:** chỉ dùng mô hình **mã nguồn mở, < 14B, công bố trước 01/03/2026**. Cấm mô hình đóng (GPT/Gemini).
- **Môi trường tái hiện:** **Kaggle Notebook, GPU T4 ×2, Internet ON** (luồng chính tạo 0.5766). Máy local chỉ để dev (không chạy model nặng).

---

## 2. Yêu cầu môi trường

### 2.1 Kaggle (luồng chính — TÁI HIỆN 0.5766)

| Mục | Cấu hình |
|---|---|
| Nền tảng | Kaggle Notebook |
| Accelerator | **GPU T4 ×2** (16GB × 2 — bắt buộc cho Qwen fp16 7B ~15GB trải 2 GPU) |
| Internet | **On** (tải HF model + dataset) |
| Python | 3.10 / 3.11 |
| Quota | Phiên ~12h, 30h GPU/tuần — đủ cho 2000 câu trong 1 phiên |

### 2.2 Local (dev / sweep cutoff — không chạy model nặng)

- Mac/Linux, Python 3.10+, CPU/MPS. Đủ để chạy `scratch/sweep_cutoff.py` (CPU, không cần GPU) và đọc/sửa code.
- **KHÔNG** dùng máy local để chạy embedding/rerank/LLM trên 2000 câu (quá chậm/lag).

---

## 3. Kiến trúc pipeline

Metric xếp hạng (ARTICLES_F2) **chỉ phụ thuộc retrieval** (Phase A + cutoff). Phase B (Qwen sinh answer) chỉ phục vụ QA, **không** ảnh hưởng ARTICLES_F2.

```
                          corpus_articles.jsonl (~93K Điều, HF tmquan/vbpl-vn lọc SME)
                                       │
                                       ├──────────────► BGE-M3 embed ──► corpus_emb.npy (N×1024, ~381MB)
                                       │
  stage1_questions.json (2000 câu) ────┤
                                       ▼
        ┌──────────────────────── Phase A (mỗi câu hỏi) ───────────────────────────┐
        │  Dense (Vietnamese_Embedding, cosine numpy, max_seq 1024)                 │
        │              ‖                                                            │
        │  BM25 (rank_bm25)                                                         │
        │              ▼                                                            │
        │  RRF có trọng số (K=60, W_BM25=0.65, W_DENSE=0.35) → CAND=80 ứng viên      │
        │              ▼                                                            │
        │  Cross-encoder rerank (Vietnamese_Reranker, RERANK_MAX=512) → top-20+score │
        └────────────────────────────────► retrieved.json ─────────────────────────┘
                                       │
                                       ▼
        Cutoff "t5m3" (top-5, margin 3.0) + drop_superseded  ──►  relevant_docs / relevant_articles
                                       │                                  │
                                       │            (Phase B, tùy chọn)   │
                                       ▼                                  ▼
        Qwen2.5-7B-Instruct (fp16, device_map=auto, 2×T4) ── ép trích "Điều X" ──► answer
                                                                          │
                                                                          ▼
                                                          results.json + submission.zip
```

**Tóm tắt 4 bước:**

1. **Corpus + embeddings** — `corpus_articles.jsonl` (1 Điều/dòng) + `corpus_emb.npy` (BGE-M3, float32, row *i* ↔ dòng *i*).
2. **Phase A (retrieval)** — Dense ‖ BM25 → RRF có trọng số → CAND=80 → rerank cross-encoder → lưu **top-20 + điểm rerank** vào `retrieved.json`.
3. **Cutoff (rẻ, offline, CPU)** — `scratch/sweep_cutoff.py` đọc `retrieved.json` → áp cutoff **t5m3** + `drop_superseded` → dựng `relevant_docs`/`relevant_articles` → `results.json` + `submission.zip`.
4. **Phase B (QA answer, tùy chọn)** — Qwen2.5-7B fp16 sinh answer ~200–350 từ, ép trích "Điều X". Không ảnh hưởng ARTICLES_F2.

---

## 4. Hướng dẫn tái hiện 0.5766 từ đầu

> **Cấu hình chốt:** corpus HF 93K · `RERANK_MAX=512` · cutoff `t5m3` (top-5, margin 3.0) · RRF `W_BM25=0.65 / W_DENSE=0.35` (K=60) · CAND=80.

### Bước 1 — Chuẩn bị dữ liệu

**Cách A (khuyến nghị, nhanh): tải dữ liệu dẫn xuất từ Drive.**
Tải 3 file lên một Kaggle Dataset rồi **Add Input** vào notebook:

| File | Mô tả | Kích thước |
|---|---|---|
| `corpus_articles.jsonl` | ~93K Điều (HF `tmquan/vbpl-vn` lọc SME) | ~91MB |
| `corpus_emb.npy` | BGE-M3 embeddings float32 (N×1024), khớp dòng với jsonl | ~381MB |
| `stage1_questions.json` | 2000 câu test (BTC cấp) | ~0.5MB |

> Drive: **{{GOOGLE_DRIVE_LINK_DATA}}** *(ĐIỀN SAU KHI UPLOAD)*
> Lưu ý: `data/corpus_emb.npy` trong repo chỉ là **symlink ~86 byte** trỏ tới `backup/time2/corpus_emb.npy` — bản `.npy` thật (~381MB) nằm trên Drive/Kaggle, không commit vào git.

**Cách B (build lại từ đầu, chậm):**
```bash
# Trên Kaggle (Internet ON) — KHÔNG chạy trên Mac
python backend/build_corpus.py     # HF tmquan/vbpl-vn → data/corpus_articles.jsonl (~93K Điều, lọc SME)
python backend/embed_corpus.py     # incremental embed → data/corpus_emb.npy (BGE-M3, 1024-dim)
```
> `stage1_questions.json` do **BTC cấp** — đặt sẵn ở `data/`. BTC **không** cấp corpus/train; corpus tự thu thập từ HF dataset.
> **Lưu ý revert vbpl:** Notebook có **Phase 0** (cell 6) build corpus từ API vbpl.vn chính thống, nhưng corpus đó đo **tệ hơn** (0.45/0.51 < 0.5766) → **đã revert**. Tái hiện 0.5766 dùng **corpus HF 93K** (Cách A) và **bỏ qua Phase 0** bằng cách Add Input corpus + emb cũ.

### Bước 2 — Cài dependencies

Trên **Kaggle** (đã có sẵn trong cell 2 của notebook):
```bash
pip install -q "sentence-transformers>=3.0" "transformers>=4.44" accelerate rank_bm25 datasets
```
Local dev (chỉ để đọc code + sweep cutoff):
```bash
pip install -r backend/requirements-local.txt
# sentence-transformers>=3.0.0, torch>=2.2.0, numpy>=1.24, rank-bm25>=0.2.2, datasets>=2.18.0, pyvi (optional)
```

### Bước 3 — Chạy Phase A trên Kaggle (`kaggle/full_pipeline_kaggle.ipynb`)

Notebook **22 cells**. Cấu hình Settings (panel phải):
- **Accelerator → GPU T4 ×2** · **Internet → On**
- **Add Input**: dataset chứa `corpus_articles.jsonl` (93K) + `corpus_emb.npy` (của corpus đó) + `stage1_questions.json`.

Chạy các cell (đánh số 0-based như trong file `.ipynb`):

| Cell | Tác dụng |
|---|---|
| 2 (`## 1`) | Cài thư viện (`pip install …`) |
| 4 (`## 2`) | `nvidia-smi -L` — xác nhận 2× T4 |
| 6 (`Phase 0`) | **BỎ QUA** (build corpus vbpl đã revert) vì đã Add Input corpus 93K |
| 8 (`## 3`) | Tìm file đầu vào ở `/kaggle/input` hoặc `/kaggle/working` |
| **10 (`Phase A`)** | **Retrieval** — Dense ‖ BM25 → RRF → CAND=80 → rerank → lưu **top-20 + score** vào `/kaggle/working/retrieved.json`. `corpus_emb.npy` khớp 93K → **bỏ qua embed**. |

> ⚠️ **Trước khi chạy Phase A: XÓA `retrieved.json` cũ** trong `/kaggle/working` (nếu có) — nếu không, cell 10 phát hiện cache sẽ **skip** Phase A (in `"Đã có retrieved.json … → BỎ QUA Phase A"`).
> ⚠️ **Đặt `RERANK_MAX = 512`** trong cell 10. Notebook hiện để `1024` (probe). Giá trị **đã chấm 0.5766 là 512** → sửa dòng:
> ```python
> MAX_SEQ, RERANK_MAX, CAND, CAND_SAVE, RR_BATCH = 1024, 512, 80, 20, 16
> ```
> (`RR_BATCH=16` an toàn VRAM ở 512 tok; ở 1024-probe notebook để `RR_BATCH=8`.)

Kết thúc Phase A → tải `retrieved.json` từ tab **Output** (hoặc **Save Version** để giữ giữa các phiên).

### Bước 4 — Sweep cutoff → `submission.zip` (local, CPU)

`retrieved.json` (top-20 + score) cho phép áp cutoff **offline, không cần GPU**:
```bash
python scratch/sweep_cutoff.py \
  --retrieved retrieved.json \
  --questions data/stage1_questions.json \
  --base kaggle/results.json \
  --outdir sweep
# → sweep/results_t5m3.json + sweep/submission_t5m3.zip
#   (cùng các biến thể t3m15, t6m4; v_k6 chỉ chạy khi có --verified)
```
- `t5m3` = **top-5, margin 3.0** (giữ điều có `score >= top1_score − 3.0`) + `drop_superseded` (bỏ luật bị thay thế + dedup version cũ). **t5m3 cho điểm cao nhất (0.5766).**
- `--base` giữ prose answer LLM (nếu đã chạy Phase B); thiếu nó thì answer chỉ liệt kê căn cứ (đủ để đo IR).
- Mặc định khi không truyền cờ: `--retrieved backup/retrieved.json`, `--base results.json`, `--questions data/stage1_questions.json`, `--outdir sweep`.

### Bước 5 — Nộp

Tải `sweep/submission_t5m3.zip` (hoặc `submission.zip` sinh từ Phase B notebook, cell 20) → nộp tại <http://leaderboard.aiguru.com.vn/>. Vòng public ~10 bài/ngày.

### (Tùy chọn) Phase B — sinh answer cho QA

Trong cùng notebook, chạy tiếp các cell Phase B:
- **cell 12** — nạp `Qwen/Qwen2.5-7B-Instruct` fp16 (`device_map="auto"` trải 2×T4) qua `transformers` (KHÔNG dùng Ollama trên Kaggle).
- **cell 18** — sinh answer FULL 2000 câu (batched `BATCH=6`, checkpoint/resume mỗi lô).
- **cell 20** — đóng `submission.zip` (flat, `results.json` ở gốc).

~2–4h. **Không** ảnh hưởng ARTICLES_F2 (daily leaderboard chỉ chấm IR).

---

## 5. Bảng kết quả & cấu hình chốt

### 5.1 Hành trình điểm (tiến hoá)

| Điểm F2 (articles) | Thay đổi chính |
|---|---|
| 0.317 | baseline |
| 0.3877 | cutoff t3m3 |
| 0.4616 | **corpus HF 93K** |
| 0.4887 | lọc hiệu lực (`drop_superseded`) |
| 0.5371 | phễu rerank 50 + t3m15 |
| 0.5608 | phễu-80 + cutoff t5m3 |
| **0.5766** | **RERANK_MAX 256 → 512** ✅ |

### 5.2 Cấu hình chốt (đã chấm 0.5766)

| Tham số | Giá trị | Nguồn |
|---|---|---|
| Corpus | HF 93K (`corpus_articles.jsonl`) | — |
| Embedding | `AITeamVN/Vietnamese_Embedding`, max_seq 1024 | `local_models_config.py` (`EMBED_MAX_SEQ_LEN`) |
| BM25 | `rank_bm25` (BM25Okapi) | — |
| Fusion | RRF có trọng số: `K=60`, `W_BM25=0.65`, `W_DENSE=0.35` | `local_models_config.py` (`RRF_K / RRF_W_BM25 / RRF_W_DENSE`) |
| Phễu ứng viên | `CAND=80` | notebook cell 10 |
| Reranker | `AITeamVN/Vietnamese_Reranker`, **`RERANK_MAX=512`**, `RR_BATCH=16` | notebook cell 10 · `local_models_config.py` (`RERANK_MAX_LEN=512`) |
| Lưu | top-20 + điểm rerank → `retrieved.json` | notebook cell 10 (`CAND_SAVE=20`) |
| Cutoff | **`t5m3`** = top-5, margin 3.0 + `drop_superseded` | `sweep_cutoff.py` GRID |

> Notebook đang để `RERANK_MAX=1024` ở cell 10 (probe đọc dài hơn) — **512 là cấu hình chính thức** đã tạo ra 0.5766. `local_models_config.py` đã đặt `RERANK_MAX_LEN = 512`.

---

## 6. Mô hình & tuân thủ

Tất cả TUÂN THỦ ràng buộc thi (mã nguồn mở, < 14B, công bố trước 01/03/2026):

| Vai trò | Mô hình | Tham số / License / Năm | HF |
|---|---|---|---|
| Embedding | `AITeamVN/Vietnamese_Embedding` (BGE-M3 fine-tune VN, 1024-dim) | ~0.6B · MIT · 2025 | <https://huggingface.co/AITeamVN/Vietnamese_Embedding> |
| Reranker | `AITeamVN/Vietnamese_Reranker` (bge-reranker-v2-m3, `XLMRobertaForSequenceClassification`, num_labels=1) | ~0.6B · MIT · 2025 | <https://huggingface.co/AITeamVN/Vietnamese_Reranker> |
| LLM (QA) | `Qwen/Qwen2.5-7B-Instruct` | 7.6B · Apache-2.0 · 2024-09 | <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct> |

- Chọn embedding **v1** (không v2) vì v2 yếu hơn ở domain legal.
- Checkpoint dùng cho 0.5766 = **3 model gốc public HF ở trên** — không có checkpoint tự train. "Phiên bản" = pin theo HF revision/commit khi nộp (`{{HF_REVISION}}` — ĐIỀN SAU).
- Hồ sơ nộp & chứng minh tuân thủ: [docs/submission/01-mo-ta-du-lieu.md](docs/submission/01-mo-ta-du-lieu.md) (mô tả dữ liệu) · [docs/submission/02-mo-hinh-su-dung.md](docs/submission/02-mo-hinh-su-dung.md) (mô hình sử dụng). Đề bài + metric: [docs/competition-overview.md](docs/competition-overview.md).

---

## 7. Dữ liệu

### 7.1 Dữ liệu dẫn xuất (cần share qua Drive: **{{GOOGLE_DRIVE_LINK_DATA}}** — ĐIỀN SAU KHI UPLOAD)

| File | Mô tả | Schema / Kích thước |
|---|---|---|
| `data/corpus_articles.jsonl` | ~93K Điều (HF `tmquan/vbpl-vn` lọc SME); 1 Điều/dòng | `{id, doc_number, clean_name, legal_type, year, article, title, text, source_url}` · ~91MB |
| `data/corpus_emb.npy` | BGE-M3 embeddings float32, N×1024 (row *i* ↔ dòng *i* jsonl) | ~381MB (repo chỉ có symlink ~86B → `backup/time2/`) |
| `data/stage1_questions.json` | 2000 câu test (BTC cấp) | `{id, question}` · ~0.5MB |

> Khóa chấm điểm = `(doc_number, article)`, ví dụ `("45/2019/QH14", "Điều 12")`.

### 7.2 Dữ liệu phụ trợ (eval/phân tích, không bắt buộc share)

- `data/zalo_eval.json` — Zalo Legal **3196 câu** có gold `mã|Điều` (proxy eval offline).
- `data/gold_dev.json` — **20 câu** gold **SYNTHETIC** (nhóm tự suy từ kiến thức luật VN — chỉ so sánh tương đối, **không** phải gold thật; cấu trúc `{_note, items:[…]}`).
- `data/sme_doc_ids_all.json` — 8020 `docNum → vbpl id`.
- `data/law.json`, `data/document_registry.json`, `data/legal_list_id.json` — thư viện web-app (không dùng cho phần thi).

---

## 8. Hướng cải tiến: fine-tune reranker (TÙY CHỌN)

> ⚠️ **Đã build pipeline, CHƯA chắc đã train/đo.** Đây là HƯỚNG cải tiến — **không** phải checkpoint tạo ra 0.5766.

Fine-tune `AITeamVN/Vietnamese_Reranker` bằng synthetic data + hard negatives (vẫn tuân thủ: base là model mở < 14B):

- Notebook end-to-end: [`kaggle/finetune_reranker_kaggle.ipynb`](kaggle/finetune_reranker_kaggle.ipynb) (15 cells).
- Pipeline: `scratch/finetune/gen_synthetic_pairs.py` (Qwen sinh query) → `mine_hard_negatives.py` (BGE-M3) → `train_reranker.py` (FlagEmbedding DDP 2×T4) → `eval_reranker.py`.
- Deps thêm trên Kaggle: `pip install "FlagEmbedding>=1.3" deepspeed`.
- Checkpoint kỳ vọng: `/kaggle/working/ft_reranker` → chia sẻ qua **{{GOOGLE_DRIVE_LINK_CHECKPOINT}}** *(ĐIỀN SAU KHI UPLOAD)*.

---

## 9. Cấu trúc thư mục

```
.
├── backend/                          # Pipeline + serving
│   ├── local_models_config.py        # Config tập trung: model id, allowlist SME, cutoff, RRF weights
│   ├── build_corpus.py               # HF vbpl-vn → corpus_articles.jsonl
│   ├── embed_corpus.py               # Incremental embed → corpus_emb.npy
│   ├── local_rag_engine.py           # Hybrid retrieval (numpy cosine + BM25 + RRF + rerank)
│   ├── local_reranker.py             # Cross-encoder reranker
│   ├── local_llm_client.py           # Ollama client + prompt (luồng local)
│   ├── legal_text_parser.py          # Parse tên luật + tách Điều
│   ├── retrieval_cutoff.py           # apply_cutoff + drop_superseded + SUPERSEDED_DOCS
│   ├── vbpl_fetch.py                 # Fetch API vbpl chính thống (ĐÃ REVERT — corpus đo tệ hơn)
│   ├── requirements-local.txt        # Deps pipeline thi
│   └── requirements.txt              # Deps web-app FastAPI (KHÔNG cần cho thi)
├── scratch/
│   ├── sweep_cutoff.py               # ★ Sweep cutoff offline từ retrieved.json → submission zip
│   ├── generate_submission.py        # Sinh submission local (Ollama)
│   ├── eval_f2.py                    # Đo P/R/F2 vs gold_dev
│   ├── audit_coverage.py             # Đo trần coverage corpus
│   ├── rebuild_corpus_vbpl.py        # (revert) build corpus từ API vbpl
│   ├── zalo-eval-harness.py          # Eval trên Zalo Legal
│   └── finetune/                     # gen_synthetic_pairs · mine_hard_negatives · train_reranker · eval_reranker
├── kaggle/
│   ├── full_pipeline_kaggle.ipynb    # ★ LUỒNG CHÍNH tạo 0.5766 (22 cells)
│   ├── finetune_reranker_kaggle.ipynb# Fine-tune reranker (cải tiến, tùy chọn — 15 cells)
│   └── results.json                  # Bản submission tham chiếu
├── data/                             # corpus + embeddings + questions (xem §7)
├── docs/                             # competition-overview.md · pipeline-architecture.md · submission/
└── plans/                            # Kế hoạch + research reports
```

---

## 10. Lưu ý vận hành

- **`(mã văn bản, số Điều)` là khóa chấm điểm** — corpus phải có mã chính xác.
- **Corpus coverage = trần recall** — thiếu luật gold → mất điểm câu đó. Mở rộng `SME_LAW_ALLOWLIST` trong `backend/local_models_config.py`.
- **Sweep cutoff là khâu chốt & rẻ** — đổi cutoff không cần chạy lại GPU; chỉ cần `retrieved.json` (top-20 + score).
- **Bảo mật:** không commit `.env`, API key, credentials.
- **Web-app** (`backend/requirements.txt`, FastAPI) **KHÔNG** dùng cho phần thi — bản web cũ đã bị loại tư cách do dùng model đóng (`google-antigravity`); chỉ giữ cho sản phẩm sau.
