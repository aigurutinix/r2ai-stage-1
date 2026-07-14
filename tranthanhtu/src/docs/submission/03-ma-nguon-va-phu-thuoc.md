# 03 — Mã nguồn & Phụ thuộc

Tài liệu này liệt kê **toàn bộ mã nguồn**, **thư viện/phụ thuộc** và **tệp cấu hình** cần thiết để tái hiện kết quả chính thức **ARTICLES_F2_MACRO = 0.5766** của hệ thống *AI Legal Assistant* (cuộc thi ROAD TO AI 2026 — R2AI).

> **Lưu ý phạm vi (đọc trước):** Repo chứa hai phần độc lập:
> - **LUỒNG THI** — pipeline truy hồi/QA offline chạy trên Kaggle (transformers fp16, 2×T4), tạo ra `results.json` nộp leaderboard. Đây là phần được chấm điểm. Con số chính thức **0.5766** đạt trên **corpus HF ~93K Điều** + reranker **off-the-shelf** @ **RERANK_MAX 512**.
> - **WEB-APP** — bản FastAPI/Ollama trình diễn sản phẩm. **KHÔNG dùng cho phần thi** (bản web cũ từng bị loại tư cách vì dùng model đóng). Chỉ giữ làm sản phẩm hậu kỳ.
>
> Mỗi mục dưới đây đánh dấu rõ `[THI]` / `[WEB]` / `[CHUNG]` / `[TÙY CHỌN]`.

---

## 1. Toàn bộ mã nguồn

### 1.1. Repository

| Hạng mục | Giá trị |
|---|---|
| GitHub | `https://github.com/lehuubao2909/ai_legal_assistant` |
| Branch nộp | `master` |
| Phiên bản model | pin theo HF revision/commit khi nộp — xem mục 2.6 và placeholder `{{HF_REVISION}}` |

> Dữ liệu dẫn xuất nặng (corpus, embeddings) **không** nằm trong Git (gitignored — xem mục 3.2) — chia sẻ qua Drive: `{{GOOGLE_DRIVE_LINK_DATA}}` *(ĐIỀN SAU KHI UPLOAD)*.

### 1.2. Cây thư mục (có chú thích vai trò)

```
ai_legal_assistant/
├── README.md                         [CHUNG]  Tổng quan repo (mô tả luồng — tham khảo)
├── requirements.txt                  [THI]    ★ Deps TÁI HIỆN pipeline thi (entrypoint chuẩn — mục 2.1)
├── .gitignore                        [CHUNG]  Loại secrets + artifact nặng khỏi Git (mục 3.2)
│
├── kaggle/                           [THI]    ★ LUỒNG TÁI HIỆN CHÍNH
│   ├── full_pipeline_kaggle.ipynb    [THI]    ★ Notebook 22 cells tạo ra 0.5766 (chạy 2×T4)
│   ├── finetune_reranker_kaggle.ipynb[TÙY CHỌN] Notebook 15 cells fine-tune reranker (cải tiến)
│   └── results.json                  [THI]    Bài nộp mẫu (output của notebook)
│
├── backend/                          [CHUNG]  Pipeline + serving (config + module dùng chung)
│   ├── local_models_config.py        [THI]    ★ Config TẬP TRUNG: model id, allowlist SME,
│   │                                          cutoff, RRF weights, RERANK_MAX_LEN (mục 3.1)
│   ├── build_corpus.py               [THI]    HF tmquan/vbpl-vn → corpus_articles.jsonl (lọc SME)
│   ├── embed_corpus.py               [THI]    Embed tăng dần (incremental) → corpus_emb.npy
│   ├── retrieval_cutoff.py           [THI]    apply_cutoff + drop_superseded + SUPERSEDED_DOCS
│   │                                          (pure-python, không deps nặng)
│   ├── legal_text_parser.py          [THI]    Tách "Điều X" trong văn bản pháp luật
│   ├── local_rag_engine.py           [WEB]    Hybrid retrieval live (numpy cosine + BM25 → RRF → rerank)
│   ├── local_reranker.py             [WEB]    Wrapper cross-encoder rerank (serving)
│   ├── local_llm_client.py           [WEB]    Client LLM qua Ollama (sinh answer)
│   ├── vbpl_fetch.py                 [THI*]   Fetch API vbpl chính thống — ĐÃ THỬ, ĐÃ REVERT
│   │                                          (corpus vbpl đo ~0.45/0.51 < 0.5766; giữ làm tham khảo)
│   ├── requirements-local.txt        [THI]    Deps pipeline thi cho LOCAL dev (mục 2.2)
│   ├── requirements.txt              [WEB]    Deps web-app FastAPI — KHÔNG cần cho thi (mục 2.4)
│   ├── .env.example                  [WEB]    Mẫu biến môi trường web-app (Ollama host…)
│   ├── rebuild_backend_env.sh        [WEB]    Script dựng venv web-app
│   ├── main.py / agent.py / rag_engine.py / ingestion.py
│   │                                 [WEB]    Bản web cũ (google-antigravity, model đóng) — KHÔNG dự thi
│   └── document_generator.py         [WEB]    Sinh văn bản .docx (tính năng sản phẩm)
│
├── scratch/                          [THI]    Công cụ offline (đo/sweep, không cần GPU)
│   ├── sweep_cutoff.py               [THI]    ★ KHÂU CHỐT: retrieved.json → results_<tag>.json
│   │                                          + submission_<tag>.zip (cutoff t5m3 cho 0.5766)
│   ├── generate_submission.py        [THI]    Sinh submission local (đường nhánh local)
│   ├── eval_f2.py                    [THI]    Đo P/R/F2 vs gold_dev (gold SYNTHETIC — tương đối)
│   ├── audit_coverage.py             [THI]    Đo trần coverage của corpus/allowlist
│   ├── rebuild_corpus_vbpl.py        [THI*]   Dựng lại corpus từ vbpl (đường đã revert)
│   ├── zalo-eval-harness.py          [THI]    Eval proxy trên Zalo Legal (3196 câu gold)
│   ├── tune_retrieval.py             [THI]    Thử nghiệm tham số retrieval
│   ├── demo_query.py                 [THI]    Truy vấn thử 1 câu (debug)
│   ├── py.sh / run_build_corpus.sh   [THI]    Helper chạy script (đặt PYTHONPATH backend/)
│   └── finetune/                     [TÙY CHỌN] Pipeline fine-tune reranker (HƯỚNG cải tiến)
│       ├── gen_synthetic_pairs.py    [TÙY CHỌN] Sinh cặp (query, điều) synthetic từ corpus
│       ├── mine_hard_negatives.py    [TÙY CHỌN] Đào hard negatives (arXiv 2412.00657)
│       ├── train_reranker.py         [TÙY CHỌN] Launcher fine-tune (FlagEmbedding, 2×T4)
│       └── eval_reranker.py          [TÙY CHỌN] Gate offline base vs fine-tuned (held-out split)
│
├── data/                             [CHUNG]  Dữ liệu (phần lớn dẫn xuất — share qua Drive)
│   ├── stage1_questions.json         [THI]    2000 câu test BTC cấp ({id, question}) — tracked Git
│   ├── corpus_articles.jsonl         [THI]    ~93K Điều, ~91MB (dẫn xuất — GITIGNORED, share Drive)
│   ├── corpus_emb.npy                [THI]    BGE-M3 embeddings N×1024 ~381MB (GITIGNORED; trong
│   │                                          checkout local là SYMLINK 86 byte → bản thật ở Drive/Kaggle)
│   ├── sme_doc_ids_all.json          [THI]    8020 docNum → vbpl id (hỗ trợ build; GITIGNORED)
│   ├── sme_doc_ids_existing.json     [THI]    Tập docNum đã có trong corpus (GITIGNORED)
│   ├── zalo_eval.json                [THI*]   Zalo Legal 3196 câu gold (proxy eval; GITIGNORED, phụ trợ)
│   ├── gold_dev.json                 [THI]    20 câu gold SYNTHETIC (tương đối, KHÔNG gold thật) — tracked
│   ├── test_questions.json           [THI]    Tập câu hỏi nhỏ để smoke-test — tracked Git
│   ├── law.json                      [WEB]    Thư viện văn bản app (KHÔNG dùng cho thi) — tracked
│   ├── document_registry.json        [WEB]    Đăng ký văn bản app (KHÔNG dùng cho thi) — tracked
│   ├── legal_list_id.json            [WEB]    Danh mục id app (KHÔNG dùng cho thi) — tracked
│   └── chroma_db/                    [WEB]    Vector store cũ (legacy, GITIGNORED)
│
├── docs/                             [CHUNG]  Tài liệu
│   ├── competition-overview.md       [CHUNG]  Tổng quan cuộc thi
│   ├── pipeline-architecture.md      [CHUNG]  Kiến trúc pipeline
│   └── submission/                   [CHUNG]  Hồ sơ nộp (tài liệu này nằm ở đây)
│
├── plans/                            [CHUNG]  Kế hoạch + research reports
│
├── frontend/ , frontend_vanilla/     [WEB]    UI web-app (KHÔNG dùng cho thi)
├── colab/                            [WEB]    Notebook Colab cũ (full_pipeline_colab.ipynb — đã thay bằng Kaggle)
├── backup/ , sweep/                  [CHUNG]  Artifact tạm (gitignored): embeddings backup, output sweep
```

`★` = file trung tâm của luồng thi. `[THI*]` = thuộc luồng thi nhưng là đường đã thử rồi REVERT, hoặc dữ liệu eval phụ trợ (giữ làm bằng chứng quyết định kỹ thuật / đo proxy; KHÔNG nằm trên đường tái hiện 0.5766).

### 1.3. Đường tái hiện 0.5766 — các file đụng tới

Thứ tự thực thi (chi tiết bước trong tài liệu kiến trúc/quy trình):

| Bước | File | Vai trò |
|---|---|---|
| 0 | `backend/build_corpus.py` + `backend/embed_corpus.py` | Dựng `corpus_articles.jsonl` (~93K Điều, từ HF `tmquan/vbpl-vn`) + `corpus_emb.npy` (BGE-M3). Chạy 1 lần, **share qua Drive** rồi nạp lại bằng **Add Input** trên Kaggle. |
| 1–2 | `kaggle/full_pipeline_kaggle.ipynb` (Phase A) | Dense (Vietnamese_Embedding) ‖ BM25 → RRF có trọng số → CAND=80 → rerank (Vietnamese_Reranker, RERANK_MAX **512**) → lưu top-20 + score vào `retrieved.json`. |
| 3 | `scratch/sweep_cutoff.py` → `backend/retrieval_cutoff.py` | Áp cutoff **t5m3** (top-5, margin 3.0) + `drop_superseded` → dựng `relevant_docs`/`relevant_articles` → `results.json` + `submission.zip`. **Đây là cutoff cho điểm cao nhất.** |
| 4 *(không ảnh hưởng ARTICLES_F2)* | `kaggle/full_pipeline_kaggle.ipynb` (Phase B) | Qwen2.5-7B-Instruct fp16 (device_map=auto, 2×T4) sinh answer ~200–350 từ, ép trích "Điều X". |

> **⚠ Hai chốt BẮT BUỘC để khớp con số chính thức 0.5766** (notebook hiện đang ở trạng thái **probe**, không tái hiện 0.5766 nếu chạy y nguyên):
> 1. **Corpus = HF 93K, KHÔNG để Phase 0 build vbpl.** Cell 6 (Phase 0) có thứ tự ưu tiên: (a) corpus sẵn ở `/kaggle/working`, (b) **Add Input** `corpus_articles.jsonl`, (c) **nếu không có cả hai → build từ vbpl** qua `rebuild_corpus_vbpl.py` (~40939 Điều — đây là đường **đã REVERT**, đo tệ hơn). Để tái hiện 0.5766 **phải Add Input cặp `corpus_articles.jsonl` (93K) + `corpus_emb.npy`** để Cell 6 dùng nhánh (b) và bỏ qua build vbpl.
> 2. **`RERANK_MAX = 512`.** Notebook hiện để `RERANK_MAX = 1024` (đang **probe**) và `RR_BATCH = 8` (1024 tok ngốn VRAM gấp đôi). Đặt lại **`RERANK_MAX = 512`** và **`RR_BATCH = 16`** trong cell tham số Phase A, rồi **xóa `retrieved.json` cũ** trước khi chạy lại (cache theo file này).

---

## 2. Thư viện / Framework / Phụ thuộc

Phụ thuộc tách làm 4 nhóm: **(2.1) entrypoint thi (root)**, **(2.2) local-dev thi**, **(2.3) cài thêm trên Kaggle**, **(2.4) web-app (không dự thi)**.

### 2.1. Entrypoint tái hiện pipeline thi — `requirements.txt` (gốc repo)

File `requirements.txt` ở **gốc repo** là entrypoint chuẩn để dựng môi trường tái hiện (`pip install -r requirements.txt`). Header file ghi rõ môi trường chuẩn = Kaggle GPU T4×2, Internet ON, Python 3.10/3.11.

| Thư viện | Version (pin tối thiểu) | Mục đích |
|---|---|---|
| `sentence-transformers` | `>=3.0.0` | Tải/encode embedding (bi-encoder BGE-M3) |
| `transformers` | `>=4.44.0` | Reranker (`XLMRobertaForSequenceClassification`) + LLM Qwen fp16 |
| `torch` | `>=2.2.0` | Backend tensor (CUDA trên Kaggle, MPS/CPU local) |
| `accelerate` | `>=0.30.0` | `device_map=auto` trải Qwen 7B lên 2×T4 |
| `rank-bm25` | `>=0.2.2` | Nhánh sparse BM25 trong hybrid retrieval |
| `numpy` | `>=1.24` | Vector store brute-force cosine (`corpus_emb.npy`) — corpus nhỏ, không cần ChromaDB/FAISS |
| `datasets` | `>=2.18.0` | Tải corpus HF `tmquan/vbpl-vn` để dựng corpus |
| `pyvi` | `>=0.1.1` *(optional)* | Tách từ tiếng Việt cho BM25 (pipeline vẫn chạy nếu thiếu) |

> Ghi chú trong file: **fine-tune (TÙY CHỌN)** cần cài thêm `"FlagEmbedding>=1.3" deepspeed` (xem 2.3). **LLM Qwen** local chạy qua **Ollama** (`ollama pull qwen2.5:7b-instruct-q4_K_M`), trên Kaggle chạy qua `transformers` (đã có ở trên). `backend/requirements.txt` là cho **bản web-app** — KHÔNG dùng để tái hiện.

### 2.2. Deps pipeline thi cho LOCAL dev — `backend/requirements-local.txt`

Dùng khi dev/đo cục bộ (Mac/Linux). Là tập con của (2.1), không có `transformers`/`accelerate` (LLM local chạy qua Ollama).

| Thư viện | Version (pin tối thiểu) | Mục đích |
|---|---|---|
| `sentence-transformers` | `>=3.0.0` | Tải/encode embedding + cross-encoder |
| `torch` | `>=2.2.0` | Backend tensor (MPS/CPU local) |
| `numpy` | `>=1.24` | Cosine brute-force (`corpus_emb.npy`) |
| `rank-bm25` | `>=0.2.2` | Nhánh sparse BM25 |
| `datasets` | `>=2.18.0` | Tải corpus HF `tmquan/vbpl-vn` |
| `pyvi` | `>=0.1.1` *(optional)* | Tách từ tiếng Việt cho BM25 |

> LLM **Qwen2.5-7B-Instruct** KHÔNG cài qua pip ở file này: local chạy qua **Ollama** (`ollama pull qwen2.5:7b-instruct-q4_K_M`). (Comment cũ trong file ghi corpus_emb "~15MB" là số liệu cũ thời corpus nhỏ; bản 93K hiện tại ~381MB — xem mục 3.)

### 2.3. Cài thêm trên Kaggle (luồng chính) — trong notebook (Cell 2)

```bash
pip install -q "sentence-transformers>=3.0" "transformers>=4.44" accelerate rank_bm25 datasets
```

| Thư viện | Version | Mục đích |
|---|---|---|
| `sentence-transformers` | `>=3.0` | Embedding (bi-encoder BGE-M3) |
| `transformers` | `>=4.44` | Reranker (`XLMRobertaForSequenceClassification`) + LLM Qwen fp16 |
| `accelerate` | (mới nhất) | `device_map=auto` trải Qwen 7B lên 2×T4 |
| `rank_bm25` | (mới nhất) | Nhánh sparse BM25 |
| `datasets` | (mới nhất) | Tải corpus HF (chỉ khi Phase 0 phải dựng lại corpus) |

**Fine-tune (TÙY CHỌN) cài thêm — trong `kaggle/finetune_reranker_kaggle.ipynb`:**

```bash
pip install "FlagEmbedding>=1.3" deepspeed
```

| Thư viện | Version | Mục đích |
|---|---|---|
| `FlagEmbedding` | `>=1.3` | Trainer fine-tune reranker BGE family (entrypoint `FlagEmbedding.finetune.reranker.encoder_only.base`) |
| `deepspeed` | (mới nhất) | Tối ưu bộ nhớ khi train trên 2×T4 (ds_stage0 = DDP) |

### 2.4. Deps web-app — `backend/requirements.txt` (KHÔNG dùng cho thi)

| Thư viện | Version | Mục đích |
|---|---|---|
| `fastapi` | `==0.110.0` | API server (sản phẩm web) |
| `uvicorn[standard]` | `>=0.46` | ASGI server |
| `python-multipart` | `==0.0.9` | Upload form |
| `python-dotenv` | `==1.0.1` | Đọc `.env` |
| `pydantic` | `>=2.6.4` | Validation |
| `python-docx` | `==1.1.0` | Sinh văn bản Word |
| `google-antigravity` | `==0.1.0` | Agent SDK bản web cũ (**model ĐÓNG → đã bị loại tư cách**; chỉ giữ cho sản phẩm sau) |
| `protobuf` | `>=7.0` | Phụ thuộc transitive của antigravity (pb2 dùng Edition 2024) |

> Nhóm này KHÔNG được dùng để tạo ra điểm thi và không nằm trên đường tái hiện 0.5766.

### 2.5. Python / CUDA / Torch

| Môi trường | Python | Accelerator | Ghi chú |
|---|---|---|---|
| **Kaggle (chính)** | 3.10 / 3.11 | GPU **T4 ×2** (16GB×2), CUDA của Kaggle | `torch>=2.2`, fp16; phiên ~12h, 30h GPU/tuần |
| Local dev | 3.10+ | CPU / MPS (Mac) | Chỉ để dev; **KHÔNG** chạy model nặng trên Mac (lag) |

`local_models_config.get_device()` tự chọn: CUDA > MPS > CPU.

### 2.6. Model & "phiên bản" (compliance: mã nguồn mở, <14B, công bố trước 2026-03-01)

| Vai trò | Model | Kiến trúc / Kích thước | License / Năm | Link HF |
|---|---|---|---|---|
| Embedding | `AITeamVN/Vietnamese_Embedding` | BGE-M3 fine-tune VN, ~0.6B, 1024-dim | MIT / 2025 | https://huggingface.co/AITeamVN/Vietnamese_Embedding |
| Reranker | `AITeamVN/Vietnamese_Reranker` | bge-reranker-v2-m3 (`XLMRobertaForSequenceClassification`, num_labels=1), ~0.6B | MIT / 2025 | https://huggingface.co/AITeamVN/Vietnamese_Reranker |
| LLM (answer) | `Qwen/Qwen2.5-7B-Instruct` | 7.6B | Apache-2.0 / 2024-09 | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct |

- Chọn Embedding **v1** (không v2) vì v2 yếu hơn ở văn bản pháp luật.
- **Checkpoint dùng cho 0.5766 = 3 model gốc public HF ở trên** — KHÔNG có checkpoint tự train. "Phiên bản" = pin theo **HF revision/commit** khi nộp: `{{HF_REVISION}}` *(ĐIỀN SAU KHI UPLOAD)*.
- **[TÙY CHỌN — cải tiến]** Reranker fine-tune từ `AITeamVN/Vietnamese_Reranker` bằng synthetic data: pipeline **đã build** (`scratch/finetune/`, `kaggle/finetune_reranker_kaggle.ipynb`), checkpoint dự kiến tại `/kaggle/working/ft_reranker`. Đây là **HƯỚNG cải tiến, CHƯA chắc đã train/đo**, KHÔNG phải checkpoint tạo ra 0.5766. Link checkpoint (nếu có): `{{GOOGLE_DRIVE_LINK_CHECKPOINT}}` *(ĐIỀN SAU KHI UPLOAD)*.

---

## 3. Tệp cấu hình cần thiết để triển khai/vận hành

### 3.1. `backend/local_models_config.py` — config tập trung

Đặt model id, allowlist SME, cutoff, RRF weights và độ dài chuỗi ở **một chỗ** (DRY) để build-corpus / retrieval đồng bộ. Các hằng số quan trọng (trích nguyên giá trị trong file):

```python
# ---- Models ----
EMBEDDING_MODEL  = "AITeamVN/Vietnamese_Embedding"
RERANKER_MODEL   = "AITeamVN/Vietnamese_Reranker"
EMBEDDING_DIM    = 1024
EMBED_MAX_SEQ_LEN = 1024   # bi-encoder (embedding)
RERANK_MAX_LEN    = 512    # cross-encoder reranker (query + snippet) — KHỚP giá trị chấm 0.5766

# ---- Retrieval cutoff (default trong config) ----
RETRIEVE_TOP_K     = 3       # default module; cutoff CHẤM 0.5766 dùng t5m3 qua sweep_cutoff.py
RETRIEVE_MIN_SCORE = None    # bỏ ngưỡng tuyệt đối (điểm rerank là logit, thang lệch theo câu)
RETRIEVE_MARGIN    = 1.5     # default module (giữ điều có điểm >= top - margin)
RETRIEVE_CAND_SAVE = 12      # số candidate (kèm score) lưu để sweep cutoff offline

# ---- Hybrid fusion: RRF CÓ TRỌNG SỐ (BM25 nặng hơn) ----
RRF_K       = 60
RRF_W_BM25  = 0.65
RRF_W_DENSE = 0.35

# ---- LLM (local serving) ----
OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_K_M"
OLLAMA_HOST  = "http://localhost:11434"

# ---- Corpus & allowlist SME ----
HF_VBPL = "tmquan/vbpl-vn"          # corpus pháp lý chính (BTC gợi ý)
HF_PHAPDIEN = "tmquan/phapdien-moj-gov-vn"   # codified (thứ cấp)
HF_ANLE     = "tmquan/anle-toaan-gov-vn"     # case law (tùy chọn)
SME_LAW_ALLOWLIST = { "59/2020/QH14", "45/2019/QH14", "38/2019/QH14", ... }  # ~33 mã văn bản SME
KEEP_LEGAL_TYPES  = ("luật", "bộ luật", "nghị định", "thông tư", "pháp lệnh")  # chỉ giữ văn bản QUY PHẠM
```

> **Lưu ý nguồn-của-sự-thật về cutoff:** `RETRIEVE_TOP_K`/`RETRIEVE_MARGIN` trong file này là **default của module** (đường retrieval live). Con số chính thức **0.5766** dùng cutoff **`t5m3` = top-5, margin 3.0** áp **offline** qua `scratch/sweep_cutoff.py` (xem GRID trong file đó: `("t5m3", 5, 3.0, None, True, False, False)`). Cấu hình notebook Phase A độc lập với default này (CAND=80, lưu top-20 + score).

### 3.2. `.gitignore` — loại secrets & artifact nặng khỏi Git

Các nhóm quan trọng (theo đúng pattern trong file):

| Nhóm pattern | Loại trừ | Lý do |
|---|---|---|
| `.env`, `.env.local`, `backend/.env`, `*.key`, `*.pem` | Secrets | **NEVER commit** credentials |
| `data/corpus_articles.jsonl`, `data/corpus_articles.jsonl.done`, `data/corpus_emb.npy`, `data/corpus_emb_ids.json` | Artifact dẫn xuất nặng | Tái sinh được; share qua Drive (corpus_emb.npy ~381MB) |
| `data/zalo_eval.json`, `data/zalo_retrieved.json`, `data/sme_doc_ids_all.json`, `data/sme_doc_ids_existing.json` | Dữ liệu phụ trợ | Lớn / regenerable |
| `results.json`, `submission.zip`, `sweep/`, `backup/` | Output pipeline | Sinh lại mỗi lần chạy |
| `data/chroma_db/` | Vector store cũ | Legacy, không dùng |
| `__pycache__/`, `.venv/`, `venv/`, `backend/venv/`, `node_modules/` | Build/env | Chuẩn |

> Hệ quả: `data/corpus_emb.npy` **không tracked trong Git**. Trong checkout local nó là **symlink ~86 byte** trỏ tới bản backup ~381MB; trên Kaggle nó được nạp qua **Add Input** (xem tài liệu *01 — Mô tả dữ liệu*).

### 3.3. Kaggle Notebook settings (môi trường tái hiện)

| Cài đặt | Giá trị bắt buộc |
|---|---|
| Accelerator | **GPU T4 ×2** (16GB×2) |
| Internet | **ON** (tải model HF + dataset) |
| Python | 3.10 / 3.11 |
| Phiên | ~12h / session, hạn mức 30h GPU/tuần |
| Add Input | Dataset chứa **`corpus_articles.jsonl` (93K)** + **`corpus_emb.npy`** (để Phase 0 dùng corpus HF, KHÔNG build vbpl) |

Tham số runtime trong `kaggle/full_pipeline_kaggle.ipynb` (Cell 10, Phase A) — **trạng thái HIỆN TẠI là probe 1024**:

```python
MAX_SEQ, RERANK_MAX, CAND, CAND_SAVE, RR_BATCH = 1024, 1024, 80, 20, 8
RRF_K, W_BM25, W_DENSE = 60, 0.65, 0.35
# ⚠ Để khớp 0.5766 ĐÃ CHẤM: đặt RERANK_MAX = 512 và RR_BATCH = 16 (512 tok đỡ VRAM hơn 1024),
#   rồi XÓA retrieved.json cũ trước khi chạy lại (Phase A cache theo file này).
```

> **Về `RR_BATCH`:** cấu hình chính thức 0.5766 dùng **RR_BATCH = 16** ở RERANK_MAX 512. Giá trị **8** chỉ là của trạng thái probe 1024 (512 tok ngốn VRAM gấp đôi nên probe phải hạ batch). `RR_BATCH` chỉ ảnh hưởng tốc độ, không ảnh hưởng điểm.

### 3.4. Bảng tổng hợp: file cấu hình → vai trò

| File cấu hình | Phạm vi | Vai trò |
|---|---|---|
| `requirements.txt` (gốc repo) | `[THI]` | **Entrypoint deps tái hiện** (sentence-transformers, transformers, torch, accelerate, rank-bm25, numpy, datasets, pyvi) |
| `backend/local_models_config.py` | `[THI]` | Nguồn-của-sự-thật: model id, RERANK_MAX_LEN, RETRIEVE_TOP_K/MARGIN/CAND_SAVE, RRF weights, allowlist SME, KEEP_LEGAL_TYPES, HF_VBPL |
| `backend/requirements-local.txt` | `[THI]` | Deps pipeline thi cho LOCAL dev (tập con của root, LLM qua Ollama) |
| `kaggle/full_pipeline_kaggle.ipynb` (Cell 2 pip + Cell 10 tham số) | `[THI]` | Install Kaggle + tham số runtime (RERANK_MAX, CAND, RRF) |
| `scratch/sweep_cutoff.py` (GRID) | `[THI]` | Định nghĩa các cấu hình cutoff (t5m3 = chính thức) |
| `.gitignore` | `[CHUNG]` | Bảo vệ secrets + loại artifact nặng |
| `backend/.env.example` | `[WEB]` | Mẫu biến môi trường web-app (Ollama host, v.v.) |
| `backend/requirements.txt` | `[WEB]` | Deps web-app FastAPI (KHÔNG dự thi) |
| Kaggle Notebook settings (UI) | `[THI]` | GPU T4×2 + Internet ON + Python 3.10/3.11 + Add Input corpus HF 93K |

---

## Phụ lục — Trung thực (bắt buộc nêu)

- **0.5766** đạt bằng reranker **OFF-THE-SHELF** (`AITeamVN/Vietnamese_Reranker`) @ **RERANK_MAX 512** trên **corpus HF 93K** — **KHÔNG dùng checkpoint tự train**.
- **Notebook ở trạng thái probe:** chạy y nguyên (RERANK_MAX 1024, Phase 0 fallback build vbpl) sẽ KHÔNG ra 0.5766. Phải (1) Add Input corpus HF 93K + emb, (2) đặt RERANK_MAX 512 / RR_BATCH 16, (3) áp cutoff t5m3 — xem mục 1.3 và 3.3.
- Fine-tune reranker là **hướng cải tiến đã build** — ghi là **TÙY CHỌN**, chưa chắc đã train/đo.
- `data/gold_dev.json` là **gold SYNTHETIC** (nhóm tự suy, 20 câu) → chỉ so sánh tương đối, KHÔNG phải gold thật.
- `backend/vbpl_fetch.py` + `scratch/rebuild_corpus_vbpl.py` là đường **đã REVERT** (corpus vbpl đo ~0.45/0.51 < 0.5766) — giữ làm bằng chứng quyết định kỹ thuật.
- Mọi link Drive/checkpoint là **placeholder** — phải ĐIỀN SAU KHI UPLOAD.

## Câu hỏi chưa giải quyết

1. HF revision/commit cụ thể của 3 model lúc nộp (`{{HF_REVISION}}`) — cần pin chính xác để tái hiện 100%.
2. Link Drive cho `corpus_articles.jsonl` + `corpus_emb.npy` (~381MB) và checkpoint fine-tune (nếu train) — chưa upload.
