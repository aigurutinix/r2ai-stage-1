# R2AI — Vietnamese Legal QA System

Hệ thống hỏi đáp pháp luật Việt Nam cho doanh nghiệp vừa và nhỏ (SME), gồm hai notebook độc lập chạy tuần tự trên Kaggle:

| Bước | Notebook | Mô tả |
|---|---|---|
| 1 | `pipeline-corpus.ipynb` | Retrieval: tìm điều luật liên quan cho 2000 câu hỏi |
| 2 | `llm-gen.ipynb` | LLM generation: dùng Qwen3-14B sinh câu trả lời + trích dẫn |

---

## Kiến trúc hệ thống

### Bước 1 — Retrieval Pipeline (`pipeline-corpus.ipynb`)

```
Corpus (6,151 điều luật)
    ├── BM25 (underthesea + 4-char n-gram, k1=1.5, b=0.4) → top-100 articles
    ├── Dense BGE-M3 (clause-level encoding) → top-200 clauses → MaxRank → top-50 articles
    └── Query Enrichment (keywords + sub-queries) → BM25 extra + Dense extra
                ↓ RRF fusion (k=60) → pool 50 candidates
    BGE-reranker-v2-m3 Sliding Window MaxP (chunk 1200 chars, overlap 200)
                ↓ Adaptive Soft-Floor cutoff (ABS_FLOOR=0.38, MAX_K=7, MIN_K=1)
    → retrieval_v6.json + submission.zip
```

### Bước 2 — LLM Generation (`llm-gen.ipynb`)

```
Old submissions (10 zip files) → Voting ensemble → Top-10 article candidates per question
    + Corpus lookup (SME merged + th1nhng0 fallback)
                ↓
    Qwen3-14B (4-bit NF4 quantization, transformers)
    System prompt: Legal QA với citation rules
    (Optional) Review pass: LLM-as-a-Judge
                ↓ Checkpoint JSONL sau mỗi câu
    → results_qwen3_14b_llm_filter.json + .zip
```

---

## Yêu cầu môi trường

| Thành phần | Yêu cầu |
|---|---|
| Nền tảng | **Kaggle Notebooks** (khuyến nghị) |
| Runtime Bước 1 | GPU T4 x2 hoặc P100 (≥ 16 GB VRAM tổng) |
| Runtime Bước 2 | GPU T4 x2 hoặc P100/A100 (≥ 16 GB VRAM với 4-bit quant) |
| Python | 3.10+ (Kaggle mặc định) |
| Kết nối internet | **Bật** (để tải model từ HuggingFace) |

---

## Dữ liệu đầu vào

Hai notebook dùng **hai nguồn dữ liệu khác nhau**.

### Bước 1 — Retrieval: Kaggle Dataset

Dataset đã có sẵn trên Kaggle tại: `letuano5/r2ai-corpus`

Notebook dùng đúng 5 file sau (+ 1 tuỳ chọn):

| File | Dung lượng | Mô tả |
|---|---|---|
| `corpus_luat_sme_merged_v3.jsonl` | 14.18 MB | Corpus SME chính (~5,658 điều luật) |
| `patch_laws_articles.jsonl` | 820 kB | 5 luật bổ sung 2024–2025 (273 điều) |
| `sme_clauses_v4.jsonl` | 51.57 MB | Clause chunks của SME corpus (~41,746 chunks) |
| `patch_laws_clauses.jsonl` | 2.89 MB | Clause chunks của patch laws |
| `R2AIStage1DATA.json` | 531 kB | 2,000 câu hỏi test |
| `r2ai_enrichment.jsonl` | 2.39 MB | *(Tuỳ chọn)* Query enrichment cache |

Đường dẫn trong notebook (Cell 3):
```python
DATASET_DIR = "/kaggle/input/datasets/letuano5/r2ai-corpus"
```

### Bước 2 — LLM: Google Drive

Tải dữ liệu từ Google Drive:
**https://drive.google.com/drive/folders/1oxWNBI49W7C5RfEVMiyl8SyxpV1JJhvw?usp=drive_link**

Sau đó upload lên **hai Kaggle Dataset riêng biệt**:

**Dataset 1: `r2ai-corpus`** (dùng chung với Bước 1, thêm file):

| File | Mô tả |
|---|---|
| `corpus_luat_sme_merged_v3.jsonl` | SME corpus (đã có từ Bước 1) |
| `th1nhng0_articles_v2.jsonl` | Fallback article corpus (~517 MB) |

**Dataset 2: `r2ai-old-submissions`** (tạo mới):

| Nội dung | Mô tả |
|---|---|
| 10 file `.zip` submission cũ | Mỗi zip chứa `results.json` với `relevant_articles` |

Đường dẫn trong notebook (cell config):
```python
SUBMISSIONS_DIR = "/kaggle/input/r2ai-old-submissions"
SME_MERGED      = "/kaggle/input/r2ai-corpus/corpus_luat_sme_merged_v3.jsonl"
TH_ARTICLES     = "/kaggle/input/r2ai-corpus/th1nhng0_articles_v2.jsonl"
```

---

## Models (tự động tải từ HuggingFace)

| Model | Dung lượng | Dùng trong |
|---|---|---|
| `BAAI/bge-m3` | ~2.27 GB | Bước 1 — Dense embedder |
| `BAAI/bge-reranker-v2-m3` | ~2.27 GB | Bước 1 — Reranker |
| `Qwen/Qwen3-14B` | ~9 GB (4-bit) / ~28 GB (fp16) | Bước 2 — LLM generation |

Để tránh warning rate limit khi tải model, thêm Kaggle Secret `HF_TOKEN`:
```
Kaggle Notebook → Add-ons → Secrets → New Secret
Key: HF_TOKEN | Value: hf_xxxxxxxxxxxxxxxxxxxx
```

---

## Hướng dẫn chạy

### Bước 1 — `pipeline-corpus.ipynb`

**Thiết lập Kaggle Notebook:**
1. Tạo notebook mới trên Kaggle, upload `pipeline-corpus.ipynb`
2. Settings → Accelerator: chọn `GPU T4 x2` (hoặc P100)
3. Settings → Internet: **On**
4. Add Data → tìm dataset `letuano5/r2ai-corpus` → Add

**Chạy theo thứ tự từng cell:**

| Cell | Hành động | Thời gian ước tính |
|---|---|---|
| Cell 1 | Cài packages: `sentence-transformers`, `rank_bm25`, `underthesea` | ~1 phút |
| Cell 2 | Import libs, kiểm tra GPU | < 30 giây |
| Cell 3 | Load config, kiểm tra file tồn tại (5 dấu ✓) | < 5 giây |
| Cell 4 | Load corpus SME + patch (6,151 điều) + clause chunks (41,746 chunks) | ~1 phút |
| Cell 5 | Build BM25 index *(có cache — nhanh từ lần 2)* | ~5 phút lần đầu |
| Cell 6 | Load BGE-M3 + encode 41,746 clauses *(có cache)* | ~15 phút lần đầu |
| Cell 7 | Encode 2,000 queries | ~2 phút |
| Cell 8 | Load enrichment + encode sub-queries (~3,030 texts) | ~5 phút |
| Cell 9 | BM25 + Dense retrieval → RRF fusion (2,000 queries) | ~3 phút |
| Cell 10 | Load BGE-reranker + Rerank 2,000 queries | ~30–60 phút |
| Cell 11 | Lưu `retrieval_v6.json` + tạo `submission.zip` | < 1 phút |

**Output tại `/kaggle/working/`:**
- `retrieval_v6.json` — kết quả retrieval đầy đủ
- `results.json` — format nộp bài (có trường `answer` rỗng)
- `submission.zip` — file nộp bài

> **Lưu ý cache:** Cell 5 và Cell 6 tự động lưu cache BM25/embedding vào `/kaggle/working/cache/`. Nếu chạy lại notebook (không restart session), hai cell này bỏ qua encode và load từ cache — tiết kiệm ~20 phút.

---

### Bước 2 — `llm-gen.ipynb`

**Thiết lập Kaggle Notebook:**
1. Tạo notebook mới, upload `llm-gen.ipynb`
2. Settings → Accelerator: `GPU T4 x2` hoặc P100/A100
3. Settings → Internet: **On**
4. Add Data → `letuano5/r2ai-corpus` (cần có `corpus_luat_sme_merged_v3.jsonl` + `th1nhng0_articles_v2.jsonl`)
5. Add Data → dataset chứa 10 file zip old submissions (tạo dataset `r2ai-old-submissions`)

**Trước khi chạy — kiểm tra 2 biến quan trọng trong cell config:**
```python
# Đảm bảo đúng tên dataset Kaggle của bạn
SUBMISSIONS_DIR = "/kaggle/input/r2ai-old-submissions"   # chứa 10 file .zip
TH_ARTICLES     = "/kaggle/input/r2ai-corpus/th1nhng0_articles_v2.jsonl"
```

**Chạy theo thứ tự từng cell:**

| Cell | Hành động | Thời gian ước tính |
|---|---|---|
| Cell install | *(Tuỳ chọn)* Đặt `INSTALL_PACKAGES = True` nếu cần cài packages | ~2 phút |
| Cell imports | Import libs, kiểm tra CUDA | < 30 giây |
| Cell config | Thiết lập đường dẫn, hyperparameters, in `REPO_ROOT` và đường dẫn file | < 5 giây |
| Cell prompts | Định nghĩa system prompt + review prompt (chỉ assign biến) | < 5 giây |
| Cell helpers | Định nghĩa các hàm tiện ích | < 5 giây |
| Cell load submissions | Load 10 zip, vote điều luật → `questions` dict | ~1 phút |
| Cell corpus lookup | Build bảng tra cứu full text từ SME corpus + fallback | ~2 phút |
| Cell checkpoint | Load checkpoint nếu đã chạy dở (resume tự động) | < 5 giây |
| Cell prepare | Chuẩn bị messages cho LLM (vote + lookup text) | ~1 phút |
| Cell load model | Load Qwen3-14B với 4-bit NF4 quantization | ~10–15 phút |
| Cell generate | Sinh câu trả lời cho 2,000 câu (checkpoint sau mỗi câu) | ~3–6 giờ |
| Cell export | Xuất `results_qwen3_14b_llm_filter.json` + `.zip` | < 1 phút |
| Cell validate | Kiểm tra format toàn bộ output | < 1 phút |

**Output tại `/kaggle/working/`:**
- `submissions/results_qwen3_14b_llm_filter.json` — kết quả đầy đủ
- `submissions/results_qwen3_14b_llm_filter.zip` — file nộp bài
- `data/eval/qwen3_14b_llm_filter_checkpoint.jsonl` — checkpoint (JSONL append-only)

> **Resume:** Nếu notebook bị ngắt giữa chừng, chạy lại từ đầu — checkpoint tự động bỏ qua các câu đã xử lý. Để bắt đầu lại hoàn toàn, đặt `IGNORE_CHECKPOINT = True` trong cell config.

---

## Cấu hình và tinh chỉnh

### pipeline-corpus.ipynb — Các tham số chính (Cell 3)

```python
# BM25
BM25_K1 = 1.5     # Tăng từ default 1.2 → phù hợp văn bản pháp luật lặp từ có chủ đích
BM25_B  = 0.4     # Giảm từ default 0.75 → ít phạt điều luật dài
BM25_K  = 100     # Top-K bài từ BM25

# Dense retrieval (clause-level)
DENSE_CLAUSE_K = 200   # Số clauses top-K
DENSE_K        = 50    # Số articles sau khi group clauses

# Reranker cutoff
ABS_FLOOR = 0.38  # Ngưỡng điểm tuyệt đối tối thiểu (giảm → recall cao hơn)
MAX_K     = 7     # Tối đa 7 điều luật trả về
MIN_K     = 1     # Tối thiểu 1 điều luật (không ép trả 2 nếu chỉ 1 cái đúng)

# Query enrichment
USE_ENRICHMENT = True   # Dùng r2ai_enrichment.jsonl
USE_KEYWORDS   = True   # Thêm BM25 leg từ keywords
USE_DECOMP     = True   # Thêm BM25 + Dense leg từ sub-queries
USE_HYDE       = False  # Không dùng HyDE (tắt để tránh nhiễu)
```

### llm-gen.ipynb — Các tham số chính (cell config)

```python
MODEL_ID       = "Qwen/Qwen3-14B"  # Có thể đổi sang Qwen3-7B nếu VRAM thấp
LOAD_IN_4BIT   = True               # 4-bit NF4 quant (~9 GB VRAM); False = fp16 (~28 GB)
MAX_ARTICLES   = 10                 # Số điều luật ứng viên đưa vào LLM prompt
MIN_VOTE       = 1                  # Ngưỡng vote tối thiểu từ old submissions
REVIEW_PASS    = False              # True = chạy LLM-as-a-Judge (x2 thời gian)
THINKING       = "off"              # "on" = bật chain-of-thought Qwen3 (chậm hơn)
TEMPERATURE    = 0                  # Greedy decoding (0 = deterministic)
RETRIES        = 1                  # Số lần retry khi LLM lỗi
ARTICLE_CHARS  = 6000               # Cắt full_text mỗi điều luật tối đa 6000 ký tự
MAX_INPUT_TOKENS = 24576            # Context window tối đa
MAX_NEW_TOKENS   = 1600             # Độ dài output tối đa
```

---

## Format output

Mỗi dòng trong `results.json`:

```json
{
  "id": 1,
  "question": "Câu hỏi pháp luật...",
  "answer": "Căn cứ Điều X..., doanh nghiệp cần...",
  "relevant_docs": ["doc_id|Tên văn bản"],
  "relevant_articles": ["doc_id|Tên văn bản|Điều X"]
}
```

Ràng buộc format (được kiểm tra tự động ở cell validate):
- `relevant_articles`: mỗi phần tử có đúng **2 dấu `|`** (3 phần)
- `relevant_docs`: mỗi phần tử có đúng **1 dấu `|`** (2 phần)

---

## Thư viện sử dụng

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| `torch` | ≥ 2.1.0 | GPU inference |
| `transformers` | ≥ 4.51.0 | Load/inference Qwen3-14B |
| `accelerate` | ≥ 0.33.0 | Multi-GPU device_map |
| `bitsandbytes` | ≥ 0.43.0 | 4-bit NF4 quantization |
| `sentence-transformers` | ≥ 2.7.0 | BGE-M3 embedder + BGE-reranker-v2-m3 |
| `rank-bm25` | ≥ 0.2.2 | BM25Okapi sparse retrieval |
| `underthesea` | ≥ 6.8.0 | Vietnamese word tokenizer |
| `numpy` | ≥ 1.24.0 | Vector operations |
| `tqdm` | ≥ 4.66.0 | Progress bars |
| `json`, `zipfile`, `pathlib`, `hashlib`, `pickle` | stdlib | File I/O, caching |

---

## Gỡ lỗi thường gặp

**`✗` file không tìm thấy khi chạy Cell 3 (Bước 1)**
→ Kiểm tra tên dataset Kaggle trong `DATASET_DIR`. Vào tab "Input" của notebook xác nhận đường dẫn thực tế.

**CUDA out of memory khi rerank (Cell 10)**
→ Notebook tự giảm `batch_sz` khi gặp OOM. Nếu vẫn lỗi, sửa `RERANK_BATCH = 16` trong Cell 10.

**Notebook Bước 2 không tìm thấy `th1nhng0_articles_v2.jsonl`**
→ Kiểm tra file đã được upload vào Kaggle dataset chưa. Tên file phải khớp chính xác (có `_v2`).

**Notebook Bước 2 không tìm thấy file zip trong `old_submission`**
→ Kiểm tra `SUBMISSIONS_DIR` trỏ đúng tên dataset Kaggle chứa các file `.zip`.

**Qwen3-14B load chậm hoặc lỗi**
→ Đảm bảo Internet đã bật trong Kaggle. Thêm `HF_TOKEN` vào Kaggle Secrets nếu gặp rate limit.

**Muốn chạy thử nhanh (smoke test)**
→ Trong cell config của `llm-gen.ipynb`, đặt `LIMIT = 5` để chỉ chạy 5 câu đầu tiên.
