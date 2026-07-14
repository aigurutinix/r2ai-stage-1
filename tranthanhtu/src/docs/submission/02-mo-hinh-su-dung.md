# Mô hình sử dụng — AI Legal Assistant (R2AI 2026)

Tài liệu này mô tả các mô hình AI dùng trong pipeline tạo ra kết quả nộp chính thức
**ARTICLES_F2_MACRO = 0.5766**, cách pin phiên bản (checkpoint), hướng dẫn tải/sử dụng,
và link chia sẻ.

> **Trung thực (quan trọng):** Điểm 0.5766 đạt được bằng reranker **OFF-THE-SHELF**
> (`AITeamVN/Vietnamese_Reranker`) ở cấu hình **RERANK_MAX = 512**, trên corpus HF ~93K Điều.
> **KHÔNG có checkpoint tự huấn luyện.** Cả 3 mô hình đều là model gốc public trên
> HuggingFace, nạp off-the-shelf.

---

## 1. Thông tin mô hình

Pipeline dùng **3 mô hình** ở 3 vai trò tách biệt: **embedding** (truy hồi dense),
**reranker** (cross-encoder xếp lại ứng viên), và **LLM** (sinh câu trả lời QA).

| # | Vai trò | Tên trên HuggingFace | Kiến trúc | #Params | License | Công bố | Tuân thủ (<14B, pre-2026-03-01) |
|---|---------|----------------------|-----------|---------|---------|---------|----------------------------------|
| 1 | **Embedding** (bi-encoder, dense retrieval) | `AITeamVN/Vietnamese_Embedding` | BGE-M3 fine-tune tiếng Việt; 1024-dim | ~0.6B | MIT | 2025 | ✅ <14B, ✅ trước 2026-03 |
| 2 | **Reranker** (cross-encoder) | `AITeamVN/Vietnamese_Reranker` | bge-reranker-v2-m3 family — `XLMRobertaForSequenceClassification`, `num_labels=1` | ~0.6B | MIT | 2025 | ✅ <14B, ✅ trước 2026-03 |
| 3 | **LLM** (sinh answer QA — Phase B) | `Qwen/Qwen2.5-7B-Instruct` | Qwen2.5 decoder-only | 7.6B | Apache-2.0 | 2024-09 | ✅ <14B, ✅ trước 2026-03 |

**Vai trò trong pipeline (tạo ra 0.5766):**

1. **Embedding** — mã hoá câu hỏi và toàn bộ corpus thành vector 1024-dim. Truy hồi dense
   bằng cosine (numpy brute-force), `max_seq_length = 1024`. Đây là **nhánh DENSE** của hybrid.
2. **Reranker** — sau khi RRF có trọng số (BM25 ⊕ Dense) chọn **CAND = 80** ứng viên,
   cross-encoder chấm điểm từng cặp (câu hỏi, Điều) và xếp lại, lấy **top-20** kèm điểm rerank.
   Đây là **lever quan trọng nhất** cho điểm số (xem §2).
3. **LLM (Qwen2.5-7B)** — **Phase B**, sinh `answer` ~200–350 từ và ép trích "Điều X".
   **KHÔNG ảnh hưởng ARTICLES_F2_MACRO** (leaderboard hằng ngày chỉ chấm IR trên
   `relevant_articles`); chỉ phục vụ tiêu chí QA khi bài được promote.

> Tham chiếu cấu hình: `backend/local_models_config.py` (`EMBEDDING_MODEL`, `RERANKER_MODEL`,
> `EMBEDDING_DIM=1024`, `EMBED_MAX_SEQ_LEN=1024`, `RERANK_MAX_LEN=512`).

---

## 2. Phiên bản checkpoint

### 2.1. "Checkpoint" trong submission này = mô hình gốc public trên HuggingFace

Cấu hình tạo ra **0.5766 KHÔNG dùng trọng số tự huấn luyện**. Cả 3 mô hình được nạp
**off-the-shelf** từ HuggingFace Hub (xem bảng §1). Vì vậy:

- **Checkpoint = 3 model gốc public HuggingFace** ở bảng §1 (link HF công khai bên dưới);
  **KHÔNG có checkpoint tự train**, **không có file `.safetensors`/`.bin` riêng** cần chia sẻ.
- **"Phiên bản"** được cố định bằng cách **pin HF revision/commit** tại thời điểm nộp.

### 2.2. Cách pin HF revision/commit

Để tái hiện chính xác, pin từng repo theo commit SHA khi finalize (ghi vào working-notes):

```text
AITeamVN/Vietnamese_Embedding @ bản mới nhất công bố trên HuggingFace (ghim theo thời điểm nộp)
AITeamVN/Vietnamese_Reranker  @ bản mới nhất công bố trên HuggingFace (ghim theo thời điểm nộp)
Qwen/Qwen2.5-7B-Instruct      @ bản mới nhất công bố trên HuggingFace (ghim theo thời điểm nộp)
```

Trong code, truyền `revision=` vào loader để khoá phiên bản:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# REV = commit SHA — dùng bản mới nhất công bố trên HuggingFace (ghim theo thời điểm nộp)
REV = "main"
tok   = AutoTokenizer.from_pretrained("AITeamVN/Vietnamese_Reranker", revision=REV)
model = AutoModelForSequenceClassification.from_pretrained("AITeamVN/Vietnamese_Reranker", revision=REV)
```

> Lấy commit SHA hiện hành: mở trang HF của model → tab **"Files and versions"** → copy
> commit ở nhánh `main`; hoặc `huggingface-cli` / `HfApi().model_info(repo_id).sha`.

### 2.3. Cấu hình chính thức của điểm 0.5766

| Tham số | Giá trị (đã chấm 0.5766) | Ghi chú |
|---------|--------------------------|---------|
| Reranker | `AITeamVN/Vietnamese_Reranker` (off-the-shelf) | KHÔNG fine-tune |
| `RERANK_MAX` | **512** | Cấu hình **chính thức** của 0.5766 (256→512 là cú thắng 0.5608→0.5766) |
| `CAND` (ứng viên vào rerank) | 80 | từ RRF có trọng số (`RRF_K=60`, `W_BM25=0.65`, `W_DENSE=0.35`) |
| `RR_BATCH` (batch rerank) | 16 | ở 512 tok; ở 1024 probe phải hạ về **8** (VRAM gấp đôi) |
| Lưu top sau rerank (`CAND_SAVE`) | 20 | kèm điểm rerank vào `retrieved.json` |
| Cutoff | `t5m3` (top-5, margin 3.0) + `drop_superseded` | bước RẺ, offline (qua `scratch/sweep_cutoff.py`) |

> ⚠️ **Hai điều chỉnh BẮT BUỘC để tái hiện đúng 0.5766** (notebook hiện ở trạng thái *probe*):
>
> 1. **`RERANK_MAX`:** notebook đang để **1024** (probe đọc dài hơn để thử cải thiện thêm).
>    Đặt lại **`RERANK_MAX = 512`** (và `RR_BATCH = 16`) trong cell cấu hình Phase A
>    (**cell 10**), rồi xoá `retrieved.json` cũ trước khi chạy lại.
> 2. **Cutoff `t5m3`:** Phase B trong notebook đặt mặc định `CUT_TOP_K, CUT_MARGIN = 3, 1.5`
>    (cell 14) — đây là cutoff áp **trực tiếp trong notebook**. Giá trị **đã chấm 0.5766**
>    là **`t5m3` (top-5, margin 3.0)**, được tạo bằng **sweep offline**: tải `retrieved.json`
>    về local → chạy `python scratch/sweep_cutoff.py` (biến thể `("t5m3", 5, 3.0, …)` có sẵn
>    trong lưới sweep) → sinh `submission_t5m3.zip`. Vì cutoff áp trên `retrieved.json` (đã
>    lưu điểm rerank), **đổi cutoff KHÔNG cần chạy lại GPU**.

---

## 3. Hướng dẫn tải & dùng checkpoint

### 3.1. Cài dependencies

Trên **Kaggle** (luồng chính tạo 0.5766, 2×T4):

```bash
pip install "sentence-transformers>=3.0" "transformers>=4.44" accelerate rank_bm25 datasets
```

Local dev (`backend/requirements-local.txt`):

```text
sentence-transformers>=3.0.0
torch>=2.2.0
numpy>=1.24
rank-bm25>=0.2.2
datasets>=2.18.0
pyvi>=0.1.1          # optional — cải thiện BM25
```

### 3.2. Nạp Embedding (sentence-transformers)

```python
from sentence_transformers import SentenceTransformer

emb = SentenceTransformer("AITeamVN/Vietnamese_Embedding", device="cuda")
emb.max_seq_length = 1024          # legal articles fit < 1024 tok; tránh O(n²) chậm
vecs = emb.encode(["Điều kiện thành lập doanh nghiệp?"],
                  normalize_embeddings=True)   # cosine = dot trên vector đã chuẩn hoá
```

### 3.3. Nạp Reranker (transformers — `AutoModelForSequenceClassification`)

> **Bắt buộc** dùng `AutoModelForSequenceClassification`, **KHÔNG** dùng `CrossEncoder` của
> sentence-transformers 5.x (ST 5.x route qua `AutoProcessor` và lỗi với model text-only này).
> Tham chiếu: `backend/local_reranker.py`.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RERANK_ID, RERANK_MAX, RR_BATCH = "AITeamVN/Vietnamese_Reranker", 512, 16  # 512/16 = cấu hình 0.5766
tok   = AutoTokenizer.from_pretrained(RERANK_ID)
model = AutoModelForSequenceClassification.from_pretrained(RERANK_ID).to("cuda").eval()

@torch.no_grad()
def rerank(query, docs):
    pairs = [[query, f"{d['title']} {d['text']}"] for d in docs]
    scores = []
    for i in range(0, len(pairs), RR_BATCH):
        inp = tok(pairs[i:i+RR_BATCH], padding=True, truncation=True,
                  max_length=RERANK_MAX, return_tensors="pt").to("cuda")
        scores += model(**inp).logits.view(-1).float().cpu().tolist()   # logit = điểm liên quan
    return sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
```

### 3.4. Nạp LLM (transformers, fp16, trải 2×T4)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LLM_ID = "Qwen/Qwen2.5-7B-Instruct"      # 7.6B, Apache-2.0
ltok = AutoTokenizer.from_pretrained(LLM_ID)
ltok.padding_side = "left"               # bắt buộc cho batched generate
if ltok.pad_token is None:
    ltok.pad_token = ltok.eos_token
# fp16 KHÔNG quant — device_map='auto' tự trải ~15GB lên 2×T4 (no OOM). Cần Accelerator = GPU T4 x2.
llm = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=torch.float16,
                                           device_map="auto", low_cpu_mem_usage=True).eval()
```

### 3.5. Cache HuggingFace trên Kaggle

- Bật **Internet → On** trong Settings để loader tải model lần đầu.
- HF cache mặc định: `~/.cache/huggingface` (hoặc đặt `HF_HOME=/kaggle/working/hf_cache`
  để cache nằm trong working dir, persist trong phiên ~12h).
- Tải về local (tuỳ chọn) để pin & tái dùng nhanh:

```python
from huggingface_hub import snapshot_download
# revision="main" = bản mới nhất công bố trên HuggingFace (ghim theo thời điểm nộp)
snapshot_download("AITeamVN/Vietnamese_Reranker", revision="main",
                  local_dir="/kaggle/working/reranker")
```

- **Embeddings của corpus** (`corpus_emb.npy`, ~381MB) được **tái dùng** giữa các vòng để
  khỏi encode lại — chỉ re-rerank khi đổi `RERANK_MAX` (xoá `retrieved.json` cũ trước).

---

## 4. Link chia sẻ checkpoint

### 4.1. Mô hình gốc (đủ để tái hiện 0.5766) — link HF công khai

| Mô hình | Link HuggingFace (truy cập công khai) |
|---------|----------------------------------------|
| Embedding | https://huggingface.co/AITeamVN/Vietnamese_Embedding |
| Reranker | https://huggingface.co/AITeamVN/Vietnamese_Reranker |
| LLM | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct |

> 3 link trên là **toàn bộ checkpoint cần thiết** cho điểm 0.5766. **KHÔNG có checkpoint
> tự train.** Không cần Drive cho phần model.

---

## Phụ lục — Tuân thủ quy định mô hình

- Tất cả 3 mô hình: **mã nguồn mở, <14B params, công bố trước 2026-03-01**. ✅
- **Cấm model đóng** (GPT/Gemini) — pipeline thi **không** dùng. (Bản web-app cũ từng dùng
  model đóng đã bị loại tư cách; chỉ giữ cho sản phẩm sau, **không** dùng cho phần thi.)
