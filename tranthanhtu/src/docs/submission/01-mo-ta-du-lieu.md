# Mô tả dữ liệu — AI Legal Assistant (R2AI 2026)

Tài liệu này mô tả toàn bộ dữ liệu dùng trong hệ thống **Truy hồi & Hỏi đáp văn bản pháp luật tiếng Việt cho SME** dự thi ROAD TO AI 2026 (R2AI). Nội dung gồm 4 mục bắt buộc theo yêu cầu BTC: (1) nguồn dữ liệu, (2) cấu trúc + định dạng, (3) hướng dẫn truy cập/sử dụng, (4) link chia sẻ Drive. Bổ sung một mục hướng dẫn **tái tạo corpus từ HuggingFace** từ đầu.

> Kết quả chính thức đã nộp hiện tại: **ARTICLES_F2_MACRO = 0.5766** (corpus HF ~93K Điều, reranker off-the-shelf @ RERANK_MAX=512). Mọi dữ liệu mô tả dưới đây gắn với pipeline tái hiện con số này (`kaggle/full_pipeline_kaggle.ipynb`).

---

## 1. Mô tả nguồn dữ liệu

Hệ thống dùng hai nhóm nguồn, **phân biệt rõ cái nào BTC cấp và cái nào nhóm tự thu thập**:

| Nguồn | Ai cung cấp | Mô tả | File tương ứng |
|---|---|---|---|
| **Test set 2000 câu** | **BTC cấp** | Bộ câu hỏi đề thi vòng 1, mỗi dòng `{id, question}`. Đây là input bắt buộc để sinh `results.json` nộp lên leaderboard. | `data/stage1_questions.json` |
| **Corpus văn bản pháp luật** | **Nhóm tự thu thập** | Xây từ HF dataset `tmquan/vbpl-vn` (158K văn bản pháp luật crawl từ cổng `vbpl.vn`), **lọc theo phạm vi SME** (doanh nghiệp nhỏ và vừa) rồi tách thành đơn vị **Điều**. BTC **KHÔNG** cấp corpus hay tập train. | `data/corpus_articles.jsonl`, `data/corpus_emb.npy` |

**Lưu ý nguồn:**
- BTC chỉ cấp **đề thi** (câu hỏi); việc thu thập, làm sạch và đánh chỉ mục văn bản pháp luật là do nhóm tự thực hiện.
- HF dataset `tmquan/vbpl-vn` là nguồn được BTC **gợi ý** cho phần corpus statutory; nhóm chọn nguồn này vì độ phủ và cấu trúc markdown ổn định.
- Đã thử nguồn API `vbpl.vn` chính thống (`backend/vbpl_fetch.py`) nhưng **đã revert** vì corpus dựng từ API đo tệ hơn (ARTICLES_F2 ~0.45–0.51 < 0.5766).

---

## 2. Cấu trúc và định dạng dữ liệu

### 2.1. Bảng tổng hợp file dữ liệu

Phân làm 3 loại: **dẫn xuất** (cần share để tái hiện), **đề thi** (BTC cấp), **phụ trợ** (eval/phân tích nội bộ, không bắt buộc share).

| File | Kích thước | Loại | Vai trò |
|---|---|---|---|
| `data/corpus_articles.jsonl` | ~91 MB | Dẫn xuất | Corpus Điều-level — nguồn truy hồi chính, chứa khóa chấm điểm `(doc_number, article)`. Bản tái hiện 0.5766 ≈ **93K dòng** (Kaggle/Drive); bản trong repo git hiện là build cục bộ nhỏ hơn (~41K dòng) — xem lưu ý dưới |
| `data/corpus_emb.npy` | ~381 MB | Dẫn xuất | BGE-M3 embeddings của corpus, dùng cho dense retrieval. **Trong repo git chỉ là symlink 86 byte** (trỏ tới `backup/…`); bản thật (~381 MB) ở Kaggle/Drive |
| `data/corpus_emb_ids.json` | nhỏ | Dẫn xuất | Thứ tự `id` trong `.npy` để kiểm tra incremental embed (sinh bởi `embed_corpus.py`) |
| `data/stage1_questions.json` | ~0.5 MB (2000 câu) | Đề thi (BTC cấp) | Input câu hỏi để sinh kết quả nộp |
| `results.json` | — | Đầu ra nộp | File kết quả nộp leaderboard (xem 2.5) |
| `data/zalo_eval.json` | ~0.7 MB (3196 câu) | Phụ trợ | Zalo Legal có gold `"mã\|Điều"` — proxy eval offline |
| `data/gold_dev.json` | ~3 KB (20 câu) | Phụ trợ | Gold **SYNTHETIC** do nhóm tự suy — chỉ so sánh tương đối, **KHÔNG phải gold thật** |
| `data/sme_doc_ids_all.json` | ~226 KB (8020 mục) | Phụ trợ | Map `docNum → vbpl id` phục vụ build corpus |
| `data/law.json`, `data/document_registry.json`, `data/legal_list_id.json` | nhỏ | Phụ trợ | Thư viện của ứng dụng web — **KHÔNG dùng cho phần thi** |

> **Phân biệt dẫn xuất vs phụ trợ:** Dữ liệu *dẫn xuất* (`corpus_articles.jsonl`, `corpus_emb.npy`, `corpus_emb_ids.json`) là sản phẩm xử lý từ nguồn HF và **bắt buộc share** để tái hiện 0.5766. Dữ liệu *phụ trợ* phục vụ eval/phân tích nội bộ hoặc app, **không bắt buộc share**.

> ⚠️ **Lưu ý số dòng corpus (quan trọng cho tái hiện):** Con số chính thức **0.5766** đo trên corpus HF **~93K Điều** (bản nằm trên Kaggle/Drive). Bản `data/corpus_articles.jsonl` *trong repo git hiện tại là một build cục bộ nhỏ hơn (~41K dòng)* — đủ để dev/kiểm thử nhưng **KHÔNG khớp 1-1 con số 0.5766**. Để tái hiện đúng điểm, dùng bundle Drive (Mục 4) hoặc Kaggle Dataset (Mục 3.2), **không** dùng file repo.

### 2.2. `corpus_articles.jsonl` — schema (1 Điều / 1 dòng JSON)

Mỗi dòng là một JSON object, **granularity = 1 Điều/dòng**. Cặp `(doc_number, article)` chính là **khóa chấm điểm** của metric ARTICLES_F2.

| Field | Kiểu | Ý nghĩa | Ví dụ |
|---|---|---|---|
| `id` | string | Khóa nội bộ duy nhất (`<code_slug>_<article>`) | `"182016NCP_Điều1"` |
| `doc_number` | string | Mã văn bản (số hiệu) | `"18/2016/NĐ-CP"` |
| `clean_name` | string | Tên văn bản đã chuẩn hóa | `"Nghị định Sửa đổi, bổ sung..."` |
| `legal_type` | string | Loại văn bản | `"Nghị định"` |
| `year` | string | Năm ban hành | `"2016"` |
| `article` | string | Số Điều | `"Điều 1"` |
| `title` | string | Tiêu đề của Điều | `"Sửa đổi, bổ sung một số điều..."` |
| `text` | string | Nội dung Điều (cắt tối đa 4000 ký tự) | `"1. Sửa đổi, bổ sung Điểm i..."` |
| `source_url` | string | URL gốc của văn bản (gateway API vbpl.vn / cổng pháp điển) | `"https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/100048"` |

Ví dụ một dòng (rút gọn `text`):

```json
{"id": "182016NCP_Điều1", "doc_number": "18/2016/NĐ-CP", "clean_name": "Nghị định Sửa đổi, bổ sung một số điều của Nghị định số 53/2013/NĐ-CP...", "legal_type": "Nghị định", "year": "2016", "article": "Điều 1", "title": "Sửa đổi, bổ sung một số điều của Nghị định số 53/2013/NĐ-CP...", "text": "1. Sửa đổi, bổ sung Điểm i và Điểm l của Khoản 1 Điều 13 như sau: ...", "source_url": "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/100048"}
```

### 2.3. `corpus_emb.npy` — embeddings

| Thuộc tính | Giá trị |
|---|---|
| Định dạng | NumPy `.npy` |
| Shape | `N × 1024` (N = số dòng corpus ≈ 93K) |
| Dtype | `float32` |
| Chuẩn hóa | L2-normalized (cosine = dot product) |
| Model sinh | `AITeamVN/Vietnamese_Embedding` (BGE-M3), `max_seq_length = 1024` |
| **Row alignment** | **Hàng `i` của `.npy` ↔ dòng `i` của `corpus_articles.jsonl`** (cùng thứ tự, 1-1). Thứ tự `id` được lưu thêm ở `corpus_emb_ids.json` để kiểm tra incremental. |

> Ràng buộc alignment là **bất biến quan trọng**: bất kỳ thao tác nào reorder/thêm/bớt dòng `corpus_articles.jsonl` đều phải embed lại tương ứng, nếu không kết quả truy hồi sẽ sai lệch. `embed_corpus.py` tự kiểm tra prefix `id` để chỉ embed phần đuôi mới (incremental) và cảnh báo nếu phát hiện lệch.

### 2.4. `stage1_questions.json` — đề thi (2000 câu)

Mảng JSON, mỗi phần tử `{id, question}`:

```json
[
  { "id": 1, "question": "Các cơ sở ươm tạo và khu làm việc chung được hưởng những chính sách hỗ trợ nào về thuế và đất đai?" },
  { "id": 2, "question": "Doanh nghiệp nhỏ và vừa được hưởng ưu đãi gì khi tham gia đấu thầu?" }
]
```

### 2.5. `results.json` — định dạng nộp

UTF-8, là **list** các object, mỗi object một câu hỏi. Nén thành **zip phẳng** (file `results.json` nằm ở gốc zip). **Tên file bắt buộc** là `results.json`.

```json
[
  {
    "id": 1,
    "question": "Các cơ sở ươm tạo và khu làm việc chung được hưởng...",
    "answer": "Theo Điều 12 ... (đoạn trả lời ~200-350 từ, có trích 'Điều X')",
    "relevant_docs": ["38/2018/NĐ-CP|Nghị định ..."],
    "relevant_articles": ["38/2018/NĐ-CP|Nghị định ...|Điều 12"]
  }
]
```

| Field | Kiểu | Ghi chú |
|---|---|---|
| `id` | int | Khớp với `stage1_questions.json` |
| `question` | string | Câu hỏi gốc |
| `answer` | string | Câu trả lời QA (sinh bởi Qwen2.5-7B). **KHÔNG ảnh hưởng metric ARTICLES_F2** — daily leaderboard chỉ chấm IR |
| `relevant_docs` | string[] | Mỗi phần tử `"mã\|tên"` (mã văn bản) |
| `relevant_articles` | string[] | Mỗi phần tử `"mã\|tên\|Điều X"` — **đây là trường được chấm ARTICLES_F2** |

> Metric xếp hạng = **ARTICLES_F2_MACRO** (recall-weighted, F2 = 5PR/(4P+R)), khớp **chính xác** trên `(mã văn bản, số Điều)`. DOCS_F2 là phụ. QA (4 tiêu chí) chỉ chấm thủ công/LLM-judge trên bài được promote hằng tuần — hiện 0.0.

---

## 3. Hướng dẫn truy cập / sử dụng dữ liệu

Có 3 cách lấy dữ liệu dẫn xuất; chọn cách phù hợp môi trường.

### 3.1. Tải từ Google Drive rồi đặt vào `data/` (local hoặc Kaggle)

1. Tải bundle từ Drive: `{{GOOGLE_DRIVE_LINK_DATA}}` *(ĐIỀN SAU KHI UPLOAD)*.
2. Giải nén và đặt các file vào thư mục `data/` ở gốc repo:
   ```
   data/corpus_articles.jsonl
   data/corpus_emb.npy
   data/corpus_emb_ids.json
   data/stage1_questions.json
   ```
3. Lưu ý: `data/corpus_emb.npy` trong repo git chỉ là **symlink 86 byte** (trỏ tới thư mục `backup/` cục bộ); phải thay bằng bản thật (~381 MB) từ Drive. Tương tự, `data/corpus_articles.jsonl` trong repo là build cục bộ ~41K dòng — dùng bản ~93K từ Drive để khớp 0.5766.

### 3.2. Add Input trên Kaggle (luồng chính — `full_pipeline_kaggle.ipynb`)

1. Upload bundle Drive thành một **Kaggle Dataset** (hoặc dùng dataset đã chuẩn bị sẵn).
2. Trong notebook `kaggle/full_pipeline_kaggle.ipynb`: **Add Input** → chọn dataset corpus.
3. Notebook trỏ đường dẫn đọc tới input đó (corpus + embeddings + questions).
4. Bật **Accelerator = GPU T4 ×2**, **Internet = ON**, chạy 22 cell → sinh `retrieved.json`, sau đó cutoff → `results.json` + `submission.zip`.

> Môi trường tái hiện: Kaggle Notebook, GPU T4 ×2 (16GB×2), Python 3.10/3.11, transformers fp16, phiên ~12h, hạn mức 30h GPU/tuần. **Không** chạy model nặng trên máy Mac (lag); local chỉ để dev.

### 3.3. Build lại corpus từ HF (không cần Drive) — xem Mục 5

Nếu muốn dựng corpus từ đầu thay vì tải Drive, chạy `build_corpus.py` + `embed_corpus.py` (Mục 5). Cần Internet để stream HF dataset.

### 3.4. Cài dependencies

Pipeline thi (`backend/requirements-local.txt`):

```bash
pip install -r backend/requirements-local.txt
# sentence-transformers>=3.0.0, torch>=2.2.0, numpy>=1.24,
# rank-bm25>=0.2.2, datasets>=2.18.0, pyvi>=0.1.1 (optional)
```

Trên Kaggle (luồng chính):

```bash
pip install "sentence-transformers>=3.0" "transformers>=4.44" accelerate rank_bm25 datasets
```

---

## 4. Link chia sẻ Google Drive

> **CHƯA điền** — placeholder, sẽ cập nhật sau khi upload. **KHÔNG bịa link.**

**Link bundle dữ liệu:** `{{GOOGLE_DRIVE_LINK_DATA}}` *(ĐIỀN SAU KHI UPLOAD)*

**Link checkpoint (tùy chọn — reranker fine-tune):** `{{GOOGLE_DRIVE_LINK_CHECKPOINT}}` *(ĐIỀN SAU KHI UPLOAD — chỉ áp dụng cho hướng cải tiến, KHÔNG phải checkpoint tạo ra 0.5766)*

### Nội dung bundle Drive

| File trong bundle | Kích thước | Bắt buộc | Mô tả |
|---|---|---|---|
| `corpus_articles.jsonl` | ~91 MB | Có | Corpus Điều-level (**~93K dòng** — bản tái hiện 0.5766, KHÔNG phải file ~41K trong repo) |
| `corpus_emb.npy` | ~381 MB | Có | BGE-M3 embeddings `N×1024` float32, align theo dòng jsonl |
| `corpus_emb_ids.json` | nhỏ | Có | Thứ tự `id` trong `.npy` |
| `stage1_questions.json` | ~0.5 MB | Có | Đề thi 2000 câu (BTC cấp) |
| `results.json` (best) | — | Tùy chọn | Bản kết quả đã nộp đạt 0.5766 (để đối chiếu) |

> **Model gốc không cần đưa vào Drive** — tải trực tiếp từ HuggingFace (link công khai, là kênh truy cập hợp lệ):
> - Embedding: https://huggingface.co/AITeamVN/Vietnamese_Embedding
> - Reranker: https://huggingface.co/AITeamVN/Vietnamese_Reranker
> - LLM: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
>
> Khi nộp chính thức, **pin "phiên bản" theo HF revision/commit**: `{{HF_REVISION}}` *(ĐIỀN SAU)*.

---

## 5. Tái tạo dữ liệu từ HuggingFace (build từ đầu)

Quy trình dựng lại corpus + embeddings không cần Drive (chạy ở `backend/`, cần Internet để stream HF):

### Bước 1 — `build_corpus.py`: HF `tmquan/vbpl-vn` → `corpus_articles.jsonl`

```bash
cd backend
python build_corpus.py                 # mode allowlist (mặc định, nhanh, chính xác — lọc theo danh sách luật SME)
# python build_corpus.py --mode keywords   # quét rộng hơn theo từ khóa chủ đề SME (recall cao hơn, chậm hơn)
# python build_corpus.py --append          # thêm luật mới vào corpus hiện có, không ghi đè
```

Script stream `tmquan/vbpl-vn` (config `documents`, ~158K văn bản), giữ các văn bản thuộc phạm vi SME, parse markdown từng văn bản thành các Điều bằng `legal_text_parser.py`, ghi mỗi Điều một dòng JSON theo schema ở **Mục 2.2**.

### Bước 2 — `embed_corpus.py`: `corpus_articles.jsonl` → `corpus_emb.npy`

```bash
cd backend
python embed_corpus.py
```

- Sinh embeddings `float32` `N×1024` bằng `AITeamVN/Vietnamese_Embedding` (`max_seq_length=1024`, `normalize_embeddings=True`).
- **Incremental & idempotent**: nếu `corpus_emb.npy` + `corpus_emb_ids.json` đã tồn tại và `id` là prefix sạch của corpus hiện tại → chỉ embed phần Điều mới rồi `vstack` nối vào (giữ row-alignment). Nếu phát hiện lệch thứ tự → embed lại toàn bộ.

> **Lưu ý tái hiện:** corpus build từ HF có thể khác nhẹ về số dòng theo thời điểm stream (dataset/version thay đổi) và theo `--mode`. Bản đo chính thức **0.5766** dùng corpus HF ~93K Điều ở chế độ tương ứng trong `kaggle/full_pipeline_kaggle.ipynb`. Để **tái hiện đúng con số**, nên tải bundle Drive (Mục 3.1/3.2) thay vì build lại từ đầu.

---

## Phụ lục — vị trí dữ liệu trong pipeline 0.5766

```
stage1_questions.json ─┐
                       ├─► Phase A (Dense BGE-M3 ‖ BM25 → RRF có trọng số, CAND=80
corpus_articles.jsonl ─┤              → rerank Vietnamese_Reranker RERANK_MAX=512)
corpus_emb.npy ────────┘              → retrieved.json (top-20 + điểm rerank)
                                          │
                       ┌──────────────────┘
                       ▼
   scratch/sweep_cutoff.py (cutoff "t5m3": top-5, margin 3.0, + drop_superseded)
                       │
                       ▼
              results.json + submission.zip  ──►  leaderboard (ARTICLES_F2 = 0.5766)
```

> Phase B (QA answer bằng Qwen2.5-7B-Instruct fp16 trải 2×T4) chạy độc lập, đổ `answer` vào `results.json`, **không ảnh hưởng** ARTICLES_F2.
