# Mô tả dữ liệu — R2AI Stage 1

Tài liệu mô tả nguồn dữ liệu, cấu trúc, định dạng và hướng dẫn truy cập/sử dụng dữ liệu trong hệ thống R2AI.

---

## 1. Tổng quan

Hệ thống R2AI Stage 1 sử dụng **hai loại dữ liệu chính**:

| Loại | Mục đích | Lưu trữ |
|------|----------|---------|
| **Benchmark questions** | 2.000 câu hỏi pháp luật để đánh giá | File JSON local (`test/`) |
| **Knowledge base (corpus)** | Đoạn văn bản pháp luật doanh nghiệp để retrieve | Qdrant vector database |

Ngoài ra có các **file cache/phụ trợ** (sub-query, retrieved chunks, output citations) dùng để tăng tốc hoặc tái hiện kết quả.

---

## 2. Link truy cập dữ liệu

> Các link dưới đây dùng để chia sẻ dữ liệu cho người đánh giá / tái hiện kết quả. **Thay `YOUR_*_FOLDER_ID` bằng link thực tế trước khi nộp sản phẩm.**

| Dữ liệu | Link | Ghi chú |
|---------|------|---------|
| **Benchmark Stage 1** (questions + subqueries + sample output) | [Google Drive — R2AI Dataset](https://drive.google.com/drive/folders/YOUR_DATASET_FOLDER_ID) | Bắt buộc để chạy pipeline |
| **Corpus gốc** (518k văn bản pháp luật) | [HuggingFace — vietnamese-legal-documents](https://huggingface.co/datasets/vohuutridung/vietnamese-legal-documents) | Nguồn corpus; lọc 25 keyword tiêu đề |
| **Corpus đã lọc / Parquet** (tuỳ chọn) | [Google Drive — R2AI Corpus](https://drive.google.com/drive/folders/YOUR_CORPUS_FOLDER_ID) | Bản export sau lọc, dùng khi tái tạo Qdrant |
| **Qdrant snapshot / export** (tuỳ chọn) | [Google Drive — Qdrant Export](https://drive.google.com/drive/folders/YOUR_QDRANT_FOLDER_ID) | Thay thế nếu không dùng Qdrant Cloud |

**Truy cập Qdrant Cloud (production):** thông tin `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` được cung cấp riêng trong file `.env` (không public).

---

## 3. Dữ liệu benchmark — câu hỏi test

### 3.1. Nguồn

- **Benchmark R2AI Stage 1** — bộ 2.000 câu hỏi pháp luật doanh nghiệp tiếng Việt do ban tổ chức cung cấp.
- Câu hỏi bao quát các lĩnh vực: luật doanh nghiệp, thuế, lao động, hợp đồng, đấu thầu, bảo hiểm xã hội, v.v.

### 3.2. File

| File | Đường dẫn | Số bản ghi | Bắt buộc |
|------|-----------|------------|----------|
| Questions | `test/R2AIStage1DATA.json` | 2.000 | Có |
| Sub-queries (precomputed) | `test/R2AIStage1_subqueries (1).json` | 2.000 | Không |
| Retrieved cache | `test/R2AIStage1_retrieved.json` | ≤ 2.000 | Không |
| Citations output | `test/R2AIStage1_citations*.json` | ≤ 2.000 | Output |
| Full answers output | `test/R2AIStage1_answers.json` | ≤ 2.000 | Output |

### 3.3. Cấu trúc — Questions (`R2AIStage1DATA.json`)

**Định dạng:** JSON array  
**Encoding:** UTF-8

```json
[
  {
    "id": 1,
    "question": "Các cơ sở ươm tạo và khu làm việc chung được hưởng những chính sách hỗ trợ nào về thuế và đất đai?"
  },
  {
    "id": 2,
    "question": "Doanh nghiệp nhỏ và vừa được hưởng ưu đãi gì khi tham gia đấu thầu?"
  }
]
```

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `id` | integer | ID câu hỏi, từ 1 đến 2000, duy nhất |
| `question` | string | Câu hỏi pháp luật tiếng Việt |

### 3.4. Cấu trúc — Sub-queries cache

**File:** `test/R2AIStage1_subqueries (1).json`

```json
{
  "id": 3,
  "original_question": "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào và phải khắc phục ra sao?",
  "token_count": 42,
  "subquery_policy": "generate_2_subqueries",
  "num_queries": 2,
  "queries": [
    "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào?",
    "Công ty phải khắc phục như thế nào khi giữ bản chính bằng cấp của nhân viên?"
  ]
}
```

| Trường | Mô tả |
|--------|-------|
| `subquery_policy` | `keep_original`, `generate_2_subqueries`, hoặc `generate_3_subqueries` |
| `queries` | Danh sách truy vấn con dùng cho hybrid retrieval |

### 3.5. Cấu trúc — Output citations

**File:** `test/R2AIStage1_citations.json` (hoặc `R2AIStage1_citations_llm_top{N}.json`)

```json
{
  "id": 1,
  "question": "...",
  "answer": "",
  "relevant_docs": [
    "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa 2017"
  ],
  "relevant_articles": [
    "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa 2017|Điều 12"
  ]
}
```

**Quy ước chuỗi trích dẫn:**

| Trường | Format | Ví dụ |
|--------|--------|-------|
| `relevant_docs` | `{law_code}\|{document_title}` | `04/2017/QH14\|Luật Hỗ trợ doanh nghiệp nhỏ và vừa 2017` |
| `relevant_articles` | `{law_code}\|{document_title}\|{article}` | `...|Điều 12` hoặc `...|Phụ lục I` |

- `answer`: rỗng khi chạy `--citations-only`; chứa văn bản sinh khi chạy full pipeline.
- Tham số `--llm-top-k N` ảnh hưởng số lượng chunk đưa vào trích dẫn (thí nghiệm: top1, top2, top3, top5, top6).

### 3.6. Cấu trúc — Retrieved cache

**File:** `test/R2AIStage1_retrieved.json`

Mỗi item gồm `id`, `question`, `chunks[]`, `sub_queries`. Mỗi chunk chứa metadata retrieve đầy đủ:

```json
{
  "rank": 1,
  "score": 0.92,
  "rrf_score": 0.016,
  "dense_score": 0.643,
  "bm25_score": 53.52,
  "rerank_score": 0.92,
  "point_id": "fde9ed38-...",
  "document_number": "04/2017/QH14",
  "document_title": "Luật Hỗ trợ doanh nghiệp nhỏ và vừa 2017",
  "article_no": "Điều 12",
  "retrieval_text": "...",
  "content_text": "..."
}
```

---

## 4. Knowledge base — corpus pháp luật

### 4.1. Nguồn gốc

Corpus được xây dựng từ dataset công khai trên Hugging Face:

**[vohuutridung/vietnamese-legal-documents](https://huggingface.co/datasets/vohuutridung/vietnamese-legal-documents)**

| Thuộc tính | Giá trị |
|------------|---------|
| Nguồn gốc | [thuvienphapluat.vn](https://thuvienphapluat.vn) |
| Tổng văn bản gốc | 518.255 documents |
| Phạm vi thời gian | 1924 – 2026 |
| Định dạng | Parquet (2 config: `metadata` ~82 MB, `content` ~3.6 GB) |
| Giấy phép | CC BY 4.0 (bộ dataset biên soạn) |

Dataset gốc gồm hai config, join qua cột `id`:

| Config | Cột chính | Mô tả |
|--------|-----------|-------|
| `metadata` | `id`, `document_number`, `title`, `url`, `legal_type`, `legal_sectors`, `issuing_authority`, `issuance_date`, `signers` | Metadata, không có nội dung |
| `content` | `id`, `content` | Toàn văn Markdown |

### 4.2. Quy trình lọc corpus

Từ 518.255 văn bản gốc, corpus R2AI được lọc qua **hai bước**:

**Bước 1 — Lọc theo 25 keyword trong tiêu đề (`title`)**

Giữ văn bản nếu **tiêu đề chứa ít nhất một** trong 25 cụm từ dưới đây (logic OR). Số lần xuất hiện được thống kê trên toàn bộ metadata gốc:

| STT | Keyword / cụm từ | Số lần xuất hiện |
|-----|------------------|------------------|
| 1 | công ty | 1.788 |
| 2 | thuế | 512 |
| 3 | doanh nghiệp | 450 |
| 4 | xử lý | 389 |
| 5 | lao động | 389 |
| 6 | đăng ký | 362 |
| 7 | hợp đồng | 320 |
| 8 | hồ sơ | 286 |
| 9 | nhân viên | 284 |
| 10 | cơ quan | 262 |
| 11 | kinh doanh | 225 |
| 12 | yêu cầu | 224 |
| 13 | quy định | 194 |
| 14 | nội dung | 193 |
| 15 | điều kiện | 190 |
| 16 | thời hạn | 184 |
| 17 | thông tin | 181 |
| 18 | hỗ trợ | 176 |
| 19 | trách nhiệm | 151 |
| 20 | nghĩa vụ | 138 |
| 21 | thông báo | 136 |
| 22 | hàng hóa | 135 |
| 23 | khách hàng | 134 |
| 24 | quỹ | 124 |
| 25 | hóa đơn | 115 |

Danh sách keyword được định nghĩa trong `scripts/document_filters.py` (`TITLE_KEYWORDS`). Có thể override bằng `--keywords-file` khi ingest.

**Bước 2 — Lọc theo hiệu lực pháp luật**

- Ngày cắt (cutoff): **2026-03-01**
- Loại bỏ văn bản hết hiệu lực trước/on cutoff
- Loại bỏ danh mục văn bản hết hiệu lực
- Tra cứu bổ sung qua `effectiveness.parquet` (vbpl.vn / VietLex) nếu có

Sau lọc, văn bản được **chunk theo cấu trúc cây pháp luật** (Chương → Điều → Khoản) rồi embed và upsert vào Qdrant.

### 4.3. Tải và tái tạo corpus từ Hugging Face

```bash
pip install datasets pandas pyarrow
```

```python
from datasets import load_dataset
import pandas as pd

# Metadata (~82 MB)
meta = load_dataset(
    "vohuutridung/vietnamese-legal-documents", "metadata"
)["data"].to_pandas()

# Lọc 25 keyword (logic giống scripts/document_filters.py)
keywords = [
    "công ty", "thuế", "doanh nghiệp", "xử lý", "lao động",
    "đăng ký", "hợp đồng", "hồ sơ", "nhân viên", "cơ quan",
    "kinh doanh", "yêu cầu", "quy định", "nội dung", "điều kiện",
    "thời hạn", "thông tin", "hỗ trợ", "trách nhiệm", "nghĩa vụ",
    "thông báo", "hàng hóa", "khách hàng", "quỹ", "hóa đơn",
]
title = meta["title"].str.lower().fillna("")
mask = False
for kw in keywords:
    mask = mask | title.str.contains(kw, regex=False, na=False)
filtered_meta = meta[mask]
print(f"Sau lọc keyword: {len(filtered_meta):,} / {len(meta):,}")

# Export metadata đã lọc (tuỳ chọn)
filtered_meta.to_parquet("../data/metadata.parquet", index=False)
```

Export metadata + lọc hiệu lực bằng script có sẵn:

```bash
python scripts/export_filtered_metadata.py \
  --data-dir ../data \
  --output ../data/filtered_by_keywords_effective.parquet
```

Ingest chunk + embed vào Qdrant:

```bash
python scripts/ingest_parquet_to_qdrant.py \
  --parquet-path ../data/full.parquet \
  --metadata-filter ../data/filtered_by_keywords_effective.parquet \
  --env-file .env
```

### 4.4. Lưu trữ — Qdrant

| Thuộc tính | Giá trị production |
|------------|-------------------|
| Platform | Qdrant Cloud |
| Collection | `vld_business_law_v2` |
| Dense vector | `dense`, 1024 chiều |
| Sparse vector | `bm25` (BM25 pre-indexed) |
| Embedding model | `AITeamVN/Vietnamese_Embedding_v2` |

### 4.5. Payload mỗi chunk (Qdrant point)

Sau khi normalize (`scripts/qdrant_config.py`):

| Trường | Mô tả | Ví dụ |
|--------|-------|-------|
| `document_id` | ID nội bộ | UUID |
| `document_number` | Mã văn bản | `04/2017/QH14` |
| `document_title` | Tên văn bản | `Luật Hỗ trợ doanh nghiệp nhỏ và vừa 2017` |
| `legal_type` | Loại văn bản | `Luật`, `Nghị định`, … |
| `article_no` | Số điều/khoản | `Điều 12` |
| `chunk_id` | ID chunk trong văn bản | string |
| `retrieval_text` | Text dùng embed/search | đoạn văn đã xử lý |
| `content_text` | Nội dung gốc chunk | đoạn văn đầy đủ |
| `source_url` | URL nguồn | `https://thuvienphapluat.vn/...` |

**Alias legacy** (vẫn được hỗ trợ): `law_code`, `law_title`, `doc_id`.

### 4.6. File Parquet local (sau khi tải & xử lý)

Thư mục `../data/` (ngoài repo, gitignored):

| File | Mô tả |
|------|-------|
| `metadata.parquet` | Metadata từ HuggingFace (518k rows) |
| `content.parquet` | Nội dung Markdown (join qua `id`) |
| `full.parquet` | Metadata + content đã merge |
| `filtered_by_keywords_effective.parquet` | Metadata sau lọc 25 keyword + hiệu lực |
| `effectiveness.parquet` | Trạng thái hiệu lực từ vbpl.vn (sidecar) |

---

## 5. Hướng dẫn sử dụng dữ liệu

### 5.1. Chuẩn bị benchmark

```bash
# Tạo thư mục test
mkdir -p test

# Giải nén file tải từ Google Drive vào test/
# Cần ít nhất: R2AIStage1DATA.json
```

### 5.2. Load questions trong code

Script `scripts/rag_answer_stage1.py` đọc questions qua hàm `load_questions()`:

```python
# Mặc định
questions_path = "test/R2AIStage1DATA.json"

# Giới hạn phạm vi
python scripts/rag_answer_stage1.py --start-id 100 --limit 50
```

### 5.3. Chế độ cache

| Mục đích | Lệnh |
|----------|------|
| Tạo cache retrieve | `--retrieve-only --output test/R2AIStage1_retrieved.json` |
| Dùng cache, bỏ Qdrant | `--skip-retrieve` |
| Resume output cũ | `--skip-answered` |

### 5.4. BM25 local cache

Khi chạy với Qdrant local (không có sparse vector trên Cloud), BM25 index được build từ corpus scroll và cache tại:

```
output/bm25_corpus.pkl
```

File này được tái sử dụng tự động nếu collection không đổi.

---

## 6. Thống kê dữ liệu

| Chỉ số | Giá trị |
|--------|---------|
| Số câu hỏi benchmark | 2.000 |
| ID range | 1 – 2000 |
| Ngôn ngữ | Tiếng Việt |
| Lĩnh vực | Pháp luật doanh nghiệp |
| Corpus gốc (HuggingFace) | 518.255 văn bản |
| Keyword lọc tiêu đề | 25 cụm từ (OR) |
| Corpus sau lọc (Qdrant) | Hàng chục nghìn chunk văn bản pháp luật doanh nghiệp |

---

## 7. Lưu ý bảo mật và chia sẻ

- File `.env` chứa API key Qdrant — **không chia sẻ công khai**.
- Thư mục `test/`, `models/`, `data/`, `output/` được liệt kê trong `.gitignore`.
- Khi nộp sản phẩm: chia sẻ benchmark qua Google Drive/OneDrive; cung cấp `.env.example` và hướng dẫn lấy credentials Qdrant riêng cho ban giám khảo (nếu được phép).
