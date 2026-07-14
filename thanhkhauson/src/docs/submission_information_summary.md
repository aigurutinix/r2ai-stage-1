# Tổng Hợp Thông Tin Nộp Bài

Tài liệu này tổng hợp các thông tin chính phục vụ đánh giá, nghiệm thu và tái hiện sản phẩm R2AI2026 Legal Assistant. Các phần mô tả chi tiết về dữ liệu và model được tách sang tài liệu riêng để tránh lặp nội dung:

- Mô tả dữ liệu: [data_description.md](data_description.md)
- Mô tả model và checkpoint: [model_description.md](model_description.md)
- Hướng dẫn cài đặt/chạy lại từ đầu: [../README.md](../README.md)

## 1. Thông Tin Chung

| Mục | Thông tin |
| --- | --- |
| Tên sản phẩm | R2AI2026 Legal Assistant |
| Lĩnh vực | Hỏi đáp và truy hồi thông tin pháp lý tiếng Việt |
| Mục tiêu | Tìm văn bản/điều luật liên quan và sinh câu trả lời có căn cứ |
| Đối tượng hướng tới | SME/doanh nghiệp và người cần tra cứu tình huống pháp lý |
| Cuộc thi | R2AI2026 BUILD AI LEGAL ASSISTANT |
| Nền tảng | https://leaderboard.aiguru.com.vn/competitions/13/ |
| Deadline tài liệu thuyết minh | Trước 17h30 ngày 30/06/2026 |

## 2. Tóm Tắt Hệ Thống

R2AI2026 Legal Assistant là hệ thống RAG pháp lý tiếng Việt. Pipeline chính:

1. Nhận câu hỏi pháp lý tiếng Việt.
2. `planner_node` phân tích câu hỏi và tạo các mục tiêu truy hồi.
3. `batch_hybrid_search_node` truy hồi kết hợp Elasticsearch BM25 và Qdrant vector search.
4. Kết quả truy hồi được hợp nhất bằng RRF và rerank bằng ViRanker.
5. `compress_node` dùng LLM để lọc các chunk thật sự liên quan, loại nhiễu và tạo bộ chứng cứ sạch.
6. `reasoning_node` sinh câu trả lời cuối bằng Qwen3.5-9B qua endpoint OpenAI-compatible; nếu thiếu căn cứ có thể gọi thêm `hybrid_search_tool`.
7. Xuất `results.json` theo schema cuộc thi.

Đầu ra mỗi câu hỏi:

```json
{
  "id": 1,
  "question": "...",
  "answer": "...",
  "relevant_docs": ["<document_number>|<doc_name>"],
  "relevant_articles": ["<document_number>|<doc_name>|<article_no>"]
}
```

## 3. Dữ Liệu Sử Dụng

Dữ liệu chính đến từ HuggingFace dataset `th1nhng0/vietnamese-legal-documents`, folder `legacy`, và dữ liệu Bộ luật/Luật crawl từ `vbpl.vn`. Bản dữ liệu đóng gói hiện tại đã loại bỏ các metadata có `effect_status = "No longer applicable"` trong `metadata_merged_active.jsonl`.

Chi tiết nguồn dữ liệu, schema, số dòng, quan hệ join và cách sử dụng 4 file JSONL nằm trong [data_description.md](data_description.md).

Các file dữ liệu cần có sau khi giải nén:

```text
data/raw_law/law_only/others_doc/parsed_others_doc_database.jsonl
data/raw_law/law_only/others_doc/metadata_merged_active.jsonl
data/raw_law/law_only/effective/law_luoc_do_merged_dedup.jsonl
data/raw_law/law_only/effective/parsed_law_database.jsonl
```

File zip dữ liệu đã tiền xử lý:

```text
preprocessed_active_legal_data_20260630.zip
```

## 4. Model Và Checkpoint

Hệ thống sử dụng các thành phần model sau:

| Thành phần | Checkpoint/package | Vai trò |
| --- | --- | --- |
| LLM | `Qwen/Qwen3.5-9B` | Lập kế hoạch, lọc ngữ cảnh và sinh câu trả lời |
| Reranker | `namdp-ptit/ViRanker` | Xếp hạng lại các chunk truy hồi |
| Embedding | `mainguyen9/vietlegal-harrier-0.6b` | Tạo embedding cho truy hồi vector |
| Tách từ tiếng Việt | `py-vncorenlp==0.1.4` | Tạo trường tokenized cho BM25 |

Chi tiết model card, điều kiện cuộc thi, lệnh tải checkpoint và ví dụ sử dụng nằm trong [model_description.md](model_description.md).

Các checkpoint mặc định đặt trong:

```text
model_cache/Qwen3.5-9B/
model_cache/ViRanker/
model_cache/vietlegal-harrier-0.6b/
model_cache/vncorenlp/
```

## 5. Mã Nguồn Và Cấu Hình

Các thành phần mã nguồn chính:

| Nhóm | File/thư mục | Vai trò |
| --- | --- | --- |
| Agent/RAG | `Agents/graph.py`, `Agents/tools/search_legal.py` | LangGraph pipeline và hybrid search |
| Cấu hình | `config.json`, `Agents/config.py`, `.env.example` | Cấu hình model, endpoint, database và biến môi trường |
| Tiền xử lý | `pre_tokenize.py` | Tạo trường `tokenized_content_search` cho BM25 |
| Build chỉ mục | `build_hybrid_index.py` | Nạp dữ liệu vào Elasticsearch và Qdrant |
| Compression | `Agents/graph.py::compress_node` | Lọc chunk liên quan và tạo bộ chứng cứ sạch trước khi reasoning |
| Chạy LLM | `start_eval.sh` | Serve Qwen3.5-9B bằng vLLM |
| Evaluation | `evaluation.py` | Chạy suy luận hàng loạt và tạo `results.json`/`submission.zip` |
| Database local | `dbs/start_dbs.sh` | Khởi động Elasticsearch và Qdrant |

Danh sách thư viện phụ thuộc nằm trong `requirements.txt`. File `.env.example` được dùng làm mẫu cấu hình và không chứa khóa thật.

## 6. Luồng Tái Hiện Kết Quả

Luồng chạy lại từ đầu được mô tả chi tiết trong [../README.md](../README.md). Tóm tắt các bước chính:

```text
cài môi trường
-> tải dữ liệu và checkpoint
-> khởi động Elasticsearch/Qdrant
-> tiền xử lý tokenized text
-> build chỉ mục hybrid
-> serve LLM
-> chạy evaluation
-> kiểm tra results.json/submission.zip
```

Các lệnh chính:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
unzip preprocessed_active_legal_data_20260630.zip -d .
bash dbs/start_dbs.sh
VNCORENLP_PRETOKENIZE_WORKERS=8 python pre_tokenize.py
INDEX_DEVICE=cuda:0 python build_hybrid_index.py
bash start_eval.sh
CHECKPOINTER_BACKEND=none python evaluation.py --input R2AIStage1DATA.json --out-dir submission_eval --graph-mode stateless
```
