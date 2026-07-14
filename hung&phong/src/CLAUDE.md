# Hướng dẫn cho Claude — dự án Chatbot VBPL

## Ngôn ngữ (QUAN TRỌNG)
- **LUÔN trả lời bằng TIẾNG VIỆT** trong mọi phản hồi, báo cáo, giải thích.
- Code/comment có thể giữ tiếng Việt như hiện trạng; commit message tiếng Việt hoặc Anh đều được.

## Đọc trước khi làm (QUAN TRỌNG)
- **Luôn đọc `project.md`** khi bắt đầu task mới để nắm thể lệ, metric chấm điểm và dữ liệu.
- **Đọc `docs/SUBMISSIONS.md`** để biết version hiện tại, điểm BTC, và hướng cải thiện tiếp theo.

## Triết lý ưu tiên (QUAN TRỌNG — chỉ thị của chủ dự án)
1. **RETRIEVE là quan trọng nhất.** F2 ưu tiên recall → tìm đủ gold mới là gốc. Đừng dồn
   sức vào khâu lọc/judge khi retrieval còn sót gold (bài học v10/v11: tối ưu nhầm chỗ).
2. **Muốn retrieve tốt thì BƯỚC ĐẦU PHẢI CÓ DATA.** Trước khi tinh chỉnh model/ranking,
   luôn kiểm tra độ phủ & chất lượng corpus: văn bản gold có trong corpus không? parse có
   đúng không? thiếu nghị định/thông tư chuyên ngành nào không?

## Quy trình BẮT BUỘC sau mỗi lần nộp (post-submission review)
> Ghi lại kết quả mỗi lần vào `docs/POST_SUBMISSION_REVIEW.md`. KHÔNG được chỉ liếc 1 con số.
1. **Mở điểm THẬT** từ `scoring_result` của BTC — mổ xẻ **cả ART lẫn DOCS** (F2/P/R), so
   với version trước. Chú ý chỉ số nào tăng/giảm và TẠI SAO.
2. **Mở đáp án đã nộp ra đọc cụ thể** — chọn mẫu câu, đối chiếu điều đã chọn vs nội dung,
   tự đánh giá đúng/sai bằng kiến thức pháp luật. Tìm pattern lỗi (sót gold? nhiễu? sai vùng?).
3. **Truy đến cùng trước khi kết luận** — không báo động sai, không kết luận "data đủ" khi
   mới check sơ bộ. Phân biệt rõ lỗi do data / ranking / lọc.
4. **Note lại** phát hiện vào `docs/POST_SUBMISSION_REVIEW.md` + cập nhật `docs/SUBMISSIONS.md`.

## Bối cảnh dự án (cập nhật 2026-06-14)
- RAG tra cứu & hỏi đáp Văn bản Pháp luật VN, phục vụ cuộc thi (xem `project.md`).
- LLM: Qwen3.5:9b qua Ollama (`qwen-vbpl`, think=True cho judge). Embedding: **AITeamVN/Vietnamese_Embedding_v2** (1024-dim, local GPU).
- Vector store: **Qdrant SERVER (Docker, `QDRANT_URL=http://localhost:6333`)** — collection **`vbpl_aiteam`**, **~285k chunk** (sau overhaul v14). Dùng HNSW native của server → `USE_HNSW=false`.
- BM25 index: **`data/bm25_vbpl_aiteam.pkl`** (285k doc). Reranker: `bge-reranker-v2-m3`.
- Phạm vi corpus: Luật Doanh nghiệp, SME, luật nền tảng VN (`ingest/scope.py`).
- **Best submission: v14 (ART_F2=0.5170, recall 0.7153, precision 0.2883 — overhaul DATA).** Bước tiếp: effect-status dedup (bản cũ trùng chủ đề) + judge CoT (lạc lĩnh vực) để đẩy precision → F2 0.6. Xem `docs/POST_SUBMISSION_REVIEW.md` Review #3.
- Tầng data đã làm lại: chunk theo cấu trúc (`ingest/chunk.py`), chuẩn hoá dấu thanh (`backend/textnorm.py`), tự verify (`scripts/inspect_data.py`), re-embed (`scripts/embed_from_parquet.py` + `embed_tail.py` cho chunk dài batch nhỏ chống OOM).
- Đánh giá: KHÔNG có gold offline → phải submit BTC mới biết điểm. `scripts/manual_eval.py` (15 câu biết gold) để sanity-check nhanh index.
- Lịch sử nộp: `docs/SUBMISSIONS.md`.

## Lưu ý kỹ thuật
- Chạy Python cần `PYTHONUTF8=1` (console cp1252 sẽ lỗi chữ tiếng Việt) và `PYTHONPATH=.`.
- **Qdrant giờ là SERVER (Docker container `qdrant`, port 6333)** — chạy được nhiều script song song (khác embedded khoá 1 process). Khởi động: `docker start qdrant`.
- Env vars cần thiết: `QDRANT_COLLECTION=vbpl_aiteam QDRANT_URL=http://localhost:6333 EMBED_BACKEND=st EMBED_ST_MODEL=AITeamVN/Vietnamese_Embedding_v2 HYBRID_SEARCH=true USE_HNSW=false USE_RERANKER=true BM25_INDEX_PATH=data/bm25_vbpl_aiteam.pkl`
- Re-embed chunk dài dễ OOM (batch 64 × chunk 3400 ký tự) → dùng `embed_tail.py` batch 8 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` để bù.
- `.env` bị gitignore (không commit); cấu hình mẫu ở `.env.example`.
