# Research: Giải pháp tốt nhất cho Vietnamese Legal IR/QA + Notebook full

**Ngày:** 2026-06-05 14:48 · **Loại:** Research + Implementation decision

## Executive Summary

Sau khi tune retrieval (F2 0.586→0.701, precision 0.285→0.693) và đối chiếu literature, **kiến trúc hiện tại đã đúng hướng winner** (ALQAC 2024: hybrid BM25+dense + rerank + LLM prompt). Không cần đổi kiến trúc. Chỉ tinh chỉnh prompt (zero-shot, nhắm 5 tiêu chí QA) + dựng notebook Colab full sinh `results.json` 2000 câu có answer thật.

## Phát hiện chính

| # | Nguồn | Kết luận → quyết định |
|---|---|---|
| 1 | [Top-2 ALQAC 2024](https://www.researchgate.net/publication/387913243) | Winner dùng hybrid BM25+sentence-transformer + LLM prompt engineering → **ta đang làm đúng**, giữ nguyên kiến trúc |
| 2 | [VLQA 2025](https://arxiv.org/html/2507.19995) | **Few-shot LÀM GIẢM chất lượng ở LLM <14B** → dùng **zero-shot grounded**. Qwen2.5-7B 32k-context mạnh cho VN legal |
| 3 | [Stanford Legal RAG](https://dho.stanford.edu/wp-content/uploads/Legal_RAG_Hallucinations.pdf) | **StrictCitations** (ép trích dẫn nguồn mọi khẳng định) = cơ chế chống bịa tốt nhất → ta đã ép "Điều X" |
| 4 | [RAG eval 2025](https://www.getmaxim.ai/articles/rag-evaluation-a-complete-guide-for-2025/) | Grounding giảm bịa ~71% nhưng không triệt tiêu → giữ answer bám retrieved text |

## Quyết định triển khai (KISS — không over-engineer)

**Giữ nguyên (đã tối ưu/đúng):**
- Retrieval: hybrid dense(BGE-M3 VN) + BM25 + RRF + reranker + cutoff (top8/min0/margin4). F2=0.701.
- Citation: điền `relevant_*` từ retrieval (chân lý) + ép "Điều X" vào answer.

**Điều chỉnh:**
- Prompt LLM: zero-shot (BỎ ý định few-shot), nhắm thẳng 5 tiêu chí QA (chính xác/đầy đủ/thực tiễn/rõ ràng) + StrictCitations. ✅ đã sửa `local_llm_client.build_prompt`.

**KHÔNG làm (YAGNI):**
- Query expansion/HyDE: literature không bắt buộc; recall đã 0.74. Bỏ.
- Dedup luật 2 phiên bản: nhiễu nhỏ. Bỏ.
- Few-shot: có hại với 7B. Bỏ.

## Notebook Colab full (deliverable)

Self-contained (không clone, không chromadb — dùng numpy cosine):
1. Upload `corpus_articles.jsonl` (3777) + `stage1_questions.json` (2000).
2. **Phase A** (embed+rerank trên GPU): embed corpus → numpy cosine dense + BM25 + RRF + rerank + cutoff → cache điều luật/câu. Free model.
3. **Phase B** (Qwen-7B 4bit): sinh answer zero-shot grounded + ép citation, **checkpoint mỗi 50 câu** (resume nếu Colab ngắt).
4. **Phase C**: `results.json` + `submission.zip` (zip phẳng) → tải về.

VRAM staged (free embed/rerank trước khi load LLM) → đỉnh ~5.5GB, vừa T4 16GB. Batched reranker. Checkpoint/resume cho 2000 câu.

## Câu hỏi mở
1. Format `<tên văn bản>` (có/không nhúng mã) — hỏi BTC; ta theo ví dụ mẫu.
2. F2 thật trên 2000 câu chỉ biết khi nộp leaderboard (gold_dev là tạm).
3. Thời gian LLM 2000 câu trên T4 free (~1-2h) có thể chạm giới hạn session → checkpoint/resume đã xử lý.
