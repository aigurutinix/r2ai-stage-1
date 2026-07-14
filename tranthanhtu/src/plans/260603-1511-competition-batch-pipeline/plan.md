# Plan: Competition Offline Batch Pipeline (Vietnamese Legal IR/QA)

**Mode:** cook (user override "sửa code luôn" → no review gates)
**Ref:** [research report](../reports/researcher-260603-1442-competition-pivot-offline-pipeline-review.md)

## Decisions (user)
1. Corpus từ HF datasets BTC: `tmquan/vbpl-vn` (chính), `phapdien-moj-gov-vn`, `anle-toaan-gov-vn`.
2. Claude tự tạo gold answers cho 20 câu mock → eval F2 nội bộ.
3. Sửa code luôn.

## Key facts
- vbpl-vn: 158k docs, `doc_number` (list) = mã, `markdown` = full text, `legal_type` clean. `legal_area` đa số "Chưa phân loại" → lọc bằng allowlist doc_number + legal_type.
- Embedding: `AITeamVN/Vietnamese_Embedding` (v1, BGE-M3, 1024-dim, MIT). Reranker: `AITeamVN/Vietnamese_Reranker`.
- LLM: `qwen2.5:7b-instruct-q4_K_M` (Ollama).
- IR join key = `mã|Điều N` (name fuzzy/phụ).

## Files
| File | Action |
|---|---|
| `backend/local_models_config.py` | NEW — model names, paths, allowlist doc_number |
| `backend/legal_text_parser.py` | NEW — parse_legal_name + split markdown→Điều (DRY) |
| `backend/build_corpus.py` | NEW — stream vbpl-vn, filter, parse → `data/corpus_articles.jsonl` |
| `backend/local_ingestion.py` | REWRITE — đọc JSONL, embed v1, fail-loud, validate code |
| `backend/local_reranker.py` | NEW — CrossEncoder wrapper |
| `backend/local_rag_engine.py` | REWRITE — v1, BM25(pyvi opt), RRF, +reranker, bỏ slang/time/conflicts |
| `backend/local_llm_client.py` | EDIT — prompt chặt hơn |
| `scratch/generate_submission.py` | REWRITE — fields từ retrieval (chân lý), fix bug match, ép citation, validate schema, zip |
| `scratch/eval_f2.py` | NEW — P/R/F2 macro vs gold |
| `data/gold_dev.json` | NEW — 20 gold answers (Claude) |
| `backend/requirements-local.txt` | NEW — pinned deps |

## Pipeline flow
question → (dense v1 raw query ‖ BM25) → RRF → reranker → Top-K Điều → [fields trực tiếp] + [LLM answer] → ép citation → results.json + zip

## Run order (user chạy)
0. `pip install -r backend/requirements-local.txt`
1. `ollama serve` + `ollama pull qwen2.5:7b-instruct-q4_K_M`
2. `rm -rf data/chroma_db`
3. `python backend/build_corpus.py` → corpus_articles.jsonl
4. `python backend/local_ingestion.py` → ChromaDB
5. `python backend/local_rag_engine.py` → smoke test retrieval
6. `python scratch/eval_f2.py` → đo F2 nội bộ
7. `python scratch/generate_submission.py` → results.json + submission.zip

## Status: CODE DONE (chưa chạy runtime — cần máy user: dataset + Ollama)

### Code review (code-reviewer) — đã xử lý
- C1 (Critical): regex Điều bắt buộc space sau dấu chấm → FIXED (`\s*`, markdown phẳng OK).
- C2 (Critical): guard chữ hoa loại nhầm điều title chữ thường → FIXED (chấp nhận alnum/quote + _REF_PREFIX loại inline ref).
- H1: thêm counter doc bị bỏ do null markdown.
- H2: keyword blob dùng `summary` thay `doc_name`. H3: mở rộng KEEP_LEGAL_TYPES.
- H4/L1/M3: clamp n_results, guard demo score, anchor eval regex.
- Verified clean: metadata keys khớp giữa ingestion↔rag↔submission; eval_f2 parse đúng.

### Còn lại (user lưu ý khi chạy)
- gold_dev.json là gold TẠM (Claude tạo) — verify trước khi tin số F2.
- Reranker `AITeamVN/Vietnamese_Reranker` load qua CrossEncoder chưa chạy thực — nếu lỗi, thử FlagEmbedding.FlagReranker.
- Chưa chạy build_corpus thực: phải grep JSONL xem mỗi luật allowlist ra đủ số Điều (Bộ luật phải 100+ Điều, không phải 3).
