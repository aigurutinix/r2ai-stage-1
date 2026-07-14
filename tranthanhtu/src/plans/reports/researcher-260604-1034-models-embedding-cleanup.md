# Research: Model hợp lệ tốt nhất + Chẩn đoán embedding nặng + Dọn folder

**Ngày:** 2026-06-04 10:34 (UTC+7) · **Loại:** Research + Cleanup plan

## TOC
1. [Embedding nặng — nguyên nhân & fix](#1)
2. [Model hợp lệ tốt nhất](#2)
3. [Dọn folder](#3)
4. [Notebook self-contained Colab](#4)
5. [Next steps + câu hỏi mở](#5)

---

## 1. Embedding nặng — CÓ bug kỹ thuật thật {#1}

**Nguyên nhân (xác nhận qua research + soi repo):**

| # | Vấn đề | Bằng chứng | Fix |
|---|---|---|---|
| A | **BGE-M3 `max_seq_length=8192` mặc định**, code KHÔNG cap | BGE-M3 docs: default 8192, fine-tune ở 1024, "set nhỏ hơn để tăng tốc". Attention O(n²) → 8192 nặng gấp ~64× so với 1024 | `model.max_seq_length=1024` (embed), `max_length=512` (rerank) |
| B | **chroma_db 1.2GB = rác cũ** (embed 169 luật / Gemini), chưa có `corpus_articles.jsonl` | `du chroma_db`=1.2G nhưng `corpus_articles.jsonl` không tồn tại → DB không từ pipeline mới | `rm -rf data/chroma_db`, embed chỉ ~12 luật SME (allowlist) |
| C | Có thể chạy CPU thay GPU | `get_device` cũ bỏ sót CUDA (đã fix phiên trước) | dùng CUDA trên Colab |

→ Embed **đúng cách** (cap seq len + 12 luật + GPU) chỉ còn **vài nghìn Điều**, nhẹ và nhanh (phút), không phải "rất nặng".

---

## 2. Model hợp lệ tốt nhất (open, <14B, trước 01/03/2026) {#2}

### Embedding — giữ `AITeamVN/Vietnamese_Embedding` (v1)
- BGE-M3 fine-tune VN, mạnh legal (Legal Zalo 2021). MIT, 0.6B, 2025. ✅
- **Bắt buộc cap `max_seq_length=1024`** (điểm fix #A). Không cần đổi model — chỉ cấu hình sai.
- Alternatives (không cần): `truro7/vn-law-embedding` (legal-specific, ít kiểm chứng), `bkai vietnamese-bi-encoder` (nhẹ hơn, PhoBERT 135M).

### Reranker — giữ `AITeamVN/Vietnamese_Reranker`
- BGE-reranker-v2-m3 VN, MIT. Pairs tự nhiên với embedding. Cap `max_length=512`.
- Alternatives mạnh: **PhoRanker** (NDCG@10 nhỉnh hơn), **ViRanker** (2025, tốt top-rank). Có thể thử sau nếu cần đẩy điểm.

### LLM <14B — `Qwen2.5-7B-Instruct` (mặc định, an toàn nhất)
| Model | Params | License | VN | Verdict |
|---|---|---|---|---|
| **Qwen2.5-7B-Instruct** | 7.6B | **Apache-2.0** | tốt (trong VLegal-Bench) | ✅ **Chọn** — mạnh + license sạch |
| Qwen3-8B | 8B | Apache-2.0 (04/2025) | tốt hơn, có reasoning | ✅ Nâng cấp được (cần `/no_think` để ổn format) |
| Gemma-2-9B-it | 9.2B | **Gemma license** | tốt | ⚠️ không phải OSI → rủi ro chữ "mã nguồn mở" |
| Vistral-7B | 7B | Mistral + điều khoản VN | **mạnh nhất VN** | ⚠️ kiểm license kỹ |
| SeaLLMs-v3-7B | 7B | SeaLLM license | tốt SEA | ⚠️ hạn chế non-commercial |
| ~~Qwen2.5-14B~~ | 14.7B | Apache | — | ❌ **>14B, loại** |

→ **Mặc định Qwen2.5-7B-Instruct** (Apache sạch, đủ mạnh). LLM chủ yếu ảnh hưởng QA (chấm thủ công sau); IR (F2, chấm ngay) phụ thuộc retrieval. Không cần model "VN-specialist" rủi ro license. Không giới hạn RAM → có thể thử Qwen3-8B nếu muốn.

---

## 3. Dọn folder (rất lộn xộn) {#3}

**Xóa (legacy web-app Gemini/Antigravity — đã pivot, đã commit nên khôi phục được):**
- `backend/`: `agent.py`, `rag_engine.py`, `ingestion.py`, `document_generator.py`, `main.py`, `requirements.txt` (cũ), `rebuild_backend_env.sh`, `.env.example` (Gemini), `static/` (169 docx — giờ dùng HF dataset)
- `frontend/`, `frontend_vanilla/` (UI cũ)
- `scratch/`: `check_names.py`, `migrate_to_chromadb*.py`, `test_*.py`, `test_law.docx`, `__pycache__`
- `data/`: `chroma_db` (rác 1.2G), `law.json`, `document_registry.json`, `legal_list_id.json` (corpus cũ — thay bằng HF)
- `notebooks/kaggle_pipeline.ipynb` (bỏ Kaggle)

**Giữ (pipeline thi):**
- `backend/`: `local_models_config.py`, `legal_text_parser.py`, `build_corpus.py`, `local_ingestion.py`, `local_reranker.py`, `local_rag_engine.py`, `local_llm_client.py` (Ollama — cho Mac), `requirements-local.txt`
- `scratch/`: `generate_submission.py`, `eval_f2.py`, `make_notebooks.py`
- `data/`: `test_questions.json`, `gold_dev.json`
- `docs/`, `plans/`, `README.md`, `.gitignore`
- `colab/legal_pipeline_colab.ipynb` (mới — self-contained)

→ Cấu trúc gọn: `backend/` (7 module pipeline) + `scratch/` (3 script) + `data/` (2 json) + `colab/` (1 notebook) + `docs/` + `plans/`.

---

## 4. Notebook self-contained Colab {#4}

- **1 folder `colab/`, 1 notebook**, KHÔNG `git clone` — tạo file code bằng `%%writefile` ngay trong notebook (đầy đủ code, test chuẩn).
- Code module = **đọc từ file canonical lúc generate** (đồng bộ, DRY). Khác biệt duy nhất với code gốc: **LLM dùng `transformers` 4-bit thay Ollama** (cell riêng + vòng sinh inline, tái dùng `build_prompt`/`build_citation_fields`/`ensure_citations`).
- Generator `scratch/make_notebooks.py` viết lại để sinh notebook self-contained này.

---

## 5. Next steps + câu hỏi mở {#5}

**Thực thi ngay (phiên này):**
1. Cap `max_seq_length` (config + ingestion + rag + reranker).
2. `git rm` toàn bộ legacy (mục §3).
3. Xóa Kaggle nb, tạo `colab/` self-contained nb.

**Câu hỏi mở:**
1. Real test set (03/06) phủ luật nào ngoài 12 allowlist? → có thể cần mở rộng allowlist / bật keywords mode.
2. Có muốn thử Qwen3-8B / Vistral-7B (mạnh hơn nhưng cần check license/format) không, hay giữ Qwen2.5-7B an toàn?
3. Mâu thuẫn quy định "dữ liệu ngoài" của BTC (chưa giải quyết từ report trước).

## Nguồn
- [BGE-M3 max_seq_length 8192, fine-tune 1024](https://huggingface.co/BAAI/bge-m3)
- [AITeamVN/Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding)
- [VLegal-Bench (Qwen2.5/Gemma2/Llama3/SeaLLMs)](https://arxiv.org/html/2512.14554)
- [ViRanker vs PhoRanker (arXiv 2509.09131)](https://arxiv.org/abs/2509.09131)
- [Improving Vietnamese Legal Retrieval w/ Synthetic Data](https://arxiv.org/abs/2412.00657)
