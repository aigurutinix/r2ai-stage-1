# Research: Vì sao kẹt 0.56 trong khi top 0.69 — và đường đột phá

**Ngày:** 2026-06-18 | **Metric:** ARTICLES_F2_MACRO (recall-weighted) | **Hiện tại:** 0.5608 (old corpus, t5m3) · **Top:** 0.69

## Executive Summary (brutal)

1. **Corpus vbpl mới TỆ HƠN** — đo xong leaderboard: t5m3 **0.4475**, t6m4 **0.5123** (đều < 0.5608 cũ). → **REVERT về corpus cũ (hf).** Corpus chưa bao giờ là nút thắt (ceiling đã ~0.9-0.97).
2. **Vấn đề cấu trúc đã định danh:** ta dùng **retriever OFF-THE-SHELF** (BGE-M3 + BGE-reranker-v2-m3, KHÔNG fine-tune). Mọi đội VN-legal đạt 0.65-0.70 đều **FINE-TUNE retriever bằng synthetic data + hard negatives**. Tinh chỉnh corpus/cutoff KHÔNG bao giờ lấp được gap 0.13 vốn là gap CHẤT LƯỢNG MODEL.
3. **Bằng chứng cứng:** paper VN-legal (arXiv 2412.00657) — BGE-M3 off-the-shelf trên Zalo Legal = **64.4 MRR@10**; bi-encoder fine-tune = **79.3**; ColBERT = **84.2** (+20-30 điểm). Đúng bằng khoảng cách 0.56→0.69.
4. **1 cheap win thử ngay (không cần train):** reranker đang cắt passage ở **256 token** — điều luật dài 1000+ token bị cụt → reranker chấm sai điều. Nâng `RERANK_MAX` 256→512/1024 = 1 lần chạy Phase A.

## Bằng chứng — số liệu

### Corpus vbpl mới (đã đo, leaderboard)
| tag | ARTICLES_F2 | A_P | A_R | DOCS_F2 | D_P | D_R |
|---|---|---|---|---|---|---|
| t5m3 | 0.4475 | 0.394 | 0.585 | 0.522 | 0.436 | 0.633 |
| t6m4 | 0.5123 | 0.439 | 0.570 | 0.549 | 0.456 | 0.610 |
| **cũ t5m3** | **0.5608** | ~0.56 | ~0.55 | — | — | — |

→ vbpl thua rõ. top-1 đổi 46% vs corpus cũ (đổi nhiều nhưng đổi theo hướng XẤU). **Bỏ vbpl, giữ corpus cũ.**

### Manh mối sắc: DOCS_F2 > ARTICLES_F2
Cả 2 lần đo: DOCS_F2 (0.52-0.55) **>** ARTICLES_F2 (0.45-0.51). **Ta tìm ĐÚNG văn bản nhưng SAI Điều bên trong.** Khớp đúng tài liệu retrieval lý thuyết: BM25/dense giỏi tìm DOC, kém định vị ĐIỀU. Reranker off-the-shelf không đủ sắc ở cấp Điều/Khoản → đây là nơi mất điểm.

## Vấn đề cốt lõi (định danh)

**Ta dùng model retrieval ZERO fine-tune.** BGE-M3 (embed) + BGE-reranker-v2-m3 (rerank) là multilingual general-purpose. Trên VN-legal chúng plateau ~0.55-0.62. Đây là trần MODEL, không phải trần data/cutoff.

**Mọi solution thắng VN-legal IR đều fine-tune:**
- **Zalo Legal 2021 winners:** train trên MS-MARCO + SQuAD2 + 80% Zalo train, synthetic query pre-train, hard negatives từ BGE-M3.
- **ALQAC 2024 Task-1 (CÙNG metric Macro-F2):** fine-tuned Bi-Encoder → Cross-Encoder rerank + negative mining; top teams **ensemble + iterative self-training trên bge-m3**.
- **SoICT Hackathon 2024 Legal Retrieval:** bi-encoder fine-tune + cross-encoder + hard-negative mining.
- **arXiv 2412.00657 (synthetic data):** off-the-shelf 64 → fine-tune 79-84 MRR@10.

## Đường đột phá (recipe SOTA, đã chứng minh)

### Bước 1 — Synthetic training data từ corpus (mentor "tất yếu")
- Với mỗi điều trong corpus → LLM (Qwen2.5-7B, hợp lệ) sinh 1-5 câu hỏi theo "khía cạnh" (aspect-guided) → cặp `(câu hỏi synthetic, điều đúng)`.
- Lọc bằng BGE-M3 top-40 + bỏ câu tự-tham-chiếu (paper: 620K→507K cặp từ 140K passage).
- Quy mô ta: ~90K điều (corpus cũ) → ~100-300K cặp. Sinh trên Kaggle GPU.

### Bước 2 — Hard negative mining
- Negatives = top-K điều SAI mà BGE-M3/retriever hiện tại trả về cho mỗi câu (khó phân biệt → tín hiệu mạnh nhất). Paper: 7 neg/query (bi-encoder), 15 (ColBERT).

### Bước 3 — Fine-tune
- **Reranker trước (đòn bẩy cao nhất, rẻ nhất):** fine-tune BGE-reranker-v2-m3 bằng InfoNCE/cross-entropy trên (query, pos, hard-negs). Đây là tầng quyết định thứ hạng Điều cuối → đánh đúng gap DOCS>ARTICLES.
- Sau đó (nếu còn lực): fine-tune bi-encoder embedding, hoặc train ColBERT (late-interaction — paper cho điểm cao nhất ở cấp passage).

### Bước 4 — LLM-rerank listwise (tùy chọn, KHÔNG cần train)
- Cho LLM xếp hạng CẢ danh sách ứng viên có điểm (KHÁC binary CÓ/KHÔNG đã FAIL). Hybrid + LLM-rerank → recall tới ~0.9+ (per SOTA 2025).

### Cheap wins thử TRƯỚC khi train (1-2 lần Phase A)
1. **`RERANK_MAX` 256 → 512/1024** — điều luật dài hết bị cụt; reranker thấy đủ Khoản → định vị Điều đúng hơn. **Đánh thẳng gap DOCS>ARTICLES.** Rẻ nhất, thử đầu tiên.
2. Thử `jina-reranker-v2` / `BAAI/bge-reranker-v2-gemma` (vẫn off-the-shelf nhưng mạnh hơn) thay BGE-reranker-v2-m3.

## Reality check

- Gap 0.56→0.69 = **0.13**, là gap fine-tune vs off-the-shelf, KHÔNG phải tinh chỉnh. Đã cạn đường tinh chỉnh (vòng 5-9: corpus, cutoff, validity, phễu — tất cả ±0.01-0.05).
- Fine-tune reranker = vài ngày + GPU + cần eval set local để khỏi bay mù qua leaderboard.
- Thứ tự ROI: **(a) RERANK_MAX cheap fix → (b) reranker mạnh hơn off-the-shelf → (c) fine-tune reranker trên synthetic data → (d) ColBERT/bi-encoder fine-tune.**

## Next Actions

1. **Revert corpus cũ** (bỏ vbpl) — chốt baseline 0.5608.
2. **Cheap test RERANK_MAX=512** trên corpus cũ → đo leaderboard (đánh gap DOCS>ARTICLES).
3. **Tạo eval gold local** (~50-100 câu verify tay từ stage1) — BẮT BUỘC trước khi fine-tune (validate offline).
4. **Sinh synthetic data + fine-tune reranker** trên Kaggle (recipe Bước 1-3).

## Unresolved (phải chốt trước khi đổ công)

- **Luật giải có CHO fine-tune không?** Rule "open model <14B, released < 2026-03-01" — model tự fine-tune có hợp lệ, hay phải checkpoint đã publish? **Sai luật = mất trắng.** Check ĐẦU TIÊN.
- Quota Kaggle GPU còn đủ cho sinh synthetic (~100-300K) + fine-tune (5-9 epochs)?
- Mục tiêu: thắng leaderboard / ship sản phẩm / học? → quyết mức đầu tư.

## Sources
- [Improving Vietnamese Legal Document Retrieval using Synthetic Data (arXiv 2412.00657)](https://arxiv.org/html/2412.00657v1)
- [Optimizing Legal Document Retrieval in Vietnamese (arXiv 2507.14619)](https://arxiv.org/pdf/2507.14619)
- [NOWJ1@ALQAC 2023 (arXiv 2309.09070)](https://arxiv.org/pdf/2309.09070)
- [Multi-stage IR for Vietnamese Legal Texts (arXiv 2209.14494)](https://arxiv.org/pdf/2209.14494)
- [Zalo AI 2022 winning solution](https://github.com/Telegram-Zalo/zac2022-e2e-qa)
- [Pre-training, Fine-tuning, Re-ranking: 3-Stage Legal QA (arXiv 2412.19482)](https://arxiv.org/pdf/2412.19482)
