# Research: Vì sao F2 nghẽn ~0.39? Workflow/data yếu chỗ nào?

**Ngày:** 2026-06-09 15:06 · **Loại:** Diagnosis (data nội bộ + research ngoài) · **Verdict:** Nút thắt = **ĐỘ PHỦ CORPUS** (32 văn bản quá ít) + **khoảng cách câu-đời-thường ↔ điều-luật**. KHÔNG phải cutoff, KHÔNG phải LLM.

## Kết quả leaderboard (sweep cutoff)
| Config | điều/câu | F2 |
|---|---|---|
| t3m3 | 2.14 | **0.3877** ⬅ đỉnh |
| t5m4 | 3.22 | 0.3867 |
| t6m6 | 4.42 | 0.3615 |
| t8m10 | 6.95 | 0.2902 |

**Đỉnh ở cấu hình chặt nhất, nới = tụt** → candidate hạng 2+ phần lớn là RÁC. Cutoff đã gần tối ưu (~2-3 điều/câu). Cải thiện so bài đầu (0.317→0.39) chủ yếu nhờ bỏ `min_score=0` + weighted RRF, nhưng **đã chạm trần của corpus/retrieval hiện tại**.

## Bằng chứng nội bộ (retrieved.json, 2000 câu)
- **Điểm rerank top-1: median = −1.05; chỉ 42.7% câu > 0; 33.5% câu < −3.** → **~57% câu KHÔNG có điều luật nào khớp tự tin**. Đây là trần điểm thật.
- Câu hỏi **gần như không cite số hiệu** (2/2000) → thuần tình huống đời thường → BM25/weighted-RRF ít tác dụng (không có "Điều 5"/"59/2020" để khớp chính xác).
- Câu hỏi dài vừa (median 37 từ) → đủ thông tin, vấn đề không phải câu hỏi quá ngắn.
- Corpus: **4.416 điều / 32 văn bản** — quá hẹp.

## Root cause (xếp theo tác động)

### 1. ĐỘ PHỦ CORPUS — nghi phạm số 1 (data)
32 văn bản hand-pick không phủ nổi 2000 câu SME (lao động, thuế, BHXH, hóa đơn, đất đai, SHTT, xây dựng, môi trường, PCCC, an toàn thực phẩm, giao thông, hành chính... + hàng trăm Nghị định/Thông tư hướng dẫn). Nếu điều gold KHÔNG có trong corpus → recall = 0 cho câu đó, vĩnh viễn. 57% câu "không khớp tự tin" khớp với giả thuyết này.

**Dữ liệu có sẵn LỚN HƠN NHIỀU (ta đang bỏ phí):**
- `tmquan/vbpl-vn` ta đang dùng có **158K văn bản** nhưng ta **lọc còn 32** (allowlist). Bỏ allowlist → dùng rộng = tăng phủ khổng lồ.
- **VLQA** (arXiv 2507.19995): ~**60.000 điều luật** Việt + 3129 câu QA — corpus statutory toàn diện.
- `VLSP2025-LegalSML/legal-pretrain` (HF): văn bản luật chính thống đã tiền xử lý.
- Bộ 127K văn bản vbpl.vn (instruction dataset) — cùng nguồn.

### 2. Khoảng cách "đời thường ↔ văn phong luật" — nghi phạm số 2 (retrieval)
Câu hỏi: "Công ty nhỏ thuê mặt bằng được hỗ trợ bao lâu?" vs điều luật: "Điều 11. Hỗ trợ mặt bằng sản xuất...". Embedding/reranker bắt yếu → top-1 logit âm. **HyDE** (Qwen viết lại câu hỏi sang văn phong luật trước khi embed) nhắm thẳng vào đây.

### 3. KHÔNG phải nút thắt
- Cutoff: đã tối ưu (sweep chứng minh).
- LLM/answer: QA chưa chấm; answer không ảnh hưởng IR F2 ngoài việc chứa "Điều X".
- Model embedding: AITeamVN/Vietnamese_Embedding (BGE-M3, 1024-token) ổn; bkai vietnamese-bi-encoder mạnh trên Zalo nhưng giới hạn 256 token (kém cho điều luật dài) → KHÔNG đổi.

## Trả lời câu hỏi vbpl.vn
- **Không có API công khai** cho vbpl.vn. `tmquan/vbpl-vn` (158K docs) CHÍNH LÀ bản scrape vbpl.vn sẵn rồi.
- → "Quay lại vbpl.vn" giá trị thấp cho phần lớn; **ROI cao hơn = dùng nhiều hơn data ĐÃ có** (bỏ allowlist 32, hoặc VLQA 60K).
- Chỉ scrape vbpl.vn trực tiếp cho **văn bản cụ thể bị null/lỗi markdown** trong snapshot HF (vd 04/2007/QH12 TNCN trước đây) — ít, làm sau.

## So sánh ALQAC (để chỉnh kỳ vọng)
- ALQAC 2024 best F2 ≈ **0.87** NHƯNG **BTC cấp sẵn corpus + train + phạm vi hẹp (vài trăm điều)**. Phương pháp top: parse câu tiếng Việt + BM25 + sentence-transformer.
- Cuộc thi này: **tự thu thập corpus, 2000 câu rộng, không train** → khó hơn HẲN. 0.38 thấp nhưng **0.5-0.6 khả thi** nếu vá coverage + HyDE. Đừng kỳ vọng 0.87.

## Khuyến nghị (ưu tiên giảm dần)

1. **MỞ RỘNG CORPUS mạnh (data) — đòn bẩy lớn nhất.**
   - Phương án A (nhanh, dùng data sẵn): bỏ/nới `SME_LAW_ALLOWLIST` → ingest TẤT CẢ `luật/bộ luật/nghị định/thông tư` từ `tmquan/vbpl-vn` (hoặc lọc rộng theo chủ đề SME mở rộng ~vài trăm-vài nghìn văn bản). Re-embed.
   - Phương án B (phủ tốt nhất): nạp **VLQA ~60K điều** làm corpus (kiểm tra license/format).
   - Cảnh báo precision: corpus to → reranker + cutoff chặt vẫn lọc được; coverage mới đặt trần recall. Rủi ro: embed lâu hơn (4.4K→60K điều: numpy vẫn OK ~vài trăm MB; embed trên Kaggle ~1-2h, làm 1 lần).
2. **HyDE** (retrieval): offline Qwen viết lại 2000 câu sang văn phong luật → embed bản viết lại. Nhắm 57% câu low-confidence.
3. **Đo lại** sau mỗi thay đổi qua leaderboard (giữ cutoff t3m3/t5m4).
4. (Sau, nặng) fine-tune embedder bằng synthetic query (arXiv 2412.00657, 2507.14619) — gain tốt nhưng tốn công, để cuối.

## Câu hỏi mở
1. **License/format VLQA + VLSP-LegalSML** — dùng làm corpus được không (corpus tự thu thập = hợp lệ; nhưng QA pairs của VLQA KHÔNG dùng để train, tránh "external data" §9). Cần verify.
2. Mở rộng corpus tới đâu là đủ mà không loãng/chậm? (vài trăm vb chủ đề SME vs toàn bộ statutory 60K). Nên thử A trước, đo, rồi quyết.
3. Leaderboard "Điểm" 0.3877 là ARTICLES_F2 hay trung bình ARTICLES+DOCS? Bấm icon "Kết quả chi tiết" 1 bản để biết DOCS_F2 vs ARTICLES_F2 → nếu DOCS cao hơn nhiều ARTICLES = đúng luật/sai điều (vấn đề rank trong luật); nếu cả 2 thấp = sai luật (coverage). Định hướng chính xác hơn.

## Nguồn
- [ALQAC 2024 summary (IEEE)](https://ieeexplore.ieee.org/document/11063484/) · [Top2 ALQAC2024 LLM](https://www.researchgate.net/publication/387913243)
- [VLQA dataset (60K điều)](https://arxiv.org/html/2507.19995v1) · [VLSP2025-LegalSML/legal-pretrain](https://huggingface.co/datasets/VLSP2025-LegalSML/legal-pretrain)
- [Improving VN Legal Retrieval w/ Synthetic Data](https://arxiv.org/html/2412.00657v1) · [Semi-Hard Negative Mining](https://arxiv.org/pdf/2507.14619)
- [AITeamVN/Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding) · [bkai vietnamese-bi-encoder](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder)
- [vbpl.vn portal](https://vbpl.vn/Pages/portal.aspx) (không có API công khai)
