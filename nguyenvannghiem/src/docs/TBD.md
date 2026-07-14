# Danh sách mục đã xác nhận

Tài liệu này liệt kê trạng thái các điểm cần xác nhận. Hầu hết đã hoàn tất.

---

## A. Quan trọng

1. ~~**Repo HuggingFace của Qwen3-8B-AWQ.**~~ ✅ DONE
   - Repo: https://huggingface.co/Qwen/Qwen3-8B-AWQ
   - vLLM: v0.23.0

2. ~~**Danh sách file submission cuối.**~~ ✅ DONE
   - Bản nộp cuối: `submission_3_1_combo_ck23k_f01_a07.json`

3. ~~**Nộp bản có `answer` đầy đủ.**~~ ✅ DONE
   - Bản cuối có answer đầy đủ (optimized prompt, anti-hallucination, 99% có "Căn cứ pháp lý").

---

## B. Link tải dữ liệu

4. ~~**Data download links.**~~ ✅ DONE
   - HuggingFace dataset: `NghiemAbe/r2ai-legal-data` (private)
   - Reranker: `NghiemAbe/r2ai-reranker-v2-ck8000` (private)

---

## C. Link tải / repo model

5. ~~**Repo HuggingFace các model.**~~ ✅ DONE
   - Qwen3-8B-AWQ: https://huggingface.co/Qwen/Qwen3-8B-AWQ
   - Vietnamese_Embedding_v2: https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2
   - Vietnamese_Reranker: https://huggingface.co/AITeamVN/Vietnamese_Reranker
   - Reranker fine-tuned ck-8000: có sẵn tại `models/reranker_finetuned_v2_ck8000/`
   - LoRA classifier ck-23000: có sẵn tại `models/legal_classifier_lora_v2_ck23000/`

---

## D. Chi tiết kỹ thuật

6. **Pin pip versions** — chưa pin. Nếu BTC yêu cầu, chạy `pip freeze` từ môi trường thực tế.

---

## E. Đóng gói submission

7. Zip folder `r2ai_submission/` để nộp.

---

## Tóm tắt

| # | Hành động | Trạng thái |
|---|-----------|------------|
| 1 | Repo + flags vLLM Qwen3-8B-AWQ | ✅ Done |
| 2 | Chốt file submission cuối | ✅ Done — `submission_3_1_combo_ck23k_f01_a07.json` |
| 3 | Bản có answer đầy đủ | ✅ Done |
| 4 | Data download links | ✅ Done — HuggingFace |
| 5 | Repo HuggingFace models | ✅ Done |
| 6 | Pin pip versions | Pending (optional) |
| 7 | Format đóng gói | Zip |
