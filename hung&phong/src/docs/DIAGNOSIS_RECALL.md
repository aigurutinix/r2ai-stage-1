# Chẩn đoán nút thắt: Recall trần ~0.57 (2026-06-13)

> Rà soát toàn bộ 11 lần điểm THẬT từ BTC + corpus, sau khi nhận ra mình mới chỉ nhìn
> F2 tổng hợp mà chưa mổ xẻ nguyên nhân. Đây là bản ghi để không lặp lại sai lầm.

## 1. Lịch sử điểm THẬT (từ scoring_result BTC, không phải pseudo-gold)

| Version | ART_F2 | ART_P | ART_R | DOCS_F2 | DOCS_R | Ghi chú |
|---------|-------:|------:|------:|--------:|-------:|---------|
| v5  | 0.409 | 0.267 | 0.519 | 0.456 | 0.567 | |
| v7  | 0.439 | 0.300 | 0.549 | 0.484 | 0.597 | mở corpus |
| **v8**  | 0.4475 | 0.327 | 0.554 | **0.510** | **0.623** | ⭐ DOCS đỉnh |
| v9  | 0.4511 | 0.280 | **0.5737** | 0.461 ⬇ | 0.573 | multi-query: R đỉnh nhưng DOCS tụt |
| **v10** | **0.4609** | **0.388** | 0.525 | 0.455 ⬇ | 0.517 | ⭐ ART đỉnh (nhờ judge tăng P) |
| v11 | 0.301 💀 | 0.220 | 0.369 | 0.430 | 0.483 | bỏ reranker → sập toàn diện |

## 2. Hai điều bị bỏ sót (do chỉ nhìn ART_F2)

1. **DOCS_F2 đạt đỉnh ở v8 (0.510), v9/v10 làm TỤT xuống 0.46.**
   Multi-query + judge vứt cả văn bản đúng, không chỉ điều nhiễu. Nếu điểm tổng có
   trọng số DOCS, **v8 có thể mới là bản tốt nhất tổng thể.**

2. **Recall thật KẸT TRẦN ~0.52–0.57 suốt v5→v10.** Leader ~0.725. Thiếu ~0.15.

## 3. Bằng chứng: nút thắt là RECALL RETRIEVAL, không phải lọc

- **v3 nhồi 10 điều/câu → recall thật chỉ 0.667.** Kể cả lấy rất rộng, **1/3 gold vẫn
  nằm ngoài top-10 retrieval.** Lọc/judge giỏi mấy cũng không tạo ra gold đã bị sót.
- **v10 chứng minh precision lên được 0.388** → khâu lọc KHÔNG phải vấn đề.
- ⇒ Mọi công sức vào judge (v10, v11) là **tối ưu nhầm chỗ**: mài giũa khâu lọc trên
  tập ứng viên vốn đã thiếu gold.

## 4. Loại trừ giả thuyết "corpus thiếu data"

- Corpus: **16,197 văn bản / 196,842 chunk / 136,527 điều.** Các luật SME cốt lõi đều có
  (DN 59/2020, SME 04/2017, Lao động 45/2019, Dân sự 91/2015, SHTT 50/2005, Thương mại
  36/2005, các NĐ 80/2021, 39/2018, 01/2021, 168/2025...).
- Chỉ **2/2000 câu** nhắc số hiệu văn bản cụ thể → câu hỏi gần như toàn bộ là **mô tả
  tình huống**, không trích dẫn. ⇒ Retrieval thuần **semantic** (khó hơn nhiều).
- Vùng câu hỏi **rất rộng**: 1018 văn bản khác nhau cho 2000 câu. Top: Bộ luật Lao động,
  Bộ luật Dân sự, Luật Thương mại, Luật DN, Luật SHTT — đều **hợp lý** cho SME (hợp đồng,
  nhân sự, mua bán, thương hiệu). Không phải trôi dạt.
- ⚠ Lỗi phụ phát hiện: `documents.parquet` có 79 mã trùng → vài title export sai (vd
  91/2015/QH13 hiển thị nhầm "Nghị quyết giám sát"). **Không ảnh hưởng submission** (bài
  nộp lấy title đúng từ Qdrant payload). Nên dọn lại file export cho sạch.

## 5. Kết luận: recall trần do RANKING (gold có trong corpus nhưng rank thấp)

Câu hỏi tình huống dùng **ngôn ngữ đời thường**, điều luật dùng **ngôn ngữ pháp lý formal**
→ khoảng cách semantic làm embedding xếp gold ra ngoài top-K.

## 6. Hướng tấn công recall (thay vì tiếp tục mài judge)

| Ưu tiên | Hướng | Lý do |
|---------|-------|-------|
| 🥇 | **HyDE / query expansion**: Qwen diễn giải câu hỏi tình huống → đoạn văn pháp lý giả định → embed cái đó | Thu hẹp gap "đời thường ↔ pháp lý" — đánh đúng gốc |
| 🥈 | **Fine-tune reranker** trên cặp (câu hỏi, điều) tự sinh (BTC CHO PHÉP finetune) | Reranker domain-specific đẩy gold lên top |
| 🥉 | **Tăng top-K vào reranker** (40-60) rồi rerank kỹ — GIỮ reranker | Cast wider net, rerank chọn lại. KHÔNG bỏ reranker (v11 đã chứng minh sập) |
| | **Ensemble đa truy vấn** (gốc + HyDE + keyword) hợp nhất | Mỗi cách bắt phần gold khác nhau |

## 7. Quyết định về v12 đang chạy

v12 (reranker + judge inline) vẫn hợp lệ — sẽ cải thiện precision như v10 đồng thời giữ
recall multi-query như v9 → kỳ vọng ART_F2 ~0.47–0.49. NHƯNG vẫn đụng trần recall ~0.57.
**v13 phải là đòn HyDE/query-expansion để phá trần recall.**
