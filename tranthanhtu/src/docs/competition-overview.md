# Đề bài: Truy hồi & Hỏi đáp Văn bản Pháp luật Tiếng Việt

> Vietnamese Legal Information Retrieval & Question Answering
> Leaderboard: http://leaderboard.aiguru.com.vn/

## 1. Bối cảnh

DN nhỏ và vừa (SME) Việt Nam khó tra cứu/áp dụng quy định pháp lý (Luật Doanh nghiệp, thuế, lao động, hợp đồng...). So với tiếng Anh/Nhật/Trung, tài nguyên Vietnamese Legal NLP còn hạn chế. Cuộc thi thúc đẩy xây hệ thống AI **tìm điều luật liên quan** và **tự trả lời câu hỏi pháp lý có căn cứ**.

## 2. Hai nhiệm vụ

### 2.1 Truy hồi thông tin (IR)
Cho tập câu hỏi Q và kho điều luật A, xác định tập con A′ ⊂ A "liên quan" tới mỗi câu hỏi. Một điều luật "liên quan" nếu câu hỏi có thể được trả lời Có/Không suy ra từ ý nghĩa điều luật đó.

### 2.2 Hỏi đáp pháp luật (QA)
Dựa trên điều luật đã truy hồi, sinh câu trả lời cho câu hỏi — không chỉ tìm đúng căn cứ mà còn hiểu & suy luận nội dung.

## 3. Mục tiêu hệ thống (5 yêu cầu)

1. **Tra cứu chính xác** — retrieval & grounding đúng từ kho dữ liệu.
2. **Hỏi đáp tiếng Việt** — hiểu ngôn ngữ tự nhiên, tình huống pháp lý.
3. **Dẫn nguồn** — trích Điều/Khoản/văn bản, hiển thị tham chiếu kiểm chứng được.
4. **Tư vấn sơ bộ + cảnh báo giới hạn** — hướng dẫn + nhắc rủi ro tuân thủ + disclaimer.
5. **Kiểm soát sai lệch** — không bịa điều luật/nguồn không tồn tại.

## 4. Dữ liệu

- BTC **chỉ cấp test set** (câu hỏi). **KHÔNG** cấp train/dev/corpus.
- Bộ đáp án chuẩn: BTC giữ kín.
- Đội thi **tự thu thập** corpus (văn bản luật, nghị định, thông tư từ nguồn chính thống; open dataset Legal NLP).
- **Nguồn dùng (BTC recommend, HuggingFace):**
  - `tmquan/vbpl-vn` — văn bản pháp luật vbpl.vn (158k docs) — **corpus chính**
  - `tmquan/phapdien-moj-gov-vn` — pháp điển (64k articles)
  - `tmquan/anle-toaan-gov-vn` — án lệ (1.9k)

**Định dạng đầu vào** (test set):
```json
{ "id": 1, "question": "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào...?" }
```

## 5. Định dạng nộp bài

File **`results.json`** (UTF-8), nén **zip phẳng** (results.json ở gốc, không thư mục con):
```json
[
  {
    "id": 1,
    "question": "...",
    "answer": "... văn bản trả lời, có nêu 'Điều X' ...",
    "relevant_docs": ["<mã văn bản>|<tên văn bản>"],
    "relevant_articles": ["<mã văn bản>|<tên văn bản>|<điều>"]
  }
]
```
- `<tên văn bản>` = Loại văn bản + Mã văn bản + Trích yếu (lưu ý: ví dụ gold lại bỏ mã trong tên → khóa join chính là **mã văn bản**).
- Thiếu câu / sai định dạng → bài không hợp lệ.
- Tên file **bắt buộc** `results.json`.

```bash
zip submission.zip results.json   # Linux/macOS
```

## 6. Phương pháp đánh giá

Dùng **macro-average** (tính cho từng câu rồi lấy trung bình).

### 6.1 IR — Precision / Recall / **F2 macro**
- Grader **tự động trích pattern "Điều X" từ trường `answer`**, so với đáp án (định danh `mã|tên|Điều X`, chuẩn hóa "Điều X").
- `F2 = (5 × P × R) / (4 × P + R)` → **recall nặng gấp ~2× precision**.
- Chấm cả 2 mức: **articles** (điều luật) và **docs** (văn bản).

### 6.2 QA — 5 nhóm tiêu chí
1. Căn cứ chính xác pháp luật (tự động) — tỷ lệ câu có ≥1 điều đúng.
2. Tính chính xác nội dung *(thủ công, hiện 0.0)*
3. Tính đầy đủ & toàn diện *(thủ công, hiện 0.0)*
4. Tính thực tiễn – khả năng áp dụng *(thủ công, hiện 0.0)*
5. Tính rõ ràng – dễ hiểu *(thủ công, hiện 0.0)*

- **LLM-as-a-Judge** chấm tự động + **chuyên gia** chấm độc lập tập con.
- QA chỉ chấm bài được **promote** lên leaderboard, **mỗi tuần 1 lần**.

> Tham chiếu: ALQAC 2024 best retrieval **F2 ≈ 0.87** (nhưng ALQAC có cấp corpus + train → cuộc thi này khó hơn, điểm thấp hơn).

## 7. Quy định mô hình (BẮT BUỘC)

| Ràng buộc | Yêu cầu |
|---|---|
| Kích thước | **< 14B** tham số |
| Thời điểm | công bố **trước 01/03/2026** (giờ VN) |
| Giấy phép | **mã nguồn mở**, trọng số tải tự do cho nghiên cứu |
| Cấm | **mô hình đóng** (GPT-4o, Gemini, ...) |
| Dữ liệu | không dùng "dữ liệu bên ngoài trong bất kỳ bước xử lý nào" (xem §9) |

→ Phải ghi cách lấy mô hình trong **working-notes paper** (điều kiện công nhận kết quả).

## 8. Quy định nộp & mốc thời gian

- Tối đa **10 bài/ngày**; Vòng Riêng (Private) tối đa **5 bài tổng**.
- QA: tự chọn 1 bài promote lên leaderboard mỗi kỳ.

| Mốc | Ngày |
|---|---|
| Khai mạc, phát hành test set | 03/06/2026 |
| Đóng cổng nộp bài | 30/06/2026 |
| Công bố Top 10 → DemoDay | 05/07/2026 |
| DemoDay, chung cuộc | 11/07/2026 |

(Tất cả 23:59 giờ VN, UTC+07:00.)

## 9. Câu hỏi mở (CẦN HỎI BTC)

1. **Mâu thuẫn dữ liệu:** Overview ghi "không dùng dữ liệu bên ngoài trong bất kỳ bước xử lý nào" nhưng Data ghi "đội tự thu thập corpus". Hai điều khoản đá nhau — được crawl/HF dataset tới mức nào?
2. IR scoring: full identifier (`mã|tên`) ghép từ `relevant_articles` mình nộp, hay chỉ parse từ `answer` text?
3. Khớp `<tên văn bản>` là fuzzy hay chỉ join trên `mã văn bản`?
