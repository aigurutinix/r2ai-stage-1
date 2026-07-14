# Cuộc thi Truy hồi & Hỏi đáp Văn bản Pháp luật Tiếng Việt

> **Vietnamese Legal Information Retrieval & Question Answering**

Tài liệu này ghi lại thể lệ, mục tiêu, phương pháp đánh giá và dữ liệu của cuộc thi —
làm kim chỉ nam cho việc xây dựng hệ thống trong repo này.

---

## 1. Tổng quan bài toán

Nhằm thúc đẩy nghiên cứu và phát triển trong lĩnh vực pháp luật, cuộc thi tổ chức về
**Truy hồi và Hỏi đáp Văn bản Pháp luật Tiếng Việt**. Cuộc thi gồm hai nhiệm vụ liên kết:

### 1.1 Truy hồi thông tin (Legal Information Retrieval)
Với một câu truy vấn, hệ thống cần truy hồi các điều luật **liên quan** làm căn cứ.
Một điều luật được coi là liên quan nếu câu truy vấn có thể được trả lời (kể cả dạng
Có/Không) suy ra từ ý nghĩa của điều luật đó.

### 1.2 Hỏi đáp pháp luật (Legal Question Answering – QA)
Dựa trên các điều luật đã truy hồi, hệ thống cần **sinh câu trả lời** cho câu hỏi pháp lý
tương ứng.

> **Mục tiêu chung:** xây dựng hệ thống AI không chỉ tìm đúng căn cứ pháp luật mà còn
> **hiểu và suy luận** nội dung pháp lý để hỗ trợ trả lời tự động cho người dùng.

---

## 2. Mục tiêu cuộc thi

Các đội thi cần xây dựng hệ thống AI có khả năng:

1. **Tra cứu pháp lý chính xác**
   - Tra cứu điều khoản trong Luật Doanh nghiệp và các văn bản liên quan đến SME.
   - Tìm kiếm, truy xuất thông tin pháp luật chính xác từ kho dữ liệu được cung cấp.
   - Ưu tiên khả năng **retrieval** và **grounding** chính xác.

2. **Hỏi đáp pháp lý bằng tiếng Việt**
   - Hiểu ngôn ngữ tự nhiên tiếng Việt.
   - Hỏi đáp các tình huống pháp lý thường gặp.

3. **Dẫn nguồn điều luật**
   - Trích dẫn điều / khoản / văn bản liên quan.
   - Hiển thị rõ nguồn tham chiếu để đảm bảo khả năng kiểm chứng.
   - Hạn chế trả lời không có căn cứ pháp lý.

4. **Tư vấn sơ bộ & cảnh báo giới hạn**
   - Đưa ra hướng dẫn pháp lý sơ bộ cho người dùng.
   - Nhắc nhở các rủi ro tuân thủ trong tình huống phổ biến.
   - Hiển thị cảnh báo giới hạn của AI.

5. **Kiểm soát nội dung sai lệch**
   - Hạn chế AI sinh thông tin sai lệch.
   - Tránh bịa điều luật hoặc nguồn tham chiếu không tồn tại.
   - Tăng độ tin cậy của câu trả lời dựa trên dữ liệu được cung cấp.

---

## 3. Các mốc thời gian quan trọng

| Ngày | Sự kiện |
|------|---------|
| **03/06/2026** | Khai mạc, phát hành tập dữ liệu kiểm thử (test set). |
| **30/06/2026** | Chính thức đóng cổng hệ thống — các đội phải hoàn thành nộp bài. |

> **Lưu ý:** Vui lòng đưa thông tin về cách thức lấy mô hình vào bài báo.

---

## 4. Phương pháp đánh giá

Hiệu năng hệ thống được đánh giá bằng **các chỉ số tự động và thủ công**. Sử dụng
**trung bình macro** (chỉ số được tính cho từng truy vấn rồi lấy trung bình) để tính
điểm cuối cùng.

### 4.1 Truy hồi thông tin

Đánh giá bằng **Precision**, **Recall** và **F2 macro** (macro-average).

**Cách so khớp điều luật (Retrieval):**
Hệ thống chấm điểm **so sánh trực tiếp** các phần tử trong trường `relevant_docs` /
`relevant_articles` của bài nộp với các điều luật trong đáp án. Định danh đầy đủ có dạng
`law_id | tên văn bản | Điều X`, được **chuẩn hóa về `Điều X`** trước khi so khớp.

> ⚠️ **Phân biệt với phần QA (4.2):** Phần Retrieval này chấm trên `relevant_articles`/
> `relevant_docs`. Việc *trích pattern `Điều X` từ trường `answer`* là **cơ chế riêng của
> tiêu chí QA "Căn cứ chính xác pháp luật"** (xem 4.2, mục 1) — KHÔNG phải của Retrieval.

| Chỉ số | Công thức |
|--------|-----------|
| **Precision** | trung bình của (số điều luật truy hồi **đúng**) / (số điều luật **đã truy hồi**) trên mỗi truy vấn |
| **Recall** | trung bình của (số điều luật truy hồi **đúng**) / (số điều luật **liên quan**) trên mỗi truy vấn |
| **F2** | `F2 = (5 × Precision × Recall) / (4 × Precision + Recall)` |

> F2 ưu tiên **Recall** hơn Precision → khuyến khích bao phủ đủ căn cứ pháp lý.

### 4.2 Hỏi đáp pháp luật

Bộ tiêu chí gồm **5 nhóm**:

1. **Căn cứ chính xác pháp luật** — tỷ lệ câu hỏi có ít nhất một điều luật được trích
   xuất đúng **từ trường `answer`** của bài nộp. *(Đánh giá tự động — scorer tìm pattern
   `Điều X` trong `answer` rồi so với `relevant_articles` của đáp án.)*
2. **Tính chính xác nội dung** — mức độ chính xác so với quy định pháp luật.
3. **Tính đầy đủ & toàn diện** — câu trả lời bao quát đủ các khía cạnh liên quan.
4. **Tính thực tiễn – khả năng áp dụng** — khả năng áp dụng thực tế trong bối cảnh pháp lý.
5. **Tính rõ ràng – dễ hiểu** — diễn đạt rõ ràng cho người đọc không chuyên.

#### 4.2.1 Đánh giá tự động (LLM-as-a-Judge)
Dùng các mô hình ngôn ngữ lớn (LLMs) đóng vai **giám khảo tự động** chấm theo bộ tiêu chí
5 nhóm. Với mỗi câu trả lời, LLM được cung cấp:
- câu hỏi,
- câu trả lời tham chiếu,
- các điều luật căn cứ,
- câu trả lời của hệ thống cần đánh giá.

LLM chấm điểm từng nhóm tiêu chí kèm lý do giải thích cụ thể.

#### 4.2.2 Con người đánh giá
Một tập con câu trả lời được **chuyên gia pháp luật** đánh giá độc lập theo cùng bộ tiêu
chí 5 nhóm, kèm nhận xét ưu/nhược điểm. Điểm Human Evaluation cuối cùng là **trung bình
cộng** của các chuyên gia tham gia.

> **Lưu ý:** 4 chỉ số thủ công (Tính chính xác nội dung, Tính đầy đủ & toàn diện, Tính
> thực tiễn, Tính rõ ràng) hiện đặt giá trị **0.0** và sẽ được cập nhật sau khi ban giám
> khảo hoàn thành đánh giá.

---

## 5. Dữ liệu cuộc thi

Ban Tổ chức **chỉ cung cấp tập dữ liệu kiểm thử (test set)**: tập câu hỏi pháp lý, dùng
làm căn cứ chấm điểm. **Không cung cấp** tập huấn luyện (train) hay tập phát triển (dev).

- **Bộ đáp án chuẩn** được giữ kín, chỉ phục vụ chấm điểm để đảm bảo khách quan, công bằng.
- Các đội **toàn quyền chủ động** thu thập và khai thác dữ liệu, bao gồm:
  - Văn bản pháp luật, thông tư, nghị định từ nguồn chính thống.
  - Dữ liệu liên quan doanh nghiệp SME (thuế, lao động, hợp đồng, v.v.).
  - Các tập dữ liệu mở (open dataset) cho bài toán Legal NLP.
  - Mọi nguồn dữ liệu hợp pháp khác.

Cuộc thi khuyến khích sáng tạo trong toàn bộ quy trình:
- Thu thập và tiền xử lý dữ liệu.
- Thiết kế chiến lược chia nhỏ & biểu diễn dữ liệu (chunking, embedding, ...).
- Tối ưu hóa cơ chế truy hồi thông tin.
- Xây dựng pipeline AI hoàn chỉnh phù hợp với kiến trúc riêng.

### 5.1 Định dạng dữ liệu đầu vào

Mỗi mẫu gồm:

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `id` | integer | Mã định danh câu hỏi. |
| `question` | string | Nội dung câu hỏi pháp lý. |

```json
{
  "id": 1,
  "question": "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa?"
}
```

### 5.2 Định dạng bài nộp (QUAN TRỌNG)

- Nộp **một file duy nhất tên `results.json`**, nén thành **file `.zip` phẳng** (KHÔNG
  bọc thư mục con — zip trực tiếp file `results.json`).
- File JSON là **một mảng** chứa kết quả cho **TOÀN BỘ câu hỏi** của test set. Câu hỏi
  **thiếu hoặc sai định dạng → tính là dự đoán không hợp lệ**.
- **Giới hạn số lần nộp mỗi ngày** (chi tiết công bố trên Dashboard) → tránh dò đáp án.

Mỗi phần tử gồm 5 trường:

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `id` | integer | Mã định danh câu hỏi (khớp test set). |
| `question` | string | Nội dung câu hỏi. |
| `answer` | string | Câu trả lời. Scorer **trích pattern `Điều X` từ đây** để chấm tiêu chí QA "Căn cứ chính xác". |
| `relevant_docs` | string[] | Mỗi phần tử: `<mã văn bản>\|<tên văn bản>`. |
| `relevant_articles` | string[] | Mỗi phần tử: `<mã văn bản>\|<tên văn bản>\|<điều>`. |

#### ⚠️ Công thức `<tên văn bản>` (dễ sai — ảnh hưởng điểm)

> `<tên văn bản>` = **Loại văn bản + Mã văn bản + Trích yếu**

Ví dụ chuẩn của BTC:
```
04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4
```
Trong đó: Loại = `Luật`, Mã = `04/2017/QH14`, Trích yếu = `Luật Hỗ trợ doanh nghiệp nhỏ và vừa`.

> **Lưu ý hiện trạng repo:** bài nộp hiện dùng dạng `04/2017/QH14|Luật Hỗ trợ doanh nghiệp
> nhỏ và vừa|Điều 4` (tên **thiếu mã văn bản** trong phần `<tên văn bản>`). Vì ART_F2 ≠ 0
> nên scorer nhiều khả năng match theo `law_id` (phần đầu trước `|`) + `Điều X` và **bỏ qua
> tên văn bản** khi chuẩn hóa. Tuy vậy **nên thử tuân thủ đúng công thức** để loại trừ rủi
> ro mất điểm — cần kiểm chứng bằng 1 lần nộp đối chứng.

### 5.3 Được phép finetune mô hình

BTC **cho phép finetune mô hình**. Không có train/dev set chính thức, nhưng đội thi được
toàn quyền thu thập dữ liệu hợp pháp để huấn luyện. Vui lòng **ghi cách thức lấy/huấn luyện
mô hình vào bài báo**.

---

## 6. Hàm ý cho hệ thống trong repo này

Dựa trên thể lệ trên, hệ thống cần lưu ý:

- **Bài nộp phải chứa pattern `Điều X` trong trường `answer`** — scorer truy hồi dựa vào
  đó. Prompt/format câu trả lời phải luôn trích dẫn `Điều X` rõ ràng.
- **F2 ưu tiên Recall** → nên truy hồi đủ rộng (`TOP_K` cao) để không bỏ sót căn cứ,
  nhưng vẫn cân bằng Precision.
- **Grounding & chống bịa (hallucination)** là tiêu chí chấm điểm trực tiếp → bắt buộc
  trích dẫn nguồn, không bịa điều luật.
- Tập trung phạm vi **Luật Doanh nghiệp & văn bản liên quan đến SME** (thuế, lao động,
  hợp đồng...).
- **Format `relevant_articles`/`relevant_docs` phải đúng công thức tên văn bản** (xem 5.2)
  → đây là điều kiện để scorer khớp đúng. Nộp `results.json` zip phẳng (xem 5.2).
- **BTC cho phép finetune** (xem 5.3): RAG vẫn là trục chính (không có train set gán nhãn),
  nhưng có thể cân nhắc finetune embedding/reranker trên dữ liệu pháp luật tự thu thập để
  tăng recall — miễn ghi rõ cách làm vào bài báo.
