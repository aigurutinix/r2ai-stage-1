# NextGen — Nội dung Slide thuyết trình

> Copy từng slide vào PowerPoint/Google Slides. Chỗ `[...]` cần bạn điền.
> Số benchmark: thay bằng con số CHÍNH THỨC của đội khi có; số nội bộ đã ghi rõ "(nội bộ, 50 câu GT)".

---

## Slide 1 — Bìa / Giới thiệu nhóm
**NextGen — Trợ lý Pháp luật AI cho Doanh nghiệp nhỏ và vừa**
*Hỏi – Đáp pháp luật tiếng Việt, trả lời kèm trích dẫn văn bản kiểm chứng được.*

- **Đội:** `[Tên đội]`
- **Thành viên:** `[Họ tên 1 — vai trò]` · `[Họ tên 2 — vai trò]` · `[Họ tên 3 — vai trò]` …
- **Sản phẩm demo (online):** `https://maitrung21-nextgen-legal-rag.hf.space` *(kèm QR)*

---

## Slide 2 — Giải pháp · Điểm mạnh · Kết quả · Khác biệt  *(slide "chốt hạ" tổng quan)*

**Giải pháp:** Trợ lý AI trả lời câu hỏi pháp luật cho SME bằng **RAG** — truy hồi văn bản pháp luật chính thức rồi để LLM soạn câu trả lời **luôn kèm căn cứ điều luật**.

**Điểm mạnh — 4 trụ cột:**
- 🏗️ **Corpus tự xây, chất lượng cao:** tự crawl ~**612 văn bản** → bóc tách theo **cấu trúc pháp lý 4 cấp** (Chương→Điều→Khoản→Điểm), **kiểm định QA tự động** (phát hiện thiếu điều/khoản, lỗi cấu trúc, text rác) → dữ liệu sạch, không phải dữ liệu thô.
- ✂️ **Chunking thông minh cho RAG:** chunk theo **đơn vị Điều**, điều dài tách theo Khoản để **không cắt cụt khi embedding** nhưng vẫn giữ đúng `Điều N` → trích dẫn chuẩn cấp Điều. ~**30.000 chunk** giàu metadata.
- 🎯 **Truy hồi lai + Rerank tinh:** **Hybrid** (ngữ nghĩa + từ khóa) + **Multi-Query** + **RRF** + rerank Cross-Encoder → lọc từ ~30k điều xuống vài điều cốt lõi nhất.
- 🛡️ **Pipeline RAG chống ảo giác:** trích dẫn = **HỢP top-2 rerank ∪ điều LLM trích (đã lọc trong pool)** → tăng recall mà vẫn grounded; **self-verify 5 quy tắc**; mỗi câu trả lời có **link văn bản gốc**.

*(Bonus: sản phẩm thật, online 24/7 — không phải bản mẫu.)*

**Kết quả (F2-macro):** **F2 Điều 0.67 · F2 Văn bản 0.73** · Recall Điều 0.71 / Văn bản 0.79 *(chi tiết Slide 8).*

**Điểm khác biệt:** Thư viện Pháp luật / LawNet chỉ **tra cứu** → NextGen **trả lời trực tiếp + trích dẫn kiểm chứng**; **corpus tự xây có QA** + **RAG chống ảo giác**; tuân thủ **LLM < 14B**.

---

## Slide 3 — Tổng quan hệ thống (Pipeline)

```
Câu hỏi ──(3 biến thể Multi-Query)
   │
   ├─(1) Hybrid Retrieval ── Dense (ngữ nghĩa) + Sparse (từ khóa) ──► RRF merge → TOP-30 thô
   │
   ├─(2) Rerank ─────────── Cross-Encoder → POOL TOP-8 (giữ 8 đoạn liên quan nhất)
   │
   ├─(3) LLM sinh answer ── LLM < 14B → câu trả lời tiếng Việt (ngữ cảnh ≤ 12.000 ký tự)
   │
   ├─(4) Chọn trích dẫn ──── HỢP (∪): top-2 rerank  ∪  {Điều LLM trích ∩ pool}
   │                          → tăng recall, vẫn grounded (LLM chỉ chọn trong pool) · derive văn bản
   │
   └─(5) Đóng gói ────────── relevant_docs / relevant_articles + answer
```
**Ý chính:** *Retrieval khỏe → Rerank lọc tinh → LLM chỉ nói dựa trên bằng chứng → Giao citation để không bịa.*

---

## Slide 4 — Vì sao đạt kết quả (1/4): DỮ LIỆU

**Nguồn ở đâu:**
- Văn bản pháp luật Việt Nam trọng tâm **SME**: Luật, Bộ luật, Nghị định, Thông tư, Quyết định, Nghị quyết…
- Tự **crawl ~612 văn bản** từ **Thư viện Pháp luật** (nguồn uy tín), quản lý URL tập trung, **chạy theo từng đợt** (giãn cách 4–7s, User-Agent thật → tôn trọng máy chủ nguồn).

**Làm sao đảm bảo luật CÒN HIỆU LỰC:**
- Chọn **phiên bản đang áp dụng** (ưu tiên bản mới / hợp nhất); tên văn bản gắn **năm áp dụng** (vd *"…áp dụng 2025"*).
- **Manifest** chuẩn hóa số hiệu ↔ tên chính thức + **cờ tin cậy (confidence)** khi suy số hiệu → dễ rà soát, loại văn bản sai.
- **Cập nhật định kỳ** khi luật thay đổi → chỉ cần chạy lại crawl + nạp nguồn.

**Điểm cộng:** mỗi đợt crawl có **báo cáo QA** (số chương/điều/khoản/điểm, cảnh báo thiếu/nhảy số, text rác) → biết ngay văn bản nào cần kiểm tra thủ công.

---

## Slide 5 — Vì sao đạt kết quả (2/4): XỬ LÝ DỮ LIỆU

**Xử lý thành gì:** JSON phân tầng 4 cấp (Chương→Điều→Khoản→Điểm) → **~30.000 chunk**, mỗi chunk giàu metadata (số hiệu, Điều, Chương).

**Chiến lược chunking (điểm kỹ thuật):**
- Chunk theo **đơn vị Điều**; **điều dài tách theo Khoản** (dưới ngưỡng ~3.500 ký tự) để **tránh embedding bị cắt cụt** (giới hạn 2048 token) — nhưng vẫn giữ `article_id = "Điều N"` nên **chấm recall theo (văn bản, Điều) không đổi**.
- Suy **số hiệu** theo thứ tự ưu tiên (số hiệu → tên luật → tên file) kèm **cờ tin cậy**.
- Embedding **`Vietnamese_Embedding` (kiến trúc bge-m3, 1024 chiều, chuẩn hóa)**, lưu Qdrant (đo DOT), **upsert idempotent** (chạy lại không nhân đôi).

**Vì sao tăng chất lượng kết quả cuối:**
- Mỗi chunk là **một điều khoản tự đủ nghĩa** → embedding "sạch", truy hồi trúng đích, **trích dẫn đúng cấp "Điều X"**.
- Metadata số hiệu + Điều → map thẳng ra **văn bản gốc + link**; QA loại nhiễu ngay từ đầu vào.

---

## Slide 6 — Vì sao đạt kết quả (3/4): RETRIEVAL

**Truy hồi lai (Hybrid Retrieval)** — kết hợp 2 tín hiệu:
- **Ngữ nghĩa (Dense):** hiểu ý câu hỏi dù dùng từ khác luật.
- **Từ khóa (Sparse):** bắt chính xác số hiệu, thuật ngữ, con số.

**Kỹ thuật tăng chất lượng đầu ra:**
- **Multi-Query:** sinh nhiều biến thể câu hỏi → phủ rộng cách diễn đạt.
- **RRF (Reciprocal Rank Fusion):** hợp nhất 2 nguồn một cách cân bằng → top-30 ứng viên.
- **Truy vấn 2 giai đoạn (prefetch + rescore):** quét rộng để không sót, rồi chấm lại cho chính xác.
- **Rerank (Cross-Encoder):** chấm lại từng cặp (câu hỏi, đoạn) → giữ **8 đoạn liên quan nhất** làm căn cứ.

> *→ Đây là "trái tim" của độ chính xác: lọc từ 30.000 điều xuống đúng vài điều cốt lõi.*

---

## Slide 7 — Vì sao đạt kết quả (4/4): LLM + GUARD (chống ảo giác)

**LLM (< 14B tham số — tuân thủ quy định BTC)** soạn câu trả lời **chỉ dựa trên** các điều đã truy hồi.

**Guard kiểm định 2 đầu:**
- **Đầu vào (Input guard):** chuẩn hóa + mở rộng câu hỏi; giới hạn ngữ cảnh ≤ 12.000 ký tự (tránh nhiễu, cắt cụt).
- **Đầu ra (Output guard) — 5 quy tắc vàng:**
  1. Mọi **"Điều X"** trong câu trả lời **phải có trong** ngữ cảnh đã truy hồi.
  2. Mọi **số hiệu văn bản** phải tồn tại trong manifest.
  3. Bắt buộc có **≥ 1 căn cứ pháp lý** + cụm dẫn "Theo quy định tại Điều…".
  4. Cảnh báo số liệu (%) không có trong ngữ cảnh.
  5. **Sinh lại** ở nhiệt độ thấp nếu vi phạm.
- **Grounding trích dẫn (HỢP):** trích dẫn = **top-2 rerank ∪ điều LLM trích (đã lọc trong pool truy hồi)** → LLM **không thể bịa** điều ngoài dữ liệu, đồng thời **bổ sung recall** từ những điều đúng mà LLM chỉ ra ngoài top-2.

> *→ Kết quả: mỗi câu trả lời đều "có bằng chứng", có thể bấm xem văn bản gốc để kiểm chứng.*

---

## Slide 8 — Kết quả đạt được (Benchmark)

**Chỉ số đánh giá truy hồi** (ground-truth mỗi câu chỉ 1–3 điều → dùng **F2**, ưu tiên recall gấp 4):

| Chỉ số | **Điều (Articles)** | **Văn bản (Docs)** |
|---|---|---|
| **F2 (macro)** | **0.67** | **0.73** |
| **Recall** | 0.71 | 0.79 |
| **Precision** | 0.62 | 0.62 |

- **Chiến lược tối ưu F2:** trích dẫn = **HỢP top-2 rerank ∪ điều LLM trích (grounded trong pool)** → bổ sung recall từ điều đúng ngoài top-2, precision vẫn giữ ~0.62 → **F2 cao nhất** (Điều 0.67, Văn bản 0.73).
- So sánh baseline: `[điền điểm baseline / đội khác nếu muốn]`.

> ⚠️ Đây là số đo trên tập đánh giá của đội — cập nhật nếu có bản chấm chính thức mới hơn.

---

## Slide 9 — Tính ứng dụng thực tế

**Ai là người dùng khi ra thị trường:**
- Chủ **doanh nghiệp nhỏ / hộ kinh doanh** không đủ tiền thuê luật sư.
- **Kế toán, nhân sự, phòng pháp chế** của công ty vừa và nhỏ.
- **Văn phòng luật nhỏ** dùng như trợ lý tra cứu nhanh.

**Dùng được ngay không?** ✅ **Rồi** — sản phẩm đã **online 24/7**, mở link (hoặc quét QR) là hỏi được ngay, không cần cài đặt.

**Tiềm năng phát triển / mở rộng:**
- Thêm lĩnh vực (thuế, lao động, hợp đồng, bản án) = **chỉ cần nạp thêm văn bản**, không train lại.
- **Freemium (B2B):** miễn phí số câu/tháng, trả phí cho gói chuyên nghiệp; đã có **đăng nhập + lưu lịch sử**.
- Tích hợp **API** cho phần mềm kế toán / quản trị doanh nghiệp.

---

## Slide 10 — Kết / Kêu gọi
- **NextGen: trợ lý pháp luật AI — trả lời có căn cứ, đã chạy thật.**
- 👉 Trải nghiệm ngay: `https://maitrung21-nextgen-legal-rag.hf.space` *(QR lớn giữa slide)*
- Cảm ơn Ban giám khảo — sẵn sàng trả lời câu hỏi.

---

### Ghi chú trình bày (không lên slide)
- Chuẩn bị **video demo 60s** dự phòng; **đánh thức Space** trước giờ thi ~3 phút.
- Câu hỏi BGK hay hỏi → xem `docs/PITCH_5MIN.md` (bảng trả lời sẵn).
