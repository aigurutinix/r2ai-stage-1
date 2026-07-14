# Chẩn đoán truy hồi — vì sao F2 mới 0.41, sai do đâu

> Soi 2000 câu thi thật (top-15 ứng viên + điểm reranker từ `submission_v4_scored.json`)
> + corpus. Không có gold → dùng tín hiệu gián tiếp (độ tự tin reranker, đọc tay từng ca,
> bao phủ corpus). Ngày 2026-06-11.

## Tổng quan: lõi RAG TỐT, nhưng 3 lỗ rò kéo điểm

- **85.3%** câu có top-1 reranker ≥ 0.8 → truy hồi tìm đúng nội dung liên quan. Các ca
  trực tiếp (Q156, Q1494, Q1503, Q1532...) căn cứ **chuẩn**. Lõi hybrid+rerank ổn.
- Điểm kẹt **không** phải lõi, mà ở 3 chỗ rò bên dưới → precision 0.26, recall 0.52.

---

## VẤN ĐỀ 1 — Nhiễu BẢN LUẬT CŨ → giết PRECISION  ⭐ (rẻ nhất, làm ngay)

**Bằng chứng số:**
- Corpus có **9.716/17.087 docs trước 2010** (57%) — chứa nhiều thế hệ cũ của cùng một luật.
- **30.3%** câu (607/2000) có ≥2 bản cùng họ luật-chính trong top-5.
- **32.7%** văn bản trong top-3 submit là **trước 2015**.

**Ví dụ (đọc tay):**
- Q1494 (DNTN thuê giám đốc): top-4 = Luật DN **59/2020** Đ190 ✓ + **68/2014** + **60/2005**
  + **13/1999** — 3 bản CŨ đã hết hiệu lực, đều rr≈1.0. Gold chỉ 59/2020 → submit 4 thì
  **3 sai** → precision 1/4 = 0.25 ≈ đúng precision thực 0.26.
- Q1400, Q400 y hệt (kèm 68/2014, 78/2006...).

**Vì sao:** reranker không phân biệt được bản 2020 vs 2014 (văn bản gần như giống chữ) →
chấm đều ~1.0. Year-boost hiện tại quá yếu (max 0.05).

**Cách chữa (THUẬT TOÁN, instant, không cần chạy lại):** trong dedup, **gộp cùng họ luật
chính → chỉ giữ số hiệu MỚI NHẤT**, bỏ hẳn bản cũ. Hậu xử lý ngay trên scored.json → v5.
→ kỳ vọng precision 0.26 → ~0.35–0.42, **F2 → ~0.46–0.50**.

---

## VẤN ĐỀ 2 — Câu TÌNH HUỐNG DÀI nhiều vế → giết RECALL  ⭐⭐ (đòn lớn nhất)

**Bằng chứng số:**
- **49.6%** câu dài >160 ký tự; **40%** (801) nhiều vế (≥2 "thì/nhưng/và" + dài).
- TB top-1 rr: câu NGẮN(≤120) = **0.954** vs câu DÀI(>200) = **0.844**. Chênh rõ rệt.
- Toàn bộ đuôi thất bại (top-1<0.4) gần như đều là câu tình huống dài.

**Ví dụ (đọc tay):**
- Q1972 "thu phí tuyển dụng…xử lý ra sao": điều ĐÚNG (NĐ 12/2022 Đ8) **được truy về #1
  nhưng rr=0.03** — reranker chấm bậy vì câu là tình huống dài, điều luật là khoản phạt khô khan.
- Q1726, Q1725, Q596, Q1127: mỗi câu gộp 2–3 vấn đề pháp lý (dân sự + hành chính + lao động)
  → 1 điều chỉ khớp 1 vế → điểm thấp, sót các vế khác.

**Vì sao:** 1 câu = nhiều câu hỏi pháp lý con. Embedding/reranker 1-shot không khớp nổi
tất cả các vế cùng lúc → sót căn cứ cho các vế phụ.

**Cách chữa (KIẾN TRÚC):** **phân rã câu hỏi** — dùng Qwen tách câu dài thành 2–4 câu hỏi
con nguyên tử → truy hồi từng câu con → hợp nhất kết quả. Đây là pattern multi-query/CRAG.
→ kỳ vọng recall câu dài 0.84→~0.92; **recall tổng 0.52 → ~0.60–0.65**.

---

## VẤN ĐỀ 3 — THIẾU DATA domain mới (AI/Công nghệ số) → 0 điểm cụm câu  ⭐ (data)

**Bằng chứng số:**
- **68/2000 (3.4%)** câu chạm AI/công nghệ số; **13 câu** top-1 rr<0.4 (gần như KHÔNG có data).
- id ví dụ: 538, 541, 614, 620, 623, 626, 694, 712, 777, 787, 866, 935, 958.

**Ví dụ:** Q623 (hồ sơ hệ thống AI rủi ro cao), Q712 (gắn nhãn minh bạch AI), Q620 (Quỹ
Phát triển AI quốc gia) → answer "Tôi chưa có đủ thông tin", top rr ~0.01–0.07. Corpus
**không có** Luật Công nghiệp công nghệ số 2025 (chương AI) — bị scope-filter loại
(title không chứa từ khoá SME) hoặc dataset chưa có.

**Cách chữa (DATA):** crawl chính thống **Luật Công nghiệp công nghệ số 2025 + nghị định
hướng dẫn** → parse điều → embed → nạp thêm. Phục hồi ~13–30 câu đang ăn 0.

---

## Kết luận data vs thuật toán

| Vấn đề | Bản chất | Ảnh hưởng | Chi phí | Kỳ vọng |
|---|---|---|---:|---|
| 1. Bản luật cũ | **Thuật toán** (+data cũ) | precision, 30% câu | **Instant** | F2 → ~0.48 |
| 2. Câu tình huống | **Kiến trúc** | recall, ~40-50% câu | ~2-4h (LLM) | recall → ~0.62 |
| 3. Thiếu AI law | **Data** | ~3.4% câu (ăn 0) | crawl+ingest | +~13-30 câu |

→ **Không phải một nguyên nhân duy nhất.** Lõi tốt; cần vá 1 (precision, ngay) → 2
(recall, lớn nhất) → 3 (data gap). Làm tuần tự, đo bằng submission sau mỗi bước.
