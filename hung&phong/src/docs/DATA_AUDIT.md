# Kiểm tra độ phủ & chất lượng DATA (2026-06-13)

> "Retrieve là quan trọng nhất; muốn retrieve tốt bước đầu phải có DATA." — chỉ thị chủ dự án.
> Audit này chạy trên `data/corpus_vbpl_v2/*.parquet` (không đụng Qdrant).

## Tổng quan
- **16,197 văn bản** / **136,527 điều** / 196,842 chunk (Qdrant).
- Đa số luật cốt lõi SME/DN **đầy đủ điều**: Luật DN 2020 (218), Lao động 2019 (220),
  Dân sự 2015 (689), Đầu tư 2020 (77), Quản lý thuế (152), Kế toán (74), Phá sản (133),
  Thương mại (324), SHTT 2005 (230), HTX 2023 (115), Đấu thầu 2023 (96), NĐ 80/2021 (35),
  NĐ 01/2021 (101)... → **nền tảng OK**.

## 🔴 Vấn đề 1 — Văn bản còn hiệu lực bị THIẾU (0 điều, không index được)
| Số hiệu | Tên | Ảnh hưởng |
|---------|-----|-----------|
| `125/2020/NĐ-CP` | Xử phạt VPHC về thuế, hóa đơn | Gold cho câu hỏi phạt thuế/hóa đơn SME |
| `99/2013/NĐ-CP` | Xử phạt VPHC về sở hữu công nghiệp | Gold cho câu hỏi vi phạm nhãn hiệu/sáng chế |
| `07/2022/QH15` | Luật sửa đổi, bổ sung Luật SHTT 2022 | Gold cho câu hỏi SHTT bản mới |
- Ngoài ra 56 Luật/NĐ/Pháp lệnh khác 0 điều nhưng **đa số CŨ/hết hiệu lực** (1998-2003) → ưu tiên thấp.
- Tổng 4,440 VB 0 điều, phần lớn là Thông tư/Quyết định cũ → không nghiêm trọng.

## 🔴 Vấn đề 2 — Parse lỗi "ĐIỀU CUỐI NUỐT HẾT" (nghiêm trọng nhất cho recall)
Một điều (thường điều cuối) gộp luôn nội dung các điều sau → các điều đó **biến mất khỏi
corpus**. Gold "Điều X" không tồn tại → retrieve ra sai số điều → **BTC chấm trượt cả
recall lẫn precision**.

**Ví dụ điển hình** `122/2021/NĐ-CP` (xử phạt KH&ĐT): Điều 1-9 bình thường, **Điều 10 =
160,385 ký tự** nuốt toàn bộ Điều 11→~89. Mất ~79 điều.

**~27 văn bản GỐC bị lỗi thật** (đã loại văn bản "sửa đổi"/danh mục có 1 điều dài tự nhiên).
Lỗi rõ (ít điều + 1 điều khổng lồ):
| Số hiệu | Điều parse được | Điều khổng lồ | Loại |
|---------|----------------:|--------------:|------|
| `122/2021/NĐ-CP` | 10 | 160k | XP kế hoạch & đầu tư |
| `67/2017/NĐ-CP` | 10 | 118k | XP VPHC |
| `60/2003/NĐ-CP` | 10 | 102k | HD Luật (cũ) |
| `17/2022/NĐ-CP` | 19 | 127k | XP chứng khoán |
| `70/2025/NĐ-CP` | 3 | 104k | Hóa đơn 2025 |
| `139/2017/NĐ-CP` | 53 | 191k | XP xây dựng |
- Tổng thể: 747 VB có ≥1 điều >20k ký tự, NHƯNG phần lớn là **văn bản sửa đổi/danh mục**
  (cấu trúc 1 điều dài tự nhiên — KHÔNG phải lỗi). Lỗi thật tập trung ở **~27 VB gốc** trên.
- **Pattern chung: các Nghị định XỬ PHẠT VPHC** hay bị (cấu trúc nhiều điều khung phạt liền
  nhau làm regex tách điều fail). Mà NĐ xử phạt = **gold rất phổ biến** trong câu hỏi SME.

## Ý nghĩa & ưu tiên
1. **Đòn DATA #1 (cao nhất): sửa parser "nuốt điều" + re-ingest ~27 NĐ gốc** (ưu tiên NĐ
   xử phạt còn hiệu lực: 122/2021, 67/2017, 17/2022, 70/2025...). Mỗi VB cứu được hàng chục điều.
2. **Đòn DATA #2: bổ sung 3 văn bản thiếu** còn hiệu lực (125/2020, 99/2013, 07/2022).
3. Sau khi sửa data → mới re-embed → đo lại recall. Đây là gốc, làm TRƯỚC khi tinh chỉnh ranking.

## ✅ ĐÃ SỬA (2026-06-13)
1. **Parser "nuốt điều"** (`ingest/parse_vbpl.py`): gate tăng-dần-cứng → **gap-tolerant**
   (nhận điều nếu `last < N ≤ last+8`, vượt điều bị miss, vẫn loại tham chiếu nhảy xa).
   Verified: BHXH 41/2024 **37→140 điều**, 41/2009 +31, 87-CP +11; không regress luật lớn.
   - Còn lại vài NĐ phức tạp (274/2025, 17/2022) gap >8 chưa cứu — chấp nhận (thiểu số,
     tăng GAP sẽ nuốt nhầm tham chiếu).
2. **Dedup VB** (`export_corpus_vbpl.py` + `run_vbpl.py`): tmquan có ~610 số hiệu trùng
   record (59/2020 ×2 = 436 điều) → giữ bản nhiều điều nhất. Verified: 59/2020 về 218.
3. **Scope lọc nhầm** (`ingest/scope.py`): giữ HẾT Luật/Bộ luật/Pháp lệnh (không cần từ
   khóa) + thêm từ khóa (sở hữu công nghiệp, kinh tế tư nhân, công nghệ...). Lấy lại
   99/2013 (35đ), 198/2025 (17đ) trước bị loại oan.
4. **Xóa th1nhng0**: config → tmquan; xóa 3.97 GB cache + `vbpl_scope.parquet`.

**Kết quả corpus mới:** 136,527 → **194,579 điều distinct**, phủ đủ phạm trù + bản 2026.

## ✅ Bổ sung 3 VB thủ công (.docx) — 2026-06-13
tmquan KHÔNG có toàn văn 3 VB này (125/2020 chỉ có công văn đính chính; 122/2020, 20/2026
không có record). WebFetch/OCR đều không lý tưởng → tải bản `.docx` chính thống
(thuvienphapluat) → parse cùng parser tmquan → nhập corpus.
- `ingest/manual_docs.py`: đọc docx (`data/manual_vbpl/*.docx`) → ParsedDoc.
- Tích hợp vào `run_vbpl.py` (embed) + `export_corpus_vbpl.py` (parquet).
- Kết quả: `125/2020` (46 điều), `122/2020` (10), `20/2026/TT-BTC` (10) — text gốc sạch.

## ✅ Dọn code th1nhng0
Xóa `ingest/download.py`, `ingest/run.py`, `scripts/_inventory.py`. Còn `tests/build_eval_set.py`
trỏ vbpl_scope (eval phụ) — sửa sau.

## ✅ Verify ĐỘ PHỦ bài bản (whitelist từ web) — 2026-06-13
Đối chiếu corpus với danh sách văn bản hướng dẫn 12 lĩnh vực trụ cột (crawl luatvietnam/
thuvienphapluat): DN, SME, Thuế, Lao động, BHXH, Đấu thầu, Kế toán, Hóa đơn, HTX, Đầu tư,
SHTT, Thương mại/Dân sự. **Kết quả: 95/95 VB whitelist đều có trong corpus.**
- Scope nâng cấp **3 tầng** (`data/scope_whitelist.json` + Luật + từ khóa) → không lọc nhầm
  (vd cứu 135/2020 "tuổi nghỉ hưu", 05/2019 kiểm toán, 21/2021 BLDS bảo đảm).
- **11 VB bổ sung từ .docx chính thống** (tmquan không có/sai): 125/2020, 122/2020, 20/2026,
  135/2020, 157/2025, 81/2018, 07/2022/NĐ-CP, 07/2022/QH15, 68/2026, 141/2026, 132/2026.
- Corpus cuối: **205,044 điều · 21,574 VB**.

## ⏳ Bước cuối (đang chạy)
- [~] **Re-embed Qdrant `vbpl_aiteam`** (`run_vbpl.py --recreate`) — đang chạy ~2-3h GPU.
- [ ] Sau đó: `scripts/build_bm25.py` + `scripts/build_hnsw.py` + `verify_index.py`.
- [ ] Nộp lại để đo recall thật → dò VB biên còn thiếu (nếu có) qua câu zero-recall.
