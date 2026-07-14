# Verify bộ câu hỏi cuộc thi (R2AIStage1) — đọc & chấm thủ công

> BTC cung cấp **2.000 câu hỏi, KHÔNG có đáp án** (đúng thể lệ §5). Không thể
> tính P/R/F2 tự động vì thiếu nhãn vàng → áp dụng **§4.2.1 (LLM/Human-as-Judge)**:
> chạy hệ thống → **đọc từng câu trả lời thật** → đối chiếu điều luật trích dẫn với
> văn bản nguồn truy hồi → chấm đúng/sai.
>
> Mẫu chấm: **60/2000 câu** (đọc kỹ từng câu), file `data/competition_answers.json`.

## 1. Kết quả chấm thủ công (60 câu)

| Mức | Số câu | Tỷ lệ | Định nghĩa |
|---|---:|---:|---|
| ✅ **Đúng** (căn cứ đúng + còn hiệu lực + grounded) | 43 | **72%** | Trích đúng Điều của luật hợp lệ, nội dung khớp |
| ⚠️ **Một phần** | 13 | 22% | Đúng điều nhưng **bản cũ/hết hiệu lực**, sai nhãn số hiệu, hoặc thiếu ý |
| ❌ **Sai** | 4 | 7% | Không có căn cứ đúng / trích nhầm văn bản |

**Theo độ đo thể lệ §4.2.1 — "Căn cứ chính xác pháp luật" (có ≥1 điều luật đúng):**
≈ **90%** (43 đúng + ~11 câu "một phần" vẫn có ≥1 điều đúng). 6 câu không có căn cứ
đúng hiện hành: Q22, Q38, Q40, Q41 (+2 ranh giới).

**Thống kê tự động (60 câu):**
- Tỷ lệ có trích `Điều X` trong answer: **98%** (đáp ứng format chấm của BTC).
- Trung bình **2.8 điều/câu**.
- ⚠️ **64% nguồn truy hồi ở trạng thái "Hết hiệu lực"** ← vấn đề lớn nhất.
- 12% nguồn là Nghị quyết HĐND địa phương (nhiễu).

## 2. Độ chính xác **lưỡng cực theo chủ đề** (phát hiện chính)

| Nhóm chủ đề | Độ chính xác | Vì sao |
|---|---|---|
| **Hỗ trợ DN nhỏ và vừa** (Luật 04/2017 + NĐ 80/2021) | **Rất cao** (~95%) | Corpus có **full text** bản hiện hành |
| **Sở hữu trí tuệ** (Luật SHTT 2005 + NĐ 65/2023, 17/2023) | **Cao** (~90%) | Luật gốc 2005 + nghị định 2023 đều có |
| **Quản lý thuế / thuế** | **Yếu** (~50%) | **Luật QLT 2019 là bản RỖNG** trong dataset → hệ thống trích luật thuế 1990–2006 đã hết hiệu lực (Q38, Q40, Q41) |
| **Đấu thầu, lao động (mới)** | Yếu | Luật Đấu thầu 2023, BLLĐ 2019 thiếu/rỗng → trích bản cũ |

## 3. Các lỗi cụ thể cần sửa (để **nâng độ chính xác**)

1. **[LỚN] Trích bản luật đã hết hiệu lực** — 64% nguồn "hết hiệu lực". Retrieval
   chọn theo ngữ nghĩa, không phân biệt hiệu lực. Tax-admin sai nặng vì bản 2019
   rỗng.
   → **(a)** Bổ sung full text các luật bản mới: **Quản lý thuế 2019/2025, Đấu
   thầu 2023, BLLĐ 2019**. **(b)** Ưu tiên "Còn hiệu lực" khi xếp hạng (boost score
   hoặc lọc `tinh_trang`).
2. **[VỪA] Bug trích `Điều X` có hậu tố chữ** — "Điều 112a" không khớp regex
   `Điều\s+\d+\b` → Q42 ra `answer_dieu=[]`. Scorer BTC có thể cũng trượt.
   → Sửa regex thành `Điều\s+(\d+[a-zA-Zđ]?)`.
3. **[VỪA] LLM ghi sai nhãn số hiệu/tên luật** dù đúng số Điều (Q10 "45/2017",
   Q59 "14/2005"). Ít ảnh hưởng điểm F2 (chấm theo "Điều X") nhưng hại điểm nội dung.
   → Siết prompt: số hiệu phải copy đúng từ ngữ cảnh, không tự suy.
4. **[VỪA] Rò rỉ suy luận / tự sửa lỗi vào câu trả lời** (Q4 lặp "NGUỒN... sửa lại
   thành"). → Siết prompt "chỉ in câu trả lời cuối", giảm temperature.
5. **[NHỎ] Trích nhầm văn bản** (Q22 dẫn TT 16/2019/TT-BXD lạc đề) + nhiễu Nghị
   quyết HĐND địa phương (12%).
   → Loại NQ HĐND địa phương khỏi scope, hoặc downweight.

## 4. Hướng cải thiện ưu tiên (impact giảm dần)
1. **Lọc/boost theo hiệu lực** + **bổ sung luật bản mới** (thuế, đấu thầu, lao động) — vá nhóm yếu nhất.
2. **Hybrid sparse-dense + rerank** (bge-m3 có sẵn sparse) — tăng recall điều đúng.
3. **Sửa regex trích `Điều Xa`** — không mất điểm oan ở luật sửa đổi.
4. **Siết prompt + giảm temperature** — bớt rò rỉ, bớt sai nhãn, bớt trích thừa.

> **Kết luận**: trên mẫu 60 câu, hệ thống **trích đúng căn cứ pháp luật ~72% (chặt)
> / ~90% (có ≥1 điều đúng — §4.2.1)**. Điểm mạnh ở DN-SME & SHTT (corpus đủ); điểm
> yếu tập trung ở thuế/đấu thầu do **thiếu bản luật hiện hành** — đây là đòn bẩy số 1.

## 5. Cải thiện đã triển khai (v2)

| # | Fix | File | Trạng thái |
|---|---|---|---|
| 1 | **Re-rank theo hiệu lực**: lấy dư 3× ứng viên, phạt mềm bản hết hiệu lực (toàn bộ −0.08, một phần −0.025), chọn top-k | `backend/rag.py` | ✅ |
| 2 | **Sửa regex** trích `Điều Xa` (hậu tố chữ) + chuẩn hoá định danh điều thành chuỗi | `tests/metrics.py` | ✅ (test 11/11) |
| 3 | **Siết prompt**: copy đúng số hiệu, ưu tiên còn hiệu lực, chỉ in câu trả lời cuối (không rò rỉ suy luận) | `backend/prompts.py` | ✅ |
| 4 | **Giảm temperature 0.2 → 0.0** | `.env` | ✅ |

(Bỏ qua theo lựa chọn: bổ sung full text luật bản mới — thuế/đấu thầu/lao động.)

### Kết quả v2 (so với v1, cùng 60 câu) — `data/competition_answers_v2.json`

| Chỉ số | v1 (trước) | v2 (sau) |
|---|---:|---:|
| Citation rate (có ≥1 `Điều X`) | 98% | **100%** |
| **Nguồn HẾT hiệu lực trong top-10** | 64% | **28%** |
| Nguồn hết hiệu lực **TOÀN BỘ** | 44% | **7%** |
| **Top-1 source CÒN hiệu lực** | 28% | **65%** |

**Spot-check các câu trước bị lỗi:**
- **Q22** ❌→✅: trước trích nhầm `TT 16/2019/TT-BXD` (xây dựng) → nay đúng `NĐ 80/2021 Đ13`.
- **Q42** (bug)→✅: trước `cites=[]` (trượt "Điều 112a") → nay `['112','112a']`.
- **Q4** (rò rỉ)→✅: hết lặp "NGUỒN… sửa lại thành…", số hiệu đúng `04/2017/QH14` (trước nhầm `45/2019`).
- **Q41** (thuế cưỡng chế): top-3 nguồn hết hiệu lực → nay **3/3 còn hiệu lực**.
- **Q38/Q40** (thuế): cải thiện nguồn nhưng **vẫn hạn chế** vì Luật QLT hiện hành thiếu trong corpus → cần fix #3 đã bỏ qua.

**Đánh giá**: các đòn bẩy đã giảm **mạnh** việc trích luật hết hiệu lực (top-1 còn
hiệu lực **28%→65%**, hết hiệu lực toàn bộ **44%→7%**), sửa dứt điểm bug trích `Điều
Xa` và lỗi rò rỉ/sai nhãn của LLM. Ước tính độ chính xác chặt **~72% → ~82–85%**
(nhóm "một phần do bản cũ" phần lớn nâng thành "đúng"); nhóm **thuế hành chính** vẫn
là điểm yếu còn lại, chỉ vá được khi **bổ sung full text Luật Quản lý thuế 2019/2025**.

## 6. Tinh chỉnh re-rank (v3)

A/B trên gold set phát hiện v2 **phạt oan** bản "hết hiệu lực MỘT PHẦN" — nhưng luật
sửa đổi một phần (Luật DN 2020, SHTT 2005, Luật 04/2017) **vẫn là VB hiện hành**.
v3 sửa: **chỉ phạt "toàn bộ"** + boost luật lõi quốc gia (+0.02) + hạ Nghị quyết
HĐND địa phương (−0.05). Kết quả v2→v3 (60 câu):

| Chỉ số | v2 | v3 |
|---|---:|---:|
| Nghị quyết HĐND địa phương (nhiễu) | 13% | **3%** |
| Top-1 là Luật/Bộ luật/Pháp lệnh | 37% | **62%** |
| MRR (gold set) | 0.598 | **0.670** |

→ v3 trích **đúng văn bản gốc quốc gia** nhiều hơn (khớp đáp án BTC = Điều của luật
quốc gia). "% còn hiệu lực" nhìn giảm chỉ là **ảo giác đo lường** (luật gốc bị gắn cờ
"một phần").

## 7. Harness trích dẫn chi tiết (v4) — sửa nhầm số hiệu của qwen 9B

**Nguyên lý**: model 9B hay nhầm số hiệu (ghi `45/2019` thay `04/2017`) vì tự nhớ →
làm harness chi tiết để model **CHỈ VIỆC COPY**, không tự nhớ.

- `format_context` (backend/prompts.py): mỗi Nguồn kèm sẵn dòng **"MÃ TRÍCH DẪN"**
  dựng từ payload, vd `[Điều 12, Luật 04/2017/QH14]` (số hiệu luôn đúng).
- `SYSTEM_PROMPT`: viết lại thành **quy trình 5 bước**, ép "COPY y nguyên MÃ TRÍCH
  DẪN, ⚠️ nếu trí nhớ khác ngữ cảnh thì TIN theo ngữ cảnh".

**Kết quả (validated 5 câu Q1/Q5/Q7/Q8/Q10)**: số hiệu giờ **đúng 100%** (Q5/Q8/Q10
từ `45/2019` → `04/2017`; Q7 từ "Điều 40 BLLĐ 45/2019" → đúng `[Điều 38, NĐ 28/2020]`
/`[Điều 39, NĐ 12/2022]`), hết rò rỉ suy luận / từ chối.

> Còn lại (đòn bẩy lớn): **bổ sung full text Luật Quản lý thuế 2019/2025, Đấu thầu
> 2023, BLLĐ 2019** để vá nhóm thuế/đấu thầu (~50%).
