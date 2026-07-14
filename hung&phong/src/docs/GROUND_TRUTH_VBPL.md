# Danh sách văn bản chuẩn theo phạm trù (ground-truth bản 2026)

> Xây từ NGUỒN ĐỘC LẬP (web: thuvienphapluat, chinhphu, dangcongsan...) theo phạm trù
> `project.md` §2, rồi đối chiếu corpus `vbpl_v2` (= tmquan/vbpl-vn). Mục tiêu: verify
> corpus có ĐỦ văn bản trong phạm trù, **bản hiệu lực 2026**, trước khi tối ưu thuật toán.
>
> Phương pháp (theo chỉ thị chủ dự án): web list → đối chiếu corpus → kiểm "lấy đúng chưa"
> (parse) → kiểm "lọc nhầm không" (scope). Trạng thái: ✅ đủ&đúng · 🟡 parse thiếu · 🔴 thiếu/0 điều.

## Kết luận tổng thể
1. **Độ phủ VĂN BẢN: TỐT** — tmquan/vbpl-vn phủ đủ luật cốt lõi, kể cả bản mới 2024-2026.
   **KHÔNG cần đổi nguồn.** (th1nhng0 chỉ còn bám ở config/eval — nên gỡ, xem cuối file.)
2. **Lỗi chính = PARSE mất điều** ở văn bản nhiều điều (gate tăng dần vỡ khi gặp tham chiếu
   "Điều N" chen giữa). Đây là đòn data #1.
3. **Thiếu hẳn** vài văn bản (đa số là NĐ xử phạt / văn bản rất mới 2026).

## Bảng đối chiếu (bản hiệu lực 2026)

### Doanh nghiệp / SME / Đầu tư
| Số hiệu | Tên | corpus | điều | TT |
|---------|-----|:------:|-----:|:--:|
| 59/2020/QH14 | Luật Doanh nghiệp 2020 | có | 218 | ✅ |
| 76/2025/QH15 | Luật sửa đổi Luật DN (01/7/2025) | có | 3 | 🟡 verify (VB sửa đổi) |
| 04/2017/QH14 | Luật Hỗ trợ DNNVV | có | 35 | ✅ |
| 168/2025/NĐ-CP | Đăng ký DN (thay 01/2021) | có | 123 | ✅ |
| 47/2021/NĐ-CP | HD Luật DN | có | 35 | ✅ |
| 80/2021/NĐ-CP | HD Luật SME | có | 35 | ✅ |
| 39/2019/NĐ-CP | Quỹ phát triển DNNVV | có | — | ✅ |
| 210/2025/NĐ-CP | Sửa NĐ 38/2018 (SME khởi nghiệp) | có | 4 | 🟡 verify |
| 61/2020/QH14 | Luật Đầu tư 2020 | có | 77 | ✅ |
| 22/2023/QH15 | Luật Đấu thầu 2023 | có | 96 | ✅ |
| 51/2014/QH13 | Luật Phá sản | có | 133 | ✅ |
| 122/2020/NĐ-CP | Liên thông ĐKDN | — | 0 | 🔴 THIẾU |
| 198/2025/QH15 | NQ cơ chế kinh tế tư nhân | — | 0 | 🔴 THIẾU |

### Thuế (bản 2026)
| Số hiệu | Tên | corpus | điều | TT |
|---------|-----|:------:|-----:|:--:|
| 67/2025/QH15 | Luật Thuế TNDN 2025 (thay 2008, hl 01/10/2025) | có | 20 | 🟡 verify (~21 điều?) |
| 48/2024/QH15 | Luật Thuế GTGT 2024 (hl 01/7/2025) | có | 18 | ✅ (18 điều đúng) |
| 38/2019/QH14 | Luật Quản lý thuế | có | 152 | ✅ |
| 320/2025/NĐ-CP | HD Luật TNDN | có | 26 | ✅ |
| 70/2025/NĐ-CP | Hóa đơn 2025 | có | 3 | 🟡 verify |
| 123/2020/NĐ-CP | Hóa đơn (cũ) | có | 61 | ✅ |
| 125/2020/NĐ-CP | Xử phạt thuế/hóa đơn | có | 0 | 🔴 0 ĐIỀU |
| 20/2026/TT-BTC | HD Thuế TNDN (12/3/2026) | — | 0 | 🔴 THIẾU |

### Lao động / BHXH (bản 2026)
| Số hiệu | Tên | corpus | điều | TT |
|---------|-----|:------:|-----:|:--:|
| 45/2019/QH14 | Bộ luật Lao động 2019 | có | 220 | ✅ |
| 41/2024/QH15 | **Luật BHXH 2024 (141 điều, hl 01/7/2025)** | có | **37** | 🔴 **MẤT 104 ĐIỀU** |
| 374/2025/NĐ-CP | HD BH thất nghiệp (01/01/2026) | có | 46 | ✅ |
| 293/2025/NĐ-CP | Lương tối thiểu vùng | có | 5 | ✅ |
| 145/2020/NĐ-CP | HD Bộ luật Lao động | có | 115 | ✅ |
| 12/2022/NĐ-CP | Xử phạt lao động | có | 64 | ✅ |
| 84/2015/QH13 | Luật An toàn vệ sinh lao động | có | 93 | ✅ |
| 50/2024/QH15 | Luật Công đoàn 2024 | có | 37 | ✅ |

## Văn bản cần SỬA / BỔ SUNG (đòn data)
**A. Sửa parser "nuốt điều" + re-parse (cứu nhiều điều gold nhất):**
- 🔴 `41/2024/QH15` BHXH 2024 (37/141), `274/2025/NĐ-CP` (16/131), `122/2021/NĐ-CP` (10/82),
  `17/2022/NĐ-CP` (19/87), `67/2026/NĐ-CP`, `87-CP`... (xem `docs/DATA_AUDIT.md`).

**B. Bổ sung văn bản thiếu hẳn (crawl từ vbpl.vn):**
- 🔴 `125/2020/NĐ-CP`, `99/2013/NĐ-CP` (parse 0 điều — có thể re-parse cứu được, kiểm tra trước).
- 🔴 `122/2020/NĐ-CP`, `198/2025/QH15`, `20/2026/TT-BTC` (thiếu hẳn — crawl bổ sung).

**C. Dọn nguồn th1nhng0 (không dùng nữa):**
- `backend/config.py` (`hf_dataset`), `ingest/download.py`, `ingest/run.py`,
  `tests/build_eval_set.py`, `scripts/_inventory.py` → chuyển hẳn sang tmquan.
- Field `tinh_trang_hieu_luc` của th1nhng0 KHÔNG đáng tin (Luật Đầu tư 2020 bị gán "hết
  hiệu lực" sai) → không dùng để lọc.
