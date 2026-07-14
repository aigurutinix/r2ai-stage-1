# Research: Tinh chỉnh vòng 2 — lọc hiệu lực văn bản + cân precision

**Ngày:** 2026-06-10 11:55 · **Trạng thái baseline:** corpus 93K đẩy ARTICLES_F2 **0.3877 → 0.4616** (+19%), DOCS_F2 0.492. Recall vượt precision (A: R0.51/P0.41; D: R0.58/P0.38) → nút thắt mới = **PRECISION** (nhiễu corpus rộng).

## Phát hiện chính

### 1. Dataset KHÔNG có trường hiệu lực
Kiểm tra schema `tmquan/vbpl-vn` (30 cột qua datasets-server API): không có `status/hiệu lực/expiry`. → Không lọc hiệu lực "chính thống" được từ data; phải heuristic.

### 2. Nhiễu văn bản HẾT HIỆU LỰC là có thật và LỚN (đo trên retrieved time2)
- **1.543 candidate thuộc luật đã bị thay thế**, dính **735/2000 câu** (chỉ với map ~18 luật lớn).
- **1.110/2000 câu** có cặp cùng-tên-khác-số-hiệu trong top-12 (version cũ/mới cạnh nhau).
- Top sót: 68/2014 (LDN cũ, 409 lần), 10/2012 (BLLĐ cũ, 306), 78/2015/NĐ-CP (ĐKDN cũ, 303), 60/2005 (LDN 2005, 267), 58/2014 (BHXH cũ, 81).
- Gold gần như chắc trỏ bản hiện hành → giữ bản cũ = sai cả docs lẫn articles → precision drain trực tiếp.

### 3. Giải pháp đã implement (2 lớp, trong `backend/retrieval_cutoff.py`)
- **(a) `SUPERSEDED_DOCS`** — map curated 18 cặp luật bị thay thế → bản hiện hành (verified: BHXH 41/2024 thay 58/2014; TCTD 32/2024 thay 47/2010, hiệu lực 01/07/2024; Đất đai 31/2024 thay 45/2013; Nhà ở 27/2023; KD BĐS 29/2023; LDN 59/2020; BLLĐ 45/2019; Đầu tư 61/2020; CK 54/2019; MT 72/2020; BLDS 91/2015; NĐ 01/2021 thay 78/2015...). Drop thẳng.
- **(b) same-name-keep-newest** — trong top-12 của 1 câu, 2 văn bản trùng clean_name (chuẩn hóa) khác năm → giữ năm mới. An toàn theo ngữ cảnh (trùng tên trong cùng top-12 ≈ version của nhau). Never-empty fallback.
- Lọc TRƯỚC cutoff → slot trống đôn candidate dưới lên (recall không mất, có khi tăng).

### 4. Sweep vòng 2 đã sinh (từ time2, không cần GPU)
| zip | cutoff | filter | điều/câu | sót superseded |
|---|---|---|---|---|
| `f_t3m3` | 3/3.0 | ✓ | 2.18 | 0 |
| `f_t4m35` | 4/3.5 | ✓ | 2.72 | 0 |
| `f_t5m4` | 5/4.0 | ✓ | 3.24 | 0 |
| `f_t6m6` | 6/6.0 | ✓ | 4.39 | 0 |
| `raw_t4m35` | 4/3.5 | ✗ (đối chứng) | 2.88 | 440 |

A/B sạch: so `f_t4m35` vs `raw_t4m35` tách riêng tác dụng filter; 4 mức cutoff tìm đỉnh mới (vòng 1 đỉnh ở chặt).

## Hướng tiếp theo (sau khi có điểm vòng 2)
1. **Nếu filter thắng** (gần chắc): bake `drop_superseded` vào notebook Kaggle `get_ctx` + Phase B; cân nhắc lọc luôn ở corpus-level (build_corpus blacklist) để slot top-12 đỡ phí.
2. **DOCS_R 0.58 > ARTICLES_R 0.51** → đúng văn bản nhưng sai điều trong văn bản. Đòn bẩy: tăng số điều/văn-bản-đúng (vd cutoff per-doc: giữ tới 2-3 điều cùng doc) — thử vòng 3.
3. **HyDE** vẫn trên bàn cho ~49% câu top-1 ≤ 0 (low-confidence) — chỉ re-embed 2000 query, tái dùng corpus_emb 93K.
4. Mở rộng `SUPERSEDED_DOCS` dần (NĐ/TT thuế-lao động hay bị thay) khi soi log.

## Câu hỏi mở
1. **0.4616 là của config nào?** (user gửi 1 JSON, không rõ tag). Cần điểm TỪNG bản vòng 1 để map đường cong → xin user.
2. BHXH 41/2024 hiệu lực 01/07/2025 — nếu BTC chốt gold trước mốc này, vài câu BHXH có thể vẫn theo 58/2014 (rủi ro nhỏ, chấp nhận).
3. Câu hỏi grader: leaderboard "Điểm" = ARTICLES_F2 hay avg(A,D)? (0.4616 khớp ARTICLES_F2MACRO → có vẻ Điểm = ARTICLES_F2).

## Nguồn
- [Schema tmquan/vbpl-vn (datasets-server)](https://datasets-server.huggingface.co/first-rows?dataset=tmquan%2Fvbpl-vn&config=documents&split=train)
- [Luật BHXH 41/2024/QH15](https://thuvienphapluat.vn/van-ban/Bao-hiem/Luat-Bao-hiem-xa-hoi-2024-557190.aspx) · [Luật TCTD 32/2024/QH15 hiệu lực 01/07/2024](https://tulieuvankien.dangcongsan.vn/he-thong-van-ban/van-ban-quy-pham-phap-luat/luat-cac-to-chuc-tin-dung-so-322024qh15-hieu-luc-thi-hanh-tu-ngay-0172024-10722) · [41/2024 toàn văn](https://xaydungchinhsach.chinhphu.vn/toan-van-luat-so-41-2024-qh15-bao-hiem-xa-hoi-119240723163650489.htm)
