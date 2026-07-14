# Nhật ký rà soát sau mỗi lần nộp (Post-Submission Review)

> Quy trình bắt buộc (xem `CLAUDE.md`): mỗi lần có điểm BTC → mổ xẻ ART + DOCS, đọc đáp án
> cụ thể, truy đến cùng nguyên nhân, note lại đây. **Retrieve là quan trọng nhất; muốn
> retrieve tốt bước đầu phải có DATA.**

---

## Review #1 — Tổng rà soát v5→v11 (2026-06-13)

### Điểm thật (BTC)
| Version | ART_F2 | ART_R | DOCS_F2 | DOCS_R |
|---------|-------:|------:|--------:|-------:|
| v8  | 0.4475 | 0.554 | **0.510** ⭐ | **0.623** ⭐ |
| v9  | 0.4511 | **0.5737** ⭐ | 0.461 ⬇ | 0.573 |
| v10 | **0.4609** ⭐ | 0.525 | 0.455 ⬇ | 0.517 |
| v11 | 0.301 💀 | 0.369 | 0.430 | 0.483 |

### Phát hiện
1. **DOCS đỉnh ở v8, v9/v10 làm tụt** → multi-query + judge vứt cả văn bản đúng. Trước
   đây bỏ sót vì chỉ nhìn ART_F2.
2. **Recall kẹt trần ~0.52–0.57 suốt 6 version.** Leader ~0.725.
3. **v11 bỏ reranker = thảm họa** (0.30). Kết luận cứng: KHÔNG bao giờ bỏ reranker.

### Nguyên nhân gốc (đã truy đến cùng)
- v3 nhồi 10 điều/câu → recall thật chỉ 0.667 ⇒ **1/3 gold nằm ngoài top-10 retrieval**.
  Nút thắt là **RECALL retrieval**, không phải khâu lọc (v10 đã đẩy P lên 0.388).
- Corpus KHÔNG thiếu luật cốt lõi (16,197 VB; DN/SME/Lao động/Dân sự/SHTT/Thương mại đều có).
- Chỉ 2/2000 câu trích số hiệu → câu hỏi thuần **mô tả tình huống** → retrieval semantic khó.
- ⇒ Recall trần do **RANKING** (gap ngôn ngữ đời thường ↔ pháp lý), không phải thiếu data.
- Cảnh báo nhịp sai đã sửa: tưởng `91/2015/QH13` là rác → đào ra là Bộ luật Dân sự (689 điều),
  chỉ lỗi title ở `documents.parquet` (79 mã trùng), **submission ghi tên đúng**.

### Việc tôi đã làm SAI
- Dồn v9→v11 vào judge/lọc (precision) trong khi nút thắt là recall ⇒ tối ưu nhầm chỗ.
- Chỉ nhìn ART_F2 tổng hợp, không mổ DOCS, không đọc đáp án cụ thể, không mở điểm thật.

### Chi tiết đầy đủ → `docs/DIAGNOSIS_RECALL.md`

---

## ✅ Audit DATA đã chạy (2026-06-13) → chi tiết `docs/DATA_AUDIT.md`
- [x] **Chất lượng parse điều** → 🔴 TÌM RA LỖI: ~27 NĐ gốc bị "điều cuối nuốt hết"
      (vd 122/2021 mất ~79 điều). Các NĐ XỬ PHẠT hay bị nhất — mà đó là gold phổ biến.
- [x] **Độ phủ NĐ/TT hướng dẫn** → đa số OK; thiếu hẳn 125/2020 (XP thuế/hóa đơn),
      99/2013 (XP SHCN), 07/2022 (sửa SHTT).
- [x] **Văn bản mới 2024-2025** → có ingest (41/2024 BHXH, 24/2024 đấu thầu, nhiều NĐ 2025);
      nhưng vài cái parse lỗi (70/2025 hóa đơn chỉ 3 điều).
- [x] **Luật cốt lõi** → đầy đủ điều (DN/Lao động/Dân sự/Đầu tư/Thuế/Kế toán/Phá sản...).

## 🎯 Đòn DATA ưu tiên (làm TRƯỚC tinh chỉnh ranking)
1. **Sửa parser "nuốt điều" + re-ingest ~27 NĐ gốc** (cứu hàng chục điều/VB) — cao nhất.
2. **Bổ sung 3 văn bản thiếu** còn hiệu lực (125/2020, 99/2013, 07/2022).
3. Re-embed phần sửa → đo lại recall.

## Hướng version tiếp theo
- **v13 = HyDE/query-expansion**: Qwen dịch câu tình huống → đoạn văn pháp lý giả định → embed
  → phá trần recall. (Đòn đánh đúng gốc thay vì mài judge.)
- **Fine-tune reranker** domain pháp luật (BTC cho phép finetune).

---

## Review #3 — v14 (OVERHAUL DATA) (2026-06-14)

### Điểm thật (BTC) — TẤT CẢ TĂNG, không đánh đổi
| Chỉ số | v13 | **v14** | Δ |
|---|---:|---:|---:|
| ARTICLES_F2 | 0.4657 | **0.5170** | **+0.051** |
| ARTICLES_RECALL | 0.6320 | **0.7153** | **+0.083** 🔥 |
| ARTICLES_PRECISION | 0.2683 | **0.2883** | +0.020 |
| DOCS_F2 | 0.4654 | 0.5203 | +0.055 |
| DOCS_RECALL | 0.6200 | 0.6867 | +0.067 |
| DOCS_PRECISION | 0.2700 | 0.3150 | +0.045 |

QA (CHINH_XAC/DAY_DU/THUC_TIEN/RO_RANG) = 0.0 — chưa chấm answer (đang dùng answer v8 cũ).
1419/2000 câu đổi kết quả so v13. **Best submission tới nay.**

### Tại sao tăng (đòn DATA chứng minh đúng triết lý "data là gốc")
- **Recall ↑0.083**: `normalize_vn` chuẩn hoá `khỏan→khoản` (48% corpus, áp ĐỐI XỨNG corpus+query)
  → BM25 hết trượt từ pháp lý phổ biến nhất; coverage đủ + chunk theo cấu trúc (data mới
  41/2024, 67/2025, 108/2025, 168/2025, 125/2020... đều trả đúng — verify bằng manual_eval).
- **Precision ↑0.020**: drop ký hiệu rác "Không số" (155 VB trộn nhầm — hết "Nghị quyết bầu bổ
  sung Phó Chủ nhiệm" lọt vào id=1203) + cắt chunk đúng ranh giới khoản giảm nhiễu.

### Mổ đáp án (đọc trực tiếp 15 câu) — NHIỄU còn lại = đòn precision tiếp theo
1. **Trùng phiên bản CŨ** (lớn nhất): id=30 trả 3 bản NĐ đăng ký DN (78/2015 + 01/2021 +
   168/2025); id=27 (28/2011 + 156/2013 QLT cũ); id=34 (164/2003 + 24/2007 TNDN cũ). → cần
   **dedup theo HIỆU LỰC, giữ bản mới nhất**. ⚠ JUDGE đơn thuần KHÔNG lọc được (bản cũ vẫn
   "trả lời được" câu hỏi về mặt ngữ nghĩa).
2. **Lạc lĩnh vực** (TT anh em): id=200 (177/2015 bảo hiểm tiền gửi + 200/2014 DN lớn trong câu
   DN siêu nhỏ); id=1203 (27/2013 + 42/2022 ĐIỆN LỰC trong câu thương mại); id=32 (TT hải quan
   trong câu thuế); id=1208 (174/2024 bảo hiểm, 84/2011 giá). → **JUDGE CoT lọc tốt loại này.**
3. **Sót luật gốc hiện hành** (recall còn hụt): id=32 thiếu 108/2025 QLT; id=34 thiếu 67/2025
   TNDN; id=200/id=17 thiếu 132/2018 (kế toán siêu nhỏ). → vài câu thuế/kế toán còn miss gold.

### Đòn tiếp theo — PRECISION là nút thắt (0.288)
- Toán: F2=0.6 cần **P≈0.43** ở R=0.715 (giữ recall). → 2 đòn BÙ NHAU:
  - **(a) Effect-status dedup**: trùng chủ đề → giữ bản mới nhất (xử pattern #1, judge không xử được).
  - **(b) JUDGE CoT**: lọc lạc lĩnh vực (xử pattern #2).
- **Reranker fine-tune**: cuối cùng.
- Gap data nhỏ còn lại (parser trượt mốc "Điều N." ở vài VB: 125/2020 Đ29 nuốt vào Đ28,
  41/2024 Đ38, 116/2020 thiếu 6, 119/2018 thiếu 8) — content phần lớn còn (mis-cite), ưu tiên thấp.

### ✅ v15 — Effect-status dedup (post-process trên v14, không re-run) — GIẢ THUYẾT XÁC NHẬN
`scripts/dedup_effect_status.py`: gom article theo (loại, tiêu-đề-chuẩn-hoá), trùng tiêu đề +
khác số ký hiệu + khác năm → giữ bản mới nhất. Bỏ 337 article (4.7%) + 312 doc (5.3%).

| | v14 | v15 | Δ |
|---|---:|---:|---:|
| ARTICLES_PRECISION | 0.2883 | **0.2983** | +0.010 |
| ARTICLES_RECALL | 0.7153 | **0.7153** | **0.000** (0 gold mất!) |
| ARTICLES_F2 | 0.5170 | **0.5255** | +0.0085 |
| DOCS_F2 | 0.5203 | 0.5306 | +0.010 |

**Recall giữ NGUYÊN tuyệt đối** → 337 bản cũ bỏ đi không cái nào là gold → **"bản cũ = nhiễu
thuần" đúng 100%**. Dedup conservative (chỉ gộp tiêu đề GIỐNG HỆT) → còn để sót (39/2018 vs
80/2021, 28/2011 vs 156/2013 — tiêu đề gần giống). **Nới mạnh hơn (fuzzy title) sẽ ăn thêm
precision mà vẫn 0 rủi ro recall.** Đòn lớn kế: JUDGE (lạc lĩnh vực).

---

## Review #4 — v16 LLM-judge (song song) (2026-06-15)

`scripts/llm_judge_parallel.py` (6 worker Ollama, 2.7h): Qwen CoT đọc từng điều → CÓ/KHÔNG.
Lọc v15 6850→3539 điều (**−48%, 1.77 điều/câu**).

| Chỉ số | v15 | v16 | Δ |
|---|---:|---:|---:|
| ARTICLES_PRECISION | 0.298 | **0.440** | **+0.142** |
| ARTICLES_RECALL | 0.7153 | 0.5937 | **−0.122** |
| ARTICLES_F2 | 0.5255 | 0.5285 | +0.003 (phẳng) |
| DOCS_F2 | 0.5306 | **0.5695** | +0.039 |

**Bài học: judge QUÁ STRICT cho F2.** Precision nhảy +0.14 nhưng recall tụt −0.12; vì F2 ưu
tiên recall (×4) nên ART_F2 đứng yên. Judge cắt cả NĐ/TT hướng dẫn (thường là gold) khi đã
có điều luật gốc (test thấy: id=11 bỏ 80/2021 Đ5, id=30 bỏ 86/2024). DOCS_F2 lại tăng vì mức
doc precision lợi hơn loss.

→ **v17 = judge MỀM**: *giữ điều nếu rank≤2 HOẶC judge CÓ* (post-process từ `judge_cache_v15.json`,
không chạy lại LLM). keep_top=2 → 2.62 điều/câu. Mục tiêu: cứu recall (top reranked thường là
gold) mà vẫn cắt đuôi nhiễu → ART_F2 vượt cả v15 (recall cao) lẫn v16 (precision cao).
**Nguyên tắc rút ra: với F2, KHÔNG hi sinh recall để lấy precision 1:1 — phải lọc CÓ CHỌN LỌC,
bảo vệ top-rank.** (v18 keep_top=3 = 0.5380, thua v17 → keep_top=2 tối ưu.)

Điểm: v15=0.5255 → v16(judge full)=0.5285 → **v17(judge keep-top-2)=0.5716** ⭐ → v18(keep-top-3)=0.5380.

---

## Review #5 — Audit TOÀN BỘ 2000 câu v17 (6 agent đọc song song) (2026-06-15)

Bung 6 sub-agent đọc trọn 2000 câu (mỗi agent ~333 câu, tự chấm bằng kiến thức luật). Nguyên
nhân gốc rễ NHẤT QUÁN cả 6 vùng, xếp theo tác động:

**① VĂN BẢN CŨ HẾT HIỆU LỰC — lớn nhất (~20% câu, 10.8% article).** Bản kế nhiệm KHÁC tiêu đề
(dedup v15 chỉ bắt trùng tiêu đề). Lao động: 95/2013, 44/2003, 05/2015, 47/2010, 113/2004 →
145/2020+12/2022. Thuế/hóa đơn: 156/2013, 28/2011, 80/2012, 119/2018, 51/2010 → 80/2021+123/2020+108/2025.
Đăng ký DN: 88/2006, 43/2010, 78/2015, 01/2021 → 168/2025. Trọng tài: 08/2003/PL → 54/2010.
**Gốc: corpus không có metadata hiệu lực** (`tinh_trang_hieu_luc` rỗng) → reranker không phân
biệt cũ/mới. → **v19 = post-process drop 65-VB blacklist (đang test).**

**② PHÂN RÃ CÂU TÌNH HUỐNG NÔNG — lỗi RECALL #1 (~15%, vùng 1000–2000).** Câu đa vế (3–4 mệnh
đề) chỉ truy 1–2 vế (id=1671 sót Luật TM Đ302-307; id=1669 sót BLDS Đ122). `subqueries.json`
phân rã tối đa 4 vế, sót mệnh đề → **đây là trần recall 0.715**. Đòn recall = phân rã sâu hơn.

**③ LẠC LĨNH VỰC (~7%):** TT-NHNN trong câu dân sự; TT-BQP/BCA trong câu lao động DN; TT-BCT
điện lực trong câu thương mại; kế toán BQP/bảo hiểm trong câu kế toán DN. Gốc: không lọc theo
cơ quan ban hành / đối tượng áp dụng.

**④ ENCODING:** `13/2015/TT-BTС` chữ С Cyrillic → không match chunk. Sửa data nhanh.

**⑤ Judge cắt còn 1 điều (~4%)** — keep-top-2 đã đỡ phần lớn.

### Lộ trình (không cần fine-tune):
1. **① drop bản cũ (v19)** — instant, đang test. Kỳ vọng +precision, recall giữ (như dedup).
2. **③ scope filter cơ quan ban hành** — instant post-process (drop TT-BQP/BCA/NHNN khi câu không nhắc).
3. **④ fix Cyrillic** — sửa data.
4. **② phân rã sâu hơn** — đòn recall lớn nhất, cần chạy lại retrieval (đắt nhất, để cuối/cùng fine-tune).

---

## Review #5 — MỔ XẺ v22 (phân rã SÂU) vs v20 (best): VÌ SAO ② THẤT BẠI (2026-06-15)

> v20 (best): ART_F2=**0.5985** P=0.4633 R=0.6903, 2.296 điều/câu.
> v22 (②): ART_F2=**0.5048** P=**0.3363** (−0.127) R=**0.6480** (−0.042), 2.729 điều/câu. **TỆ CẢ HAI ĐẦU.**
> 1504/2000 câu khác nhau; v22 THÊM 2110 điều, BỎ 1243 điều so v20.

> ⚠️ **ĐÍNH CHÍNH QUAN TRỌNG (xác minh từ log build):** v20 **ĐÃ dùng phân rã NÔNG**
> (`subqueries.json`, max 4 vế, **max_k=4**). v22 đổi MỘT LÚC 2 thứ: (1) nông→SÂU
> (`subqueries_deep.json`) → trôi dạt; (2) **max_k 4→6** → tự nó nhồi +0.43 điều/câu.
> ⇒ recall giảm do subquery sâu trôi dạt; precision sập do CẢ subquery nhiễu LẪN max_k=6.
> **Phân rã KHÔNG phải đòn sai — phân rã SÂU + max_k cao mới sai.** Phân rã nông của v20
> neo tốt hơn (id 1719: nông giữ "cưỡng chế ngừng hóa đơn"; sâu đổi sang "đình chỉ phát hành").
> Agent #4 chấm "68% trôi dạt" là chấm file SÂU, KHÔNG phải phân rã nông của v20.

### Phương pháp
4 sub-agent đọc song song 4 lát cắt diff (`data/_diff_v20_v22/`), tự chấm bằng kiến thức luật (không gold).

### Bằng chứng hội tụ (4 góc độc lập đều chỉ về phân rã)
| Lát cắt | Phát hiện |
|---|---|
| Điều v22 THÊM (mẫu 55/663) | **68% NHIỄU** → P sập |
| Điều v22 BỎ (toàn bộ 139) | **~68% GOLD rơi oan** → R giảm |
| Câu tráo bộ điều (mẫu 55/702) | **62% TỆ hơn** v20 |
| Chất lượng phân rã (mẫu 60/1115) | **68% subquery TRÔI DẠT** = gốc rễ |

### 6 cơ chế hỏng (xếp theo sức phá hoại)
1. **Trôi dạt lĩnh vực** (~35%): từ khóa trùng bề mặt, sai luồng. id 884 "hợp đồng vô hiệu" (lao động)→hợp đồng xây dựng; id 784 "quá cảnh hàng hóa"→nhập cảnh người.
2. **Sinh vế NGOÀI câu gốc** (~35%): id 1425 câu hỏi "nguyên tắc" → LLM thêm "điều kiện/thủ tục" → loãng trọng số gold còn 1/3.
3. **Judge giết điều LUẬT KHUNG** (~29% câu mất): subquery hỏi "mức phạt" → loại Điều 19 Luật CĐ / Đ12 Luật DN / Đ175 BLLĐ (điều quy định hành vi/quyền). Câu phức hợp cần CẢ điều-hành-vi LẪN điều-chế-tài.
4. **Tách mục liệt kê của 1 điều** (~28%): id 861/954 "gồm những gì" → 5 vế → không vế nào pull đủ điều gốc.
5. **Paraphrase thay thuật ngữ** (~22%): id 1719 "cưỡng chế nợ thuế"→"đình chỉ phát hành" mất neo; + subquery thành câu KHẲNG ĐỊNH (id 554,1600) → embedding lệch.
6. **Merge max_k đẩy gold ra** (~18%): N subquery×top-k → pool to → cắt max_k → gold hạng giữa rớt (v20 retrieve 1 lần nên gọn). + **bug ký tự Trung "创业吗？"** (id 1010).

### v22 LÀM ĐÚNG 2 thứ (giữ lại để áp cho v20)
- Bỏ điều cross-domain v20 lỡ giữ (id 693: v20 dính nhầm Luật Thi hành án **hình sự**).
- Nâng VB hết hiệu lực→bản mới (id 1020 NĐ cũ→168/2025). ⚠️ RỦI RO: đề có thể soạn trước khi NĐ mới ban hành → **giữ CẢ HAI**, đừng thay hẳn.

### KẾT LUẬN & lộ trình mới
**Phân rã NÔNG = GIỮ (v20 đã dùng, là best). Phân rã SÂU + max_k cao = LOẠI (v22 −0.094).**
Điểm ngọt: nông max 4 vế + max_k=4 (như v20). Đừng đi sâu hơn, đừng tăng max_k. Lộ trình:
- **[A — an toàn, làm ngay]** Sửa judge: CẤM loại điều luật-khung/quyền/định nghĩa (phân biệt 2 loại gold). Effect-status giữ CẢ bản cũ+mới. Phạt nhẹ điều phạm-vi-hẹp khi câu nói "công ty" chung.
- **[B — nếu vẫn tăng recall, làm ĐÚNG]** Phân rã chỉ BỔ SUNG (`[câu_gốc]+subqueries` union, giữ neo); chỉ câu đa-vấn-đề thật; n_subs≤2; subquery phải là câu hỏi, giữ thuật ngữ, không thêm vế. HOẶC an toàn hơn: **mở rộng top-k trên CHÍNH câu gốc** (top-30/50)+reranker thay vì phân rã.
- **[C — phá trần thật]** **Fine-tune reranker** trên đường single-query v20: đẩy gold rank 3-10 lên top, KHÔNG rủi ro trôi dạt. Khuếch đại cái v20 đang làm tốt.

---

## Review #6 — Campaign v24, Exp B (collapse phiên bản) — 2026-06-15

> Chủ dự án: **nộp tự do, KHÔNG fine-tune tới khi TB chỉ số >0.75** (nhiều đội lên 0.75 không fine-tune).
> Vòng lặp mới = nộp BTC → đọc điểm + đáp án → đòn tiếp. Chỉ lo articles+docs (QA=0.0, xác nhận lại).

**Điểm v24_collapse:** ART_F2 **0.5997** (P 0.4733, R **0.6903**) · DOCS_F2 0.6092. vs v20 (0.5985/0.4633/0.6903).
- Bỏ 78 điều bản cũ ở 68 câu (cùng họ, bản mới CÓ mặt → recall-safe). **Recall GIỮ CHÍNH XÁC 0.6903** ✓ (xác nhận collapse an toàn), **P +0.010**.
- **F2 chỉ +0.0012** vì F2 nặng recall ×4 → **precision-only gần như vô dụng để lên điểm**.

**KẾT LUẬN cốt lõi (định hướng cả campaign):** RECALL kẹt y nguyên 0.6903 = trần thật. Mọi đòn precision (collapse/judge/blacklist) chỉ nhích lẻ. **Phải ĐẨY RECALL** = thêm điều gold đang thiếu. Giữ collapse làm lớp nền sạch (free, không hại). Đòn recall đang thử: **Exp A (bơm vế chế tài ①)**; tiếp theo HyDE, doc2query, ensemble BGE-M3 (đội ALQAC/COLIEE dùng — đều non-finetune).
- Pattern ② (bản cũ) ĐÃ XÁC NHẬN có thật nhưng **biên nhỏ** (~3-4% câu có cặp bỏ được).

---

## Review #7 — Exp A penalty ① (v24_penalty) — 2026-06-16

**Điểm v24_penalty:** ART_F2 **0.603** (P 0.4733, R **0.6937**) · DOCS_F2 0.6154. (v24_collapse: 0.5997/0.4733/0.6903).

**2 phát hiện:**
1. **Penalty additions KHÔNG nhiễu:** thêm 391 điều/220 câu, **P GIỮ NGUYÊN 0.4733** (điều thêm ~47% gold = bằng nền), R +0.0034. Whitelist NĐ xử phạt theo lĩnh vực định tuyến chuẩn (12/2022 lao động 244, 125/2020 thuế 71, 98/2020 TM 15, 45/2022 MT 14, 41/2018 KT 14...).
2. **Đã NGANG/VƯỢT leader ART_F2** (0.603 vs ~0.592). **P ta (0.473) > leader (0.462).** Gap DUY NHẤT = **RECALL 0.694 vs 0.725 (−0.03)**.

**KẾT LUẬN chiến lược:**
- Đòn NHẮM-MỤC-TIÊU (penalty, 15% câu) chỉ nhích recall +0.003 vì macro pha loãng. → **cần đòn TỎA RỘNG mọi câu**.
- **Precision còn DƯ ĐỊA** (thêm điều mà P không tụt) → bơm recall mạnh tay được mà không sợ sập P.
- Hướng: v25_full (retrieve rộng + max_k 5 + judge keep-top-2, đang chạy) → rồi HyDE/doc2query/ensemble BGE-M3 (tỏa rộng, non-finetune).
- **Nút cổ chai tốc độ = judge CoT đơn luồng (~0.05 câu/s)**. Cần tách recall (retrieve, nhanh) khỏi precision (judge, chậm) để lặp nhanh.

---

## Review #8 — BẢN ĐỒ RECALL-GAP toàn 2000 câu (đọc trên v24_penalty) — 2026-06-16

8 agent đọc ~1750 câu + toàn văn điều v24 đã chọn, phân loại gold còn THIẾU. Phân bố:

| Loại gap | % | Bản chất |
|---|---|---|
| **DIEU_KHUNG** | 25% | có NĐ, thiếu LUẬT GỐC (BLLĐ Đ104/124-130, BLDS Đ16, Luật ATVSLĐ Đ6, Luật BVNTD Đ10-11, Luật SHTT Đ118) |
| **VE_THU_TUC** | 20% | có Luật, thiếu NĐ/TT hướng dẫn thủ tục/hồ sơ (TT 80/2021, NĐ 168/2025, TT 78/2021) |
| VAN_BAN_MOI | 18% | bản cũ lọt top, bản mới KHÔNG được chọn (NĐ 22/CP 1994, 109/2010, TT 68/2019...) |
| NHAM_THUAT_NGU | 14% | sai domain — NỔI BẬT: lao động DN tư → kéo điều CÔNG CHỨC/VIÊN CHỨC (Luật CB-CC 22/2008, Luật VC 58/2010); cross-subdomain SHTT |
| VE_CHE_TAI | 12% | còn sót NĐ xử phạt (penalty lever đã đỡ phần lớn) |
| DA_DOMAIN | 8% | câu đa lĩnh vực, thiếu 1 lĩnh vực |
| THIEU_DATA | 3% | gold không có trong corpus |

**KẾT LUẬN CHỐT:** THIEU_DATA chỉ 3% → **DATA ĐỦ**. Recall sót (31% theo BTC) là **SELECTION/RANKING**, không phải thiếu data. Cụ thể:
- **DIEU_KHUNG + VE_THU_TUC = 45%** = thiếu điều BỔ TRỢ (đáp án cần CẢ luật gốc LẪN nghị định hướng dẫn — co-citation). → **exp_cocite.py**: bơm loại còn thiếu.
- VAN_BAN_MOI 18% → version routing (thay cũ bằng mới, kể cả khi mới không co-retrieve).
- NHAM_THUAT_NGU 14% → HyDE / lọc domain công-chức cho câu lao động tư.

**Lưu ý phương pháp:** agent gap-rate (~25-35% câu khó) < BTC recall-gap (31% gold) — agent CHẤP NHẬN điều "trông hợp lý" của v24 là đúng, nên phần lớn 31% còn lại là **chọn điều plausible-nhưng-không-trùng-gold-BTC** (ranking) → khớp với kết luận "nút thắt là ranking/selection".

---

## Review #9 — v25_full (dựng lại từ đầu, retrieve rộng) THẤT BẠI — 2026-06-16

| Chỉ số | v24_penalty | v25_full | Δ |
|---|---:|---:|---:|
| ARTICLES_F2 | 0.603 | **0.521** | −0.082 |
| ARTICLES_PRECISION | 0.4733 | **0.3207** | −0.15 (SẬP) |
| ARTICLES_RECALL | 0.6937 | **0.6903** | ~0 |
| DOCS_F2 | 0.6154 | 0.5166 | −0.099 |

3.026 điều/câu (v24: 2.39). **PHÁT HIỆN QUYẾT ĐỊNH:**
- **Recall ĐỨNG YÊN 0.6903** dù retrieve rộng (orig_topk 14, sub_topk 10) + decomp + penalty + max_k 5.
  → Nới việc CHỌN/retrieve **KHÔNG lôi thêm gold**. Gold thiếu (31%) **KHÔNG nằm trong pool ứng viên** (top-40).
- → **Recall bị chặn cứng ~0.69 ở TẦNG RETRIEVER (embedding AITeamVN + BM25), KHÔNG phải tầng selection.**
- Precision SẬP 0.47→0.32 vì 3.03 điều/câu = nhồi nhiễu. **Precision CỰC nhạy số điều.**

**ĐỔI HƯỚNG:** mọi đòn "chọn nhiều hơn / retrieve rộng hơn" trên CÙNG retriever là ngõ cụt (v25 chứng minh).
Phải **ĐỔI POOL ỨNG VIÊN**: HyDE (query→pseudo-điều-luật→embed), ensemble BGE-M3 (embedding khác),
doc2query (corpus→câu hỏi cho BM25). Đây mới là cách làm gold XUẤT HIỆN trong candidates.
exp_cocite (bổ túc luật/nđ) dùng CÙNG retriever → rủi ro chỉ thêm nhiễu như v25; nhưng additions
có MỤC TIÊU (đúng điều bổ trợ) nên còn cơ hội — sẽ đo.

---

## Review #10 — Đường biên recall/precision của v24 ĐÃ KỊCH (2026-06-16)

3 thí nghiệm liên tiếp trên v24 (best 0.603):
| Bản | Đổi | R | P | F2 |
|---|---|---:|---:|---:|
| v24 | base | 0.6937 | 0.4733 | 0.603 |
| v25_full | dựng lại, retrieve rộng, 3.03 điều/câu | 0.6903 | 0.3207 | 0.521 |
| v28 | +465 luật gốc (bừa) | 0.7003 | 0.435 | 0.59 |
| v29 | +176 luật gốc (co-citation cha-con) | 0.7003 | 0.46 | 0.6009 |

**KẾT LUẬN:** v24 ở ĐỈNH đường biên recall/precision của retriever+judge hiện tại; đường biên PHẲNG quanh F2≈0.60.
- Thêm gold (v28/v29) → recall +0.0066 nhưng precision pha loãng bù lại → F2 đứng yên.
- v25 (retrieve rộng) → recall KHÔNG tăng (0.69), precision sập → F2 −0.082.
- Phía precision: chỉ còn ~93 bản-cũ + 25 luật công-chức (~2.5% điều) xóa được → gain tí xíu.
- **Retriever max recall = 0.7153 (v15), ở đó P=0.298.** Leader R 0.725 / P 0.462 → đường biên TỐT HƠN HẲN.

**HỆ QUẢ:** đã vắt kiệt data + hậu-xử-lý ở ~0.60. Khoảng cách tới leader = CHẤT LƯỢNG XẾP HẠNG
(gold lên top → recall+precision đều cao). Non-finetune còn: ensemble BGE-M3 / HyDE (đổi pool/ranking,
đắt + không chắc). Chắc chắn nhất: fine-tune embedding/reranker (đang gate ở 0.75).

---

## Review #11 — Đòn RANKING (non-finetune) sau khi frontier v24 kịch — 2026-06-16

Frontier data+post-process kịch ở 0.60 (Review #10). Khoảng cách tới leader = CHẤT LƯỢNG XẾP HẠNG.
3 agent search → 3 đòn non-finetune cải thiện ranking (KHÔNG fine-tune, KHÔNG đổi embedding vì
AITeamVN-v2 vẫn mạnh nhất VN-legal):

1. **Swap reranker → AITeamVN/Vietnamese_Reranker** (BGE-M3 base, fine-tune 1.1M triplet VN legal,
   Zalo acc@1 ~0.7944 vs bge-reranker-v2-m3 generic). Drop-in (đổi env RERANKER_MODEL).
   **Calibrate trên câu thật: xếp ĐÚNG gold lên top** — Q"hộ KD" → Luật DNNVV Đ16 = #1 (rr 0.999),
   chính luật gốc mà bge xếp thấp + judge cắt oan (DIEU_KHUNG). → đánh đúng gap #1 (24%).
   Backend đã làm `RERANKER_MODEL` env-configurable. v30 = re-run no-judge với reranker này.
2. **Qwen LISTWISE reranker** thay judge CÓ/KHÔNG: NOWJ@COLIEE 2025 #1 dùng (DeepSeek/QwQ rank top-35),
   +14.8% F1 tuyệt đối. Judge ta chỉ LỌC, không XẾP HẠNG → đây là khác biệt cốt lõi với leader.
   + InsertRank (tiêm điểm BM25 vào prompt, +3-16%).
3. **BGE-M3 sparse + multi-vector (ColBERT)**: đã có model, chưa bật 2 mode kia. Hybrid +15-25% recall.
   + Jina-ColBERT-v2 rerank, citation graph 1-hop (luật↔NĐ).

Lộ trình: #1 (đang chạy) → đo → #2 (listwise) → #3. Tất cả non-finetune.

---

## Review #12 — v30 (reranker AITeamVN, no-judge, rebuild) THẤT BẠI nhưng NHIỄU — 2026-06-16

v30: ART_F2 0.5387 (R 0.6463, P 0.4017) vs v24 0.603. DOCS_R 0.70 (cao nhất).
**CONFOUND — không kết tội reranker:** v30 đổi cùng lúc (a) reranker AITeamVN, (b) MẤT tuning v20
(không blacklist/scope/judge-keep-top-2 — là rebuild full_v25 no-judge). Recall tụt chủ yếu do (b).
+ adaptive ratio=0.80 lệch calib cho reranker mới (điểm SẮC → cắt quá tay, id4 còn 1 điều).

**PATTERN XÁC NHẬN:** mọi REBUILD từ đầu (v25=0.521, v30=0.539) đều THUA v24 vì mất tuning chuỗi
v15→v20 (blacklist + scope + judge keep-top-2 + penalty). v24 = tối ưu cục bộ.
Test reranker SẠCH cần chạy ĐÚNG pipeline v20 + reranker mới + RE-JUDGE (cache vô hiệu → ~10h).

**Tình hình tổng:** đã thử non-finetune: decomp(fail), penalty(+0.003), cocite(neutral),
recover(neutral), rebuild rộng(−), reranker-swap(confound −). Tất cả ≤ v24 (0.603).
Frontier data+post-process KỊCH. Đòn còn lại: (1) Qwen LISTWISE rerank thay judge (research #1,
+14.8% F1 — chưa thử, chậm); (2) reranker test sạch trong pipeline v20 (~10h); (3) fine-tune (gate 0.75).

---

## Review #13 — v31 (Qwen LISTWISE rerank thay judge) — 2026-06-17

**Điểm BTC:** ART_F2 **0.5311** (P **0.570**, R **0.549**) · DOCS_F2 0.5899 (P 0.60, R 0.6033).
Build: dense+BM25 → AITeamVN reranker top-8 → collapse phiên bản (tầng ứng viên) → Qwen listwise
chọn subset (few-shot prompt, think=True, temp=0). TB **1.808 điều/câu** (v24: 2.39). 39632s/2000 câu.

**Đọc kết quả — listwise CHÍNH XÁC nhưng UNDER-RECALL (đúng dự đoán từ test 50 câu):**
- **Precision NHẢY 0.570** — cao HƠN đội đầu bảng (0.462)! → Qwen chọn điều rất đúng trọng tâm, sạch nhiễu.
  Version-leak (bản cũ) gần như hết nhờ collapse tầng ứng viên + few-shot prompt.
- **Recall SẬP 0.549** (v24 ~0.69) → F2 recall×4 phạt nặng → 0.5311 < v24 0.603.

**Mổ diff v31 vs v24 (cắt mất GÌ):**
- v31 bỏ **2957 điều** vs v24: **2037 NĐ/TT** + 848 luật → cắt cụt vế HƯỚNG DẪN/THỦ TỤC nhiều nhất.
- **1459/2000 câu (73%) THIẾU một vế cặp luật↔NĐ:** 719 chỉ-luật (thiếu NĐ), 740 chỉ-NĐ (thiếu luật).
- 785 câu (39%) chỉ 1 điều. → ĐÚNG pattern DIEU_KHUNG + VE_THU_TUC của gap map (46%).

**KẾT LUẬN — KHÔNG phải thất bại, là HƯỚNG ĐI ĐÚNG:** listwise chứng minh selection chất lượng cao
(P 0.57, thừa headroom). Lỗ duy nhất là recall do cắt cặp luật↔NĐ. → FIX = **floor recall**: bù vế
bổ trợ (NĐ nếu thiếu / luật nếu thiếu) điểm rr cao, dựng OFFLINE từ sidecar (không chạy lại Qwen).
Variants: floor@0.5=2.118 đ/câu (bù 620), floor@0.3=2.162 (bù 708), floor@0.15=2.207 (bù 797).

**ĐO THẬT B0.3 (floor@0.3):** ART_F2 **0.5387** (P 0.457, R 0.604) · DOCS_F2 0.5977 (P 0.47, R 0.673).
Sàn hồi recall +0.055 (0.549→0.604) NHƯNG P tụt −0.113 (0.57→0.457) → điều bù phần lớn NHIỄU.
F2 nhích +0.008, **VẪN dưới v24 0.603**. **B0.15 (ngưỡng 0.15): F2 0.538, P 0.443, R 0.604** —
recall Y HỆT B0.3 dù thêm điều → điều rr<0.3 toàn NHIỄU → **recall listwise BỊ CHẶN 0.604 do pool top-8/1-query HẸP**, không phải do ngưỡng.

**KẾT LUẬN TỚI HẠN — frontier P-R (F2=5PR/(4P+R)):**
| Điểm | P | R | F2 |
|------|---|---|----|
| v24 | 0.40 | **0.69** | **0.6026** |
| v31 thuần | **0.57** | 0.549 | 0.5531 |
| B0.3 | 0.457 | 0.604 | 0.5675 |
| đội đầu | 0.462 | **0.725** | 0.6509 |

- **Listwise CAP recall ~0.60** (Qwen TỰ giới hạn ~1.8 điều/câu bất kể max_k/floor) → không bao giờ chạm R=0.69 của v24. Listwise = đòn PRECISION, KHÔNG phải đòn recall.
- **Đội đầu R=0.725 > trần retriever của ta (0.7153, v15)** → họ thắng nhờ **RETRIEVER mạnh hơn**, không phải lọc giỏi hơn. Mọi post-process (judge/listwise/floor/penalty) chỉ ĐÁNH ĐỔI trong trần đó.

**HƯỚNG ĐI THẬT để chạm 0.75:** nâng **TRẦN RECALL của retriever** (>0.7153) — đúng triết lý "retrieve quan trọng nhất". Đòn: **RAG-fusion / multi-query** (research #139: nhiều reformulation + RRF → recall↑) để bơm gold vào pool, RỒI dùng **listwise (đã chứng minh P=0.57) lọc nhiễu**. Kết hợp = nhắm hồ sơ đội đầu (R cao + P cao). decomp v22 từng fail vì chỉ thêm nhiễu KHÔNG có tầng lọc mạnh — nay có listwise làm tầng lọc.

**Bài học:** listwise tách 2 năng lực — RANKING (đã rất tốt, P 0.57) vs COVERAGE (trần ~0.60, không cứu được bằng floor). Khi P cao bất thường mà F2 vẫn thua → nút thắt là RECALL ở tầng RETRIEVE, phải sửa ở GỐC chứ không post-process.

---

## Review #14 — v33 (v24 + BẢN ĐỒ HỌ PHIÊN BẢN) + chốt hạ filtering — 2026-06-17

**Điểm BTC v33:** ART_F2 **0.603** (P 0.473, R 0.694) · DOCS_F2 0.6154 — **TIES v24 (bản tốt nhất)**.

**Bản đồ họ phiên bản (workflow 20 agent, 134 họ/381 VB):** thay 6 rule tay của `family_tag` bằng
map toàn corpus. Gom phiên bản chính xác theo lĩnh vực (xphc thuế ≠ xphc lao động), kiểm phản biện.
- **LỖI over-collapse phát hiện & sửa:** họ LUẬT gom cả base + sửa-đổi (Luật SHTT 50/2005 + 07/2022...)
  → coi sửa-đổi là "mới" rồi bỏ 50/2005 = SAI (luật sửa đổi KHÔNG thay luật gốc). → LOẠI họ luật khỏi
  map (chỉ giữ NĐ/TT/QĐ); rule "law:" cũ xử lý luật-thay-thế đúng rồi.
- **Gate an toàn:** áp map lên v24 → bỏ 66 NĐ/TT cũ (103/2006, 155/2016 môi trường, 16/2016 SHTT...),
  KHÔNG đụng luật gốc, **F2 KHÔNG đổi (0.603)** → map AN TOÀN, là phần cố định pipeline. Nhưng v24 quá
  thưa (chỉ 58/2000 câu có 2 bản đồng hiện) nên map NEUTRAL trên v24.

**CHỐT HẠ chiến lược (từ v31→B0.3→B0.15→v32→v33):**
- Mọi tầng FILTERING (judge/listwise/floor/map) đều **≤ 0.603**. Listwise cho P cao (0.57) nhưng
  cap recall ~0.55-0.60 (Qwen chọn ~1.8-2.0 điều/câu) → không vượt được v24.
- **v33 R=0.694 đã GẦN KỊCH trần retriever 0.7153** → filtering chỉ xoay trong trần. Hết dư địa.
- Đòn DUY NHẤT còn lại để phá trần recall (non-finetune): **RAG-fusion rewrite** — sinh query diễn
  đạt lại để bắt gold mà query gốc bỏ sót. ĐANG TEST 30 câu đo độ phủ fusion vs 1-query.

**Bài học:** khi recall đã sát trần retriever, MỌI hậu xử lý đều bão hoà. Phải sửa ở tầng RETRIEVE
(query/embedding), không phải tầng chọn-lọc. Map phiên bản là cải tiến đúng & an toàn nhưng chỉ phát
huy khi pool dày (listwise) — mà listwise lại cap recall → map chưa đổi được điểm. Giữ map cho mọi bản sau.
