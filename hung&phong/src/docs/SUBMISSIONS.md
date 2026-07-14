# Lịch sử bài nộp & điểm (Submission Changelog)

> Mỗi bản nộp + thay đổi + điểm BTC chấm. ARTICLES/DOCS_F2 chấm tự động; 4 tiêu chí QA
> (chính xác/đầy đủ/thực tiễn/rõ ràng) BTC chấm sau (hiện 0.0). Cập nhật 2026-06-14.

## Bảng điểm

| Bản | Thay đổi chính | ART_F2 | ART_P | ART_R | DOCS_F2 | Ghi chú |
|---|---|---:|---:|---:|---:|---|
| v2 | RAG cơ bản (dense + prompt harness) | 0.376 | 0.282 | 0.451 | — | mốc đầu |
| v3 | broaden top-10 điều | 0.329 | 0.127 | 0.668 | — | ❌ precision sập (nhồi điều) |
| v4 adaptive | hybrid+rerank + chọn thích ứng (avg 2.61) | **0.414** | 0.263 | 0.524 | 0.422 | ✅ tốt nhất ART |
| v4 top-3 | cố định 3 điều | 0.405 | 0.247 | 0.524 | 0.425 | adaptive > top-3 |
| v5 | + gộp bản luật cũ (IMPROVE#1) | 0.409 | 0.267 | 0.519 | **0.456** | ✅ DOCS tăng mạnh; ART phẳng |
| v6 | + phân rã câu hỏi (IMPROVE#2) | *(chưa nộp)* | | | | test: cơ chế chạy, còn trôi dạt do thiếu data |
| **v7** | **+ mở corpus luật nền tảng (IMPROVE#3)** | **0.439** | 0.300 | 0.549 | **0.484** | ✅ best trước v8 |
| **v8** | **+ đổi embed → AITeamVN-v2 (IMPROVE#4)** | **0.4475** | — | — | — | ✅ tăng từ v7; A/B embed: R@1 0.80 vs bge-m3 0.52 |
| **v9** | **+ phân rã câu hỏi full (IMPROVE#2, 1029 câu)** | **0.4511** | 0.280 | 0.574 | **0.4606** | recall+recall, precision vẫn thấp |
| **v10** | **+ LLM-judge CoT (Qwen lọc điều nhiễu)** | **0.4609** | 0.388 | 0.525 | **0.4552** | ✅ best ART_F2; judge tăng P nhưng giảm R |
| **v11** | **+ wide retrieval top-20 + judge tích hợp** | *(đang chạy ~550/2000)* | | | | judge CoT + wide top-20; answer vẫn từ v8 |
| **v12** | **+ sinh lại answer từ articles đã chốt (few-shot)** | *(chờ v11)* | | | | answer grounded; dự kiến tăng ROUGE |
| **v13** | baseline TRƯỚC overhaul (multi-query, data cũ) | 0.4657 | 0.2683 | 0.6320 | 0.4654 | mốc so sánh |
| **v14** | **🔥 OVERHAUL DATA: chunk theo cấu trúc + bỏ overlap + normalize `khỏan→khoản` + drop rác "Không số" + re-embed 285k** | **0.5170** | **0.2883** | **0.7153** | **0.5203** | recall +0.083 & precision +0.020 (cả 2 đều tăng) |
| **v15** | **+ effect-status dedup (bỏ bản CŨ trùng tiêu đề, giữ mới nhất) — post-process, không re-run** | **0.5255** | **0.2983** | **0.7153** | **0.5306** | best ART_F2; P +0.010, **recall GIỮ NGUYÊN (0 gold mất)** → bản cũ = nhiễu thuần |
| **v16** | **+ LLM-judge song song (Qwen lọc lạc-lĩnh-vực, 1.77 điều/câu)** | 0.5285 | **0.4400** | 0.5937 | **0.5695** | judge QUÁ STRICT: P +0.142 nhưng **R −0.122** → ART_F2 phẳng; **DOCS_F2 +0.039** ✓ |
| **v17** | **+ judge MỀM (giữ top-2 ∪ CÓ) — cứu recall, post-process từ cache** | **0.5716** | 0.3933 | 0.6937 | **0.5754** | ✅✅ **BEST**; +0.046 vs v15. "Bảo vệ top-rank, đừng đổi recall 1:1" đúng |
| v18 | judge mềm keep_top=3 | 0.5380 | 0.3317 | 0.7003 | 0.5369 | ⬇ thua v17: rank-2 thêm vào chủ yếu là nhiễu → **keep_top=2 tối ưu** |
| **v19** | **+ drop 65 VB CŨ hết hiệu lực (audit 6-agent toàn 2000 câu)** | **0.5877** | **0.4367** | 0.6903 | **0.5963** | ✅✅ **BEST, SÁT LEADER**; P +0.043, recall −0.003 (giữ) → bản cũ = nhiễu (xác nhận lần 3) |
| **v20** ⭐ | **+ blacklist mở rộng (90 VB) + scope filter BQP/BCA/BNV/NHNN** | **0.5985** | **0.4633** | 0.6903 | **0.6081** | ✅✅ **BEST, NGANG LEADER**; P +0.027, recall giữ. Chỉ bằng data + hậu xử lý, KHÔNG fine-tune |
| v22 | ② phân rã SÂU (cap 4→5) + retrieve lại + judge + clean | 0.5048 | 0.3363 | 0.6480 | 0.5404 | ❌ **THẤT BẠI** −0.094: subquery LLM viết lại TRÔI DẠT khỏi neo pháp lý → câu gốc retrieve tốt hơn. Recall GIẢM |
| **v24_collapse** | **gộp PHIÊN BẢN (cùng họ→giữ năm mới nhất), recall-safe; augment trên v20** | **0.5997** | **0.4733** | **0.6903** | **0.6092** | ✅ best nhẹ; bỏ 78 điều bản cũ (68 câu) → **P +0.010, recall GIỮ CHÍNH XÁC 0.6903**. F2 chỉ +0.0012 → **precision-only vô dụng, RECALL mới là đòn**. QA vẫn 0.0 |
| **v24_penalty** | **+ bơm vế chế tài ① (whitelist NĐ xử phạt theo lĩnh vực) trên v24_collapse** | **0.603** | **0.4733** | **0.6937** | **0.6154** | ✅ best; thêm 391 điều/220 câu → **P GIỮ 0.4733 (điều thêm ~47% gold, KHÔNG nhiễu)**, R +0.0034. **NGANG/VƯỢT leader ART_F2 (~0.592); P>leader. Gap còn lại = RECALL (−0.03)**. Đòn nhắm-mục-tiêu nhích nhỏ → cần đòn TỎA RỘNG |

## Mục tiêu (leader)
ART_F2 **0.592** (P 0.462, R 0.725) · DOCS_F2 **0.633** (P 0.470, R 0.783).
v20 đã NGANG leader. Đòn còn lại (sau khi loại phân rã — xem Review #5):
1. **Sửa judge** — cấm loại điều luật-khung/quyền (recall, an toàn, làm ngay).
2. **Mở rộng top-k câu gốc + reranker** — câu cá sâu KHÔNG trôi dạt (thay cho phân rã).
3. **Fine-tune reranker** — phá trần thật, khuếch đại đường single-query v20.
> ✅ **Phân rã NÔNG GIỮ** (v20 đã dùng `subqueries.json` max 4 vế, max_k=4 — là best).
> ❌ **Phân rã SÂU LOẠI** (v22 = `subqueries_deep.json` + max_k 6 → −0.094): subquery sâu
> trôi dạt khỏi neo pháp lý + max_k cao nhồi điều. Điểm ngọt = nông, đừng sâu hơn. Review #5.

## Diễn giải các bước

- **v2→v3:** F2 ưu tiên recall nên thử nhồi 10 điều/câu → precision sập 0.28→0.13, F2 GIẢM.
  Bài học: gold chỉ ~1.3-1.9 điều/câu, nhồi điều phản tác dụng.
- **v3→v4:** thêm BM25 (lexical) + cross-encoder rerank (bge-reranker-v2-m3) + **chọn thích
  ứng** theo độ tự tin reranker (mỗi câu một số điều) thay vì K cố định. ART_F2 0.33→0.41.
- **v4→v5:** corpus có nhiều thế hệ cùng luật (Luật DN 1999/2005/2014/2020) chấm đều ~1.0 →
  submit kèm bản hết hiệu lực → **gộp họ luật chính, giữ bản mới nhất**. DOCS_F2 +0.035;
  ART_F2 phẳng (điều sai phần lớn là "liên quan-nhưng-không-gold", không phải bản cũ).
- **v5→v6:** 40% câu là tình huống nhiều vế → **phân rã** thành câu con, retrieve từng vế,
  hợp nhất. Test 199 câu khó: cơ chế đúng (Q1726 rác→điều thật) nhưng **trôi dạt** ở câu
  dân sự (Q596) — lộ ra corpus thiếu luật nền tảng → ưu tiên v7 trước.
- **v6→v7:** mở corpus (+119 luật nền tảng: BLDS, Quảng cáo, BVQLNTD, GDĐT, AI). Sau đó
  phân rã mới hết trôi dạt. Xem `docs/DIAGNOSIS.md`, `docs/DATA_PROVENANCE.md`.

- **v7→v8:** Swap embedding từ bge-m3 sang AITeamVN/Vietnamese_Embedding_v2 (pháp lý VN).
  A/B test: R@1 0.80 vs 0.52. BM25 index giữ nguyên (lexical không phụ thuộc embedding).
- **v8→v9:** 1029/2000 câu được phân rã thành 2-4 câu con → retrieve từng câu con → merge max-score.
  Recall local tăng (0.90) nhưng precision giảm (0.28 → không lọc). BTC: +0.004 ART_F2.
- **v9→v10:** LLM-judge (Qwen3.5 + CoT) đọc toàn văn từng điều → CÓ/KHÔNG → lọc nhiễu.
  P tăng 0.28→0.39; R giảm 0.57→0.53 (judge hơi strict). BTC: +0.010 ART_F2 (best ART).
- **v10→v11:** Wide retrieval top-20 (không reranker) + judge tích hợp. Mục tiêu: recall tăng
  (gold ở rank 9-15 vẫn tìm thấy), judge giữ precision. Answer vẫn từ v8 (chưa regrounding).
- **v11→v12:** Sinh lại answer từ articles đã chốt + few-shot examples. Mục tiêu: tăng ROUGE
  (answer grounded trực tiếp trên điều luật, không phải v8's stale answer).
- **v13→v14 (OVERHAUL DATA — đòn lớn nhất tới giờ, +0.051 ART_F2):** rà & làm lại tầng data:
  + Chunk lại theo CẤU TRÚC Điều→Khoản→Điểm (`ingest/chunk.py`), bỏ overlap (BTC chấm theo Điều),
    header lấy từ text (hết lỗi `\n` giữa từ "họ⏎at"), chặn điều quái vật (930k→1 chunk), id md5.
  + `backend/textnorm.py`: chuẩn hoá dấu thanh `khỏan→khoản` (48% corpus), áp ĐỐI XỨNG
    corpus + query → BM25 hết trượt từ pháp lý phổ biến nhất.
  + Drop ký hiệu rác "Không số"/rỗng (155 VB trộn nhầm), gom theo `(số ký hiệu, title)`.
  + Re-embed 285.020 chunk (sort độ dài tăng tốc) + rebuild BM25 285k.
  + **Cả recall LẪN precision đều tăng ở cả ART & DOCS — lợi ích sạch. Chi tiết: `POST_SUBMISSION_REVIEW.md` Review #3.**

## Pseudo-gold local eval (2026-06-12, 1927 câu từ v2-v10 voting)
| Version | F2_est | P_est | R_est | Zero-R% |
|---------|--------|-------|-------|---------|
| v7 | 0.897 | 0.684 | 0.996 | 0.3% |
| v8 | 0.890 | 0.681 | 0.987 | 0.9% |
| v5 | 0.856 | 0.655 | 0.949 | 4.7% |
| v4a | 0.813 | 0.607 | 0.911 | 5.9% |
| v9 | 0.772 | 0.537 | 0.900 | 8.4% |
| v10 | 0.715 | 0.681 | 0.763 | 13.0% |
*Lưu ý: Pseudo-gold build từ chính các submission → circular bias. BTC rank ngược (v10 > v9 > v8) do tổng quát hóa tốt hơn.*

## Vị thế (tham chiếu leaderboard)
- Đội dẫn đầu ART_F2 ~0.59-0.80 (cập nhật 1 tuần trước).
- Nút thắt: precision-điều (xếp đúng điều lên top) + answer quality (ROUGE).
- Hướng vượt: wide retrieval (recall) → judge CoT (precision) → few-shot answer (ROUGE).
