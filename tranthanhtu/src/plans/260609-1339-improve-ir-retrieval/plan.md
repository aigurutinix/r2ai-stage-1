# Cải thiện IR retrieval (F2) — toàn diện

**Bối cảnh:** Bài nộp đầu tiên (real 50-Q dev gold): ARTICLES_F2 **0.317** (P0.41/R0.32), DOCS_F2 **0.377** (P0.42/R0.38). QA=0 (chưa chấm, đúng spec). Mục tiêu: kéo F2 lên bằng recall (F2 nặng recall 2×) + đúng luật hơn.

**Chẩn đoán (data từ retrieved.json):**
- 87% câu chỉ trả 1 điều (avg 1.19) → cutoff quá chặt → giết recall.
- ~60% câu ra sai văn bản (DOCS P/R ~0.4) → retrieval quality + coverage.
- Không có gold thật → dùng **leaderboard (10 bài/ngày, tới 30/06)** làm oracle.

## Phases

### Phase 1 — Tách cutoff khỏi retrieval (backbone) ✅
Lưu top-N candidate KÈM điểm rerank → cutoff thành bước rẻ, sweep offline (không GPU).
- [x] `backend/retrieval_cutoff.py` — hàm cutoff thuần (dùng chung engine/submission/sweep)
- [x] config: nới cutoff mặc định (top6/margin6/min None) + `RETRIEVE_CAND_SAVE=12`
- [x] `local_rag_engine` → gọi `apply_cutoff` chung
- [x] Kaggle Phase A: lưu top-12 + `score` (bỏ cutoff inline)
- [x] Kaggle gen (B1b/B2): áp cutoff qua `get_ctx` (CUT_TOP_K/MARGIN/MIN ở cell 5b)
- [x] `generate_submission --retrieved`: áp cutoff (cache giàu)
- [x] `scratch/sweep_cutoff.py`: sinh biến thể results_*.json (giữ prose LLM, gắn lại căn cứ)
- [x] + **weighted RRF (Phase 3 #1)** bundle vào Phase A: BM25 0.65 / dense 0.35 (re-run 1 lần được cả 2)
- [x] validate: py_compile OK, notebook 0 syntax err, smoke-test cutoff/reattach OK

**→ User chạy lại Phase A 1 lần (rich + weighted RRF) → Phase B (base answers) → sweep cutoff → nộp ~5 biến thể → chọn tốt nhất.**

### Phase 2 — Coverage (data) ⏳ — NÚT THẮT SỐ 1 (xác nhận qua leaderboard)
Bằng chứng: sweep cutoff đỉnh 0.3877 ở 2-3 điều/câu (nới = tụt) + **57% câu rerank top-1 ≤ 0** (không khớp tự tin) + corpus chỉ 32 vb. Report: `researcher-260609-1506-ir-bottleneck-coverage-diagnosis.md`.
- [x] Chẩn đoán: coverage 32 vb quá hẹp là trần recall.
- [x] Chốt hướng (user): **quét rộng vbpl-vn theo chủ đề SME** (`build_corpus --mode keywords`).
- [x] Mở rộng `SME_TITLE_KEYWORDS` (12 → 35 từ khóa: + việc làm/tiền lương/công đoàn/ATVSLĐ/BHYT/BHTN/lệ phí/kiểm toán/cạnh tranh/quảng cáo/NTD/XNK/hải quan/chứng khoán/xây dựng/nhà ở/BĐS/môi trường/PCCC/ATTP/xử phạt VPHC).
- [x] Build keywords (local stream 158K) → 248K điều thô.
- [x] Phân tích: 53% là Quyết định + Nghị quyết (nhiễu hành chính) → **lọc còn 93K điều / 5131 vb** (Luật/Bộ luật/NĐ/TT/Pháp lệnh). Bỏ idea dedup-version (clean_name cắt cụt → nguy hiểm).
- [x] `KEEP_LEGAL_TYPES` bỏ QĐ/NQ (build sau ra 93K trực tiếp).
- [x] **Build trên Kaggle** (Phase 0 cell: clone repo + stream) — tránh upload 209MB qua mạng yếu. Committed.
- [x] User chạy Kaggle corpus 93K → retrieved.json (time2). Top-1 rerank: median −1.05→**0.11**, tự tin >0: 42.7%→**50.6%**.
- [x] **Leaderboard vòng corpus-93K: ARTICLES_F2 0.3877 → 0.4616 (+19%)**, DOCS_F2 0.492. Recall vượt precision (R0.51/P0.41) → nút thắt mới = precision.
- [ ] (sau) targeted vbpl.vn scrape cho vb null-markdown.

### Phase 2b — Lọc HIỆU LỰC văn bản (precision) ⏳
Report: `researcher-260610-1155-validity-filter-precision-tuning.md`. Dataset KHÔNG có trường hiệu lực → heuristic 2 lớp trong `retrieval_cutoff.py`:
- [x] `SUPERSEDED_DOCS` curated 18 cặp (LDN 68/2014→59/2020, BLLĐ 10/2012→45/2019, BHXH 58/2014→41/2024, Đất đai 45/2013→31/2024, TCTD 47/2010→32/2024...) — đo được 1543 candidate / 735 câu dính.
- [x] `drop_superseded()`: curated drop + same-name-keep-newest (trong top-12/câu), never-empty, lọc TRƯỚC cutoff.
- [x] Sweep vòng 2: `f_t3m3 / f_t4m35 / f_t5m4 / f_t6m6` + `raw_t4m35` (đối chứng). Bản f_ sạch 100% superseded; raw còn 440.
- [x] **Vòng 2 leaderboard: filter hiệu lực THẮNG** — f_t3m3 **0.4887** (P0.467/R0.519, DOCS_F2 0.5162) vs t3m3 0.4616 (+0.027); A/B sạch f_t4m35 0.4705 vs raw_t4m35 0.4544 (+0.016). Đỉnh vẫn cutoff chặt t3m3.
- [x] Bake `drop_superseded` + cutoff t3m3 vào notebook Kaggle (cell 5b get_ctx) → Phase B dùng config thắng.
- [x] Sweep vòng 3 (vi chỉnh quanh t3m3): `f_t2m25` (1.64 đ/c) / `f_t3m2` (1.92) / `f_t3m4` (2.38) / `f_t3m3_sib` (2.67 đ/c, same docs — test giả thuyết DOCS_R>ART_R = thiếu điều cùng văn bản).
- [x] **Vòng 3: f_t3m2 = 0.4975 ĐỈNH MỚI** (P0.49/R0.519, DOCS_F2 0.5348) — margin 2.0 tăng precision +0.023, recall giữ nguyên. **sib-expand BỊ BÁC** (0.4766: P sập 0.467→0.418, R chỉ +0.007 → điều "anh em" kéo thêm đa số sai).
- [x] Chốt f_t3m2 → bake vào notebook (cell 5b) + backend config (TOP_K=3, MARGIN=2.0). Cutoff KỊCH TRẦN trên phễu CAND-20.
- [x] HyDE BỊ LOẠI (user): embed văn bản LLM bịa = rủi ro ảo giác + không hợp sản phẩm thật. Thay bằng:

### Phase 4 — Mở phễu rerank + (tùy chọn) query rewriting ⏳
Trần recall 0.519 = giới hạn của top-12 từ phễu RRF-20. Điều đúng hạng 21-50 chưa bao giờ được rerank.
- [x] Notebook Phase A: **CAND 20→50** (rerank 50 ứng viên/câu, ~+10-15p; corpus_emb tái dùng — corpus KHÔNG đổi).
- [x] Sweep GRID vòng 4: `c50_t3m2` (anchor vs 0.4975) / `c50_t3m15` / `c50_t3m25` / `c50_t4m2`.
- [x] **Vòng 4 (time3, phễu 50): c50_t3m15 = 0.5371 ĐỈNH MỚI** (P0.563/R0.553, DOCS_F2 0.5621); c50_t3m2 = 0.5286. Phễu 50 = +0.031~0.040. Nội bộ: top-1 median 0.11→0.71, 542 câu đổi top-1 tốt hơn, 42% top-12 là candidate mới.
- [x] Phát hiện: m1.5 vs m2.0 recall Y HỆT 0.553 → điều trong khoảng margin 1.5-2.0 toàn sai → còn dư địa siết.
- [x] Bake c50_t3m15 vào notebook (CUT 3/1.5) + config (MARGIN=1.5). Sweep vòng 5 (vi-margin): `c50_t3m1` (1.57 đ/c) / `c50_t3m125` (1.69) / `c50_t2m15` (1.53).
- [x] Vòng 5 (vi-margin): không bản nào vượt → tạm chốt c50_t3m15 (0.5371).
- [x] **Vòng 6 — nới mù: THẤT BẠI** (~0.49, P sập 0.3, R chỉ ~0.6). KẾT LUẬN VÀNG: trần recall của cache top-12 ≈ 0.62-0.65 — top-1 leaderboard (R 0.7253) có gold NGOÀI cache ta → gap nằm THƯỢNG NGUỒN (phễu/coverage), không phải cutoff. Nới mù nhặt 4 đá / 1 vàng.
- [⏳] **Vòng 7 — LLM-verify (nới có kiểm soát)**: Phase V (notebook cell 5c) — Qwen chấm CÓ/KHÔNG từng candidate top-12 (~1.5-2h, checkpoint, resume từ input) → `verified.json` → `sweep_cutoff --verified` (GRID v_k4/v_k6/v_k8: top-1 luôn giữ + candidate được chấm CÓ). Kỳ vọng: P giữ ~0.5 trong khi R tiến sát trần cache → F2 0.55-0.58.
- [ ] Nếu vòng 7 chưa đủ: nâng trần cache — phễu 80 + save-20 + verify-20 (1 phiên Kaggle), hoặc multi-query decomposition; audit coverage (7611 doc keyword-match bị null markdown khi build!).
- Lưu ý: answer LLM từ Phase B (sinh trên ctx top-3) TÁI DÙNG được cho mọi cutoff — `sweep_cutoff --base results.json` giữ prose, dựng lại fields + gắn lại căn cứ.
- [ ] (tùy thời gian, sau Phase B) query rewriting runtime — đẩy recall 0.553; LLM-verify candidate — đẩy precision.

- [x] **Vòng 7 LLM-verify: ĐÃ NỘP** (answer LLM Phase B nộp + Phase V verify chạy). Chốt: chỉ xét **ARTICLES_F2** để xếp hạng.

### Phase 6 — RECALL-FIRST (đua top, chỉ ARTICLES_F2) ⏳
**So bảng top:** mình ART P0.563 (CAO NHẤT) / R0.553 / F2 0.5371 (#3). Top1 P0.46/R0.725/F2 0.5916; Top2 P0.41/R0.705/F2 0.5859. → **Cả 2 đội đổi precision lấy recall; mình thừa precision chưa tiêu.** Gap = RECALL (~0.15), bị chặn bởi pool top-12 (trần ~0.62). Khoảng cách F2 chỉ 0.055.
- [x] **Vòng 8 — phễu SÂU**: notebook Phase A CAND 50→80, save top-20 (nâng trần pool). VB verify 16→24. Sweep GRID: anchor `t3m15` + `v_k6/v_k8/v_k10/v_k12` (verify rồi nới rộng — tiêu precision thừa thành recall).
- [x] **Vòng 8 (time4, phễu 80) — RECALL-FIRST THẮNG**: t5m3 = **0.5608 KỶ LỤC** (P0.514/R0.620) > t3m15 0.5498 (P0.603/R0.563) > t6m4 0.5325 (P0.423/R0.633). ĐẢO NGƯỢC round 6: phễu sâu nâng trần recall → nới vừa (3 đ/c) giờ thắng chặt. top-1 conf 0.71→1.57. Đỉnh ~3 đ/c.
- [x] Micro-sweep quanh t5m3: t4m3/t5m25 (2.7 đ/c) / t5m35/t6m3 (3.3 đ/c) — bracket tìm đỉnh.
- [x] Phase V verify chạy xong (time4): 18% CÓ, 3.57 CÓ/câu, top-1 CÓ 58%, 421 câu 0-CÓ (giữ top-1 sàn). Sweep `v_k6` (2.90 đ/c) / `v_k8` (3.12) / `v_k10` (3.23) — ~3 đ/c, A/B sạch vs t5m3 (3.03, score-cut, 0.5608): verify-CHỌN có hơn score-CẮT cùng số điều?
- [x] **Verify THẤT BẠI**: v_k8 0.4475 (P0.39/R0.585) ≪ t5m3 0.5608 — Qwen-7B binary nhiễu hơn điểm reranker. BỎ verify. t5m3 giữ ngôi.

### Phase 9 — Rebuild corpus từ API CHÍNH THỐNG vbpl ⏳ (user cấp gateway)
API `vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{id}` (id = trailing `--{id}` ở source_url): parse Điều chuẩn (prov-article) + **effStatus chính thức** + phục hồi 7611 doc null-markdown. Verify: 100024→hết hiệu lực, 139877→còn hiệu lực, 52 điều parse sạch.
- [x] `backend/vbpl_fetch.py` + `scratch/rebuild_corpus_vbpl.py` (ids→fetch đa luồng→corpus chỉ CÒN HIỆU LỰC). `is_in_force` chỉ bỏ "hết hiệu lực toàn bộ" (recall-safe). Test 3 doc OK.
- [x] Notebook Phase 0 → vbpl rebuild.
- [ ] User Kaggle: Phase 0 (ids ~20p + fetch ~30p) → corpus mới (in-force + coverage) → Phase A **embed lại** (~1-2h, KHÔNG add emb cũ) → retrieve funnel-80 → sweep quanh t5m3 → nộp. So 0.5608.
- [ ] researcher `researcher-260615-0923-article-recall-levers.md` (đang chạy): phục hồi 7611 doc null-markdown (structure_json/extracted_json?), fine-tune reranker synthetic (mentor: 2-stage R 0.626 > BGE 0.544), gated article-spray. → quyết đòn lớn tiếp.
- [ ] Moonshot nếu phễu+verify plateau: fine-tune reranker trên cặp synthetic Qwen-sinh từ corpus (vài ngày T4, ceiling cao nhất).

**Leaderboard journey: 0.317 → 0.3877 → 0.4616 (corpus 93K) → 0.4887 (lọc hiệu lực) → 0.4975 → 0.5371 (phễu 50) → 0.5608 (phễu 80 + nới vừa t5m3) = +77%. Còn cách top-1 (0.5916) ~0.031. Đòn tiếp: verify + corpus-recovery.**

### Phase 3 — Retrieval quality ⏳ (report `researcher-260609-1340-vn-legal-retrieval-quality.md`)
Ước tính combine #1+#2+#3 → F2 0.317 → **0.41-0.46**.
- [x] #1 **weighted RRF** BM25 0.65/dense 0.35 (+0.08-0.12) — đã bundle vào Phase 1.
- [ ] #2 rerank pool 10→15-20 + softer margin (sweep ở Phase 1) + sigmoid-normalize điểm (+0.05-0.09).
- [ ] #3 **offline HyDE** — Qwen viết lại câu hỏi sang văn phong luật trước khi embed (+0.03-0.07). `backend/query_expansion.py`.
- [ ] #6 boost BM25 khi query có số luật/Điều (regex) — ít giá trị vì câu hỏi SME hiếm khi cite số.
- [ ] Giữ article-level chunking (report khuyến nghị — khớp grader mã|Điều).
- [ ] Verify format join `relevant_docs/articles` vs grader (chuẩn hóa mã văn bản).

### Phase 4 — Validation loop ⬜
- [ ] Mỗi thay đổi → 1 bài leaderboard → ghi P/R/F2 → giữ/bỏ. Ghi log biến thể (no silent caps).

## Quyết định
- vLLM bỏ (lệch CUDA-build Kaggle) → transformers fp16 2×T4.
- numpy cosine giữ (corpus nhỏ). Không ChromaDB/turbovec.

## Câu hỏi mở
1. 50 câu gold là câu nào (id 1-50? random?) — không biết → chỉ có tín hiệu tổng hợp.
2. Grader ghép article identifier từ `relevant_articles` field hay parse `answer`? (spec mơ hồ §6.1) — ảnh hưởng cách tối ưu.
