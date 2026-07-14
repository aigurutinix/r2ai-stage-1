# Mô tả dữ liệu (Data Description)

Tài liệu mô tả toàn bộ nguồn dữ liệu dùng trong hệ thống truy hồi điều luật R2AI Task 3.1.

---

## Tổng quan dung lượng

| Thư mục / File | Dung lượng | Nội dung |
|----------------|-----------|----------|
| `vbpl_dataset/` | 6.6 GB | Corpus văn bản pháp luật gốc (full text + chunks + metadata) |
| `data_final/` | 616 MB | QA có citation (ground truth tự xây) |
| `synthetic_qa/` | 939 MB | QA tổng hợp do LLM sinh (43 file) |
| `data_final_hard_negatives.jsonl` | 259 MB | Train data reranker (96,456 dòng) |
| `retrieval_index_dense/` | 4.8 GB | FAISS dense index (612K vectors) |
| `retrieval_index_bm25s_v7/` | 2.2 GB | BM25S index v7 (1.88M chunks) |
| `fewshot_index/` | 927 MB | FAISS index trên data_final questions |

---

## 1. `vbpl_dataset/` — Corpus văn bản pháp luật

Nguồn chính: cơ sở dữ liệu pháp luật Việt Nam. `vbpl_dataset/build.py` thu thập văn bản từ:

- **vbpl.vn** (Văn Bản Pháp Luật) — nguồn full text chính (~94% văn bản).
- **vanban.chinhphu.vn** — metadata văn bản (ItemID, hiệu lực, lược đồ).
- **luatvietnam** — nguồn full text dự phòng cho văn bản chưa map được.

### Cấu trúc

```
vbpl_dataset/
├── chunks/              ← văn bản đã chunk theo Điều (đơn vị truy hồi)
│   ├── _stats.json      ← thống kê tổng
│   ├── bo_luat/  bo_luat.jsonl      ← per-doc JSON + gộp jsonl theo loại
│   ├── luat/  luat.jsonl
│   ├── nghi_dinh/  nghi_dinh.jsonl
│   ├── thong_tu/  thong_tu.jsonl
│   └── ... (hien_phap, phap_lenh, nghi_quyet, nghi_quyet_lien_tich,
│            thong_tu_lien_tich, quyet_dinh)
├── metadata/            ← metadata theo loại văn bản (.json)
├── full_text/           ← văn bản đầy đủ (raw .txt theo loại)
├── build.py             ← crawl / build metadata + full_text
├── legal_chunker.py     ← logic chunk theo Điều / Khoản / Điểm
└── export_chunks.py     ← parse full_text → chunks/*.json + *.jsonl
```

### Thống kê (`chunks/_stats.json`)

- **total_files**: ~40,435 văn bản (corpus ~42,232 docs, ~95.7% có full text).
- **total_chunks**: ~461K điều (đơn vị truy hồi cấp Điều).
- **parse_type**: `dieu` (>32K docs), `numbered` (~7.6K), `article` (~226, văn bản tiếng Anh).
- **10 loại văn bản**: hien_phap, bo_luat, luat, phap_lenh, nghi_dinh, nghi_quyet,
  nghi_quyet_lien_tich, thong_tu, thong_tu_lien_tich, quyet_dinh.
- **luoc_do**: ~285K quan hệ giữa các văn bản (HD chi tiết, sửa đổi, bổ sung...).

### Chunking strategy (`legal_chunker.py`)

Đơn vị chunk = **Điều** (article). Tách tại marker `^Điều\s+(\d+\w*)` (hoặc `^Article\s+N`
cho văn bản tiếng Anh, hoặc số La Mã cho `numbered`). Bên trong mỗi Điều còn parse:

- **Khoản** (clause): regex `^(\d+)[-.]`
- **Điểm** (point): regex `^([a-zđ])\)`

### Cấu trúc một chunk (record trong `*.jsonl`)

```json
{
  "chunk_id": "luat/1005#dieu_2",
  "doc_id": "luat/1005",
  "doc_type": "luat",
  "parse_type": "dieu",
  "lang": "vi",
  "paywall_lines": 0,
  "article_number": "2",
  "article_title": "...",
  "path": {"phan": null, "chuong": "Chương 1 ...", "chuong_index": 1, "muc": null},
  "content": ["..."],
  "khoan": [{"so": 1, "content": "...", "diem": [{"label": "a", "content": "..."}]}],
  "char_count": 0,
  "citation_keys": ["Điều 2", "khoản 1 Điều 2", "điểm a khoản 1 Điều 2"],
  "prev_article": "1",
  "next_article": "3"
}
```

Các trường quan trọng cho retrieval: `chunk_id`, `doc_id`, `article_number`, `article_title`,
`content`, `citation_keys`.

### Metadata (`metadata/*.json`)

Chứa: tên (`doc_title` / `trich_yeu`), số hiệu (`law_id`, vd `61/2020/QH14`), loại, hiệu lực
(`hieu_luc`: `con_hieu_luc` / `het_hieu_luc` / `het_hieu_luc_1_phan` / `dinh_chi`),
`ngay_ban_hanh`, `ngay_hieu_luc`, lược đồ (`luoc_do`).

> **Lưu ý hiệu lực**: GT của benchmark **bao gồm cả articles từ văn bản đã hết hiệu lực** →
> KHÔNG được lọc theo `hieu_luc` (thí nghiệm lọc làm giảm recall -8%).

### Vấn đề `law_id` rỗng

- ~84,925 chunks (~13.9%) thiếu `law_id`: chủ yếu văn bản cũ (pre-2000, format "NĐ 40/CP" không
  parse được). Không fix được bằng mapping; impact nhỏ.

---

## 2. `data_final/` — QA có citation (ground-truth tự xây)

QA tự thu thập, mỗi câu có `article_cite` trỏ về Điều cụ thể. Đóng vai trò **cầu nối ngôn ngữ
đời thường → pháp lý** cho BM25S, **nguồn train reranker** (qua hard negatives), và **nguồn
few-shot** cho bước sinh `answer`.

### Nguồn (xem `data_final/docs.md`)

- **vbpl.vn** (Thư Viện Pháp Luật) — full text + QA theo chủ đề.
- **chinhsachonline.vn** — dataset QA.
- **vksndtc** (Viện Kiểm sát Nhân dân Tối cao) — dataset QA.

### Cấu trúc

```
data_final/
├── chinhsachonline/     ← QA từ chinhsachonline.vn
├── tvpl/                ← QA từ Thư Viện Pháp Luật (nhiều thư mục con theo chủ đề)
├── vksndtc/             ← QA từ VKSND Tối cao
├── cite_mapping.json    ← mapping citation → (doc, Điều)
└── docs.md              ← mô tả nguồn
```

### Quy mô & schema

- **163,677 câu QA**, trong đó **155,867 câu có citation** (`article_cite`).
- Schema một QA item:

```json
{
  "question": "...",
  "article_cite": [{"title": "Điều N ...", "content": "...", "doc_title": "..."}],
  "answer": "...",
  "full_answer": "...",
  "link": "...",
  "source": "chinhsachonline | vksndtc | tvpl"
}
```

- `article_cite` → tạo chunk QA bổ sung vào corpus BM25S, trỏ về article được cite.
- `full_answer` → **few-shot mẫu** cho bước sinh `answer`: với mỗi query test tìm QA tương tự
  nhất → đưa câu hỏi + câu trả lời mẫu vào prompt để hiệu chỉnh văn phong/độ chi tiết.

> `clean_text.py` sửa lỗi xuống dòng từ web crawl (vksndtc, chinhsachonline): ghép dấu thanh bị
> tách, nối câu bị ngắt giữa chừng, bỏ footer "Ban Biên tập".

---

## 3. `synthetic_qa/` — QA tổng hợp (LLM sinh)

QA do LLM sinh tự động từ nội dung các Điều luật (`code/gen_synthetic_questions.py`), nhằm mở
rộng "cầu nối ngôn ngữ" cho BM25S.

### Cách sinh

- **Input**: chunk từ `vbpl_dataset/chunks/` (nội dung Điều, cắt ≤ 1500 chars vào prompt).
- **Model**: LLM (Qwen3-8B-AWQ qua vLLM), `temperature=0.7`, `max_tokens=500`.
- **Prompt**: yêu cầu sinh 1-3 câu hỏi/điều bằng **ngôn ngữ đời thường** (không nhắc tên luật) +
  câu trả lời ngắn — tạo biến thể từ vựng để BM25S match được câu hỏi người dùng thực.

### Cấu trúc & quy mô

- **43 file JSON** `synthetic_NNNNNN_NNNNNN.json`, mỗi file ~29K record.
- Tổng QA thô lớn → sau dedup còn **~1.27M unique** (loại trùng).
- Corpus BM25S v7 cuối = 1.27M synthetic + 459K VBPL + 155K data_final QA = **~1.88M chunks**.

### Cấu trúc một record (9 trường)

```json
{
  "question": "Quyền lực ở nước mình thuộc về ai vậy bác?",
  "answer": "Tất cả quyền lực thuộc về nhân dân.",
  "chunk_id": "hien_phap/1534#dieu_6",
  "van_ban_id": "hien_phap/1534",
  "law_id": "",
  "doc_title": "Hiến pháp Không số",
  "doc_description": "Hiến pháp năm 1980",
  "article_number": "6",
  "article_title": "..."
}
```

### Insight quan trọng

- **Synthetic QA tăng recall ceiling BM25S gần gấp đôi**: BM25S gốc recall ~52% → BM25S v7
  (+1.27M synthetic) recall **93.6%**. Đây là yếu tố quan trọng nhất cho BM25S.
- **Synthetic QA chỉ hữu ích cho BM25S (keyword matching), KHÔNG cho dense embedding** —
  format `doc_title` ngắn khác VBPL gốc → thêm vào dense index làm giảm precision (-10.8%).

---

## 4. `data_final_hard_negatives.jsonl` — Train data Reranker

File train **DUY NHẤT** cho reranker fine-tuned. Sinh bởi `code/mine_hard_negatives_dense.py`.

### Cách tạo

1. Lấy mẫu **10K câu hỏi** từ `data_final` theo phân bố chủ đề khớp tập test
   (doanh_nghiep 35%, lao_dong / thue ~9%, ...).
2. Encode mỗi câu bằng Vietnamese_Embedding_v2 → dense search **top-20** trong corpus.
3. Dedup theo `(law_id, article_number)`, loại các positive (điều được cite).
4. **Positive** (label=1) = điều luật được cite; **hard negative** (label=0) = candidate
   **rank 5-12** (gần đúng ngữ nghĩa nhưng sai → "hard"). ~7 hard negatives / positive.

### Schema & thống kê

```json
{"query": "...", "passage": "Nghị định ... - khoản 1 Điều 14 ...\n1. Phạt tiền ...", "label": 1}
```

- **96,456 dòng** tổng. **9,150 query** có cả positive + negative.
- **12,057 positive** (label=1) + **84,399 negative** (label=0). Avg ~1.32 pos, ~9.22 neg/query.
- Train script tách **500 query held-out** làm eval set (xem `model_description.md` mục 3).

---

## 5. `R2AIStage1DATA.json` — Tập test R2AI

- **2,000 câu hỏi**, mỗi câu `{"id": int, "question": str}`. Không có đáp án.
- Đây là input cuối cùng để sinh submission.

### Phân bố (từ query classification)

- Chủ đề: Lao động ~22%, Doanh nghiệp ~17%, Thuế ~14%, Thương mại ~13%.
- Multi-hop: ~71%. Điểm yếu: nhóm AI/Công nghệ (văn bản rất mới chưa có trong corpus).

### Data coverage

- **Document-level**: 99.6% GT docs có trong corpus.
- **Article-level**: 97.0% GT (doc_id, Điều) có trong corpus.
- **Kết luận**: data đủ, bottleneck là **ranking quality**, không phải coverage.

---

## 6. Các index đã build

| Thư mục | Mô tả | Nội dung |
|---------|-------|----------|
| `retrieval_index_dense/` | Dense FAISS (best) | `faiss.index`, `embeddings.npy`, `metas.pkl`, `texts_count.txt`. ~612K vectors, dim=1024, L2-normalized. |
| `retrieval_index_bm25s_v7/` | **BM25S v7 (best BM25S)** | `bm25s_model/`, `metas.pkl`. ~1.88M chunks (vbpl + QA + 1.27M synthetic), tokenize tiếng Việt (underthesea), BM25 lucene (k1=1.5, b=0.75). |
| `fewshot_index/` | Few-shot QA index | `questions.faiss`, `meta.pkl`. ~155K vectors từ data_final questions (kèm cite titles cho gen_answer). |

> **Index dùng cho best pipeline**: `retrieval_index_dense/` + `retrieval_index_bm25s_v7/`.

---

## 7. Cache / artifacts trung gian

| File | Mô tả |
|------|-------|
| `query_decompose_cache.json` | Cache sub-queries (kết quả decompose) |
| `hyde_cache.json` | Hypothetical documents do LLM sinh ra cho mỗi query (bước 1 của HyDE) |
| `hyde_candidates.pkl` | HyDE candidates per query — kết quả dense search từ hypothetical docs (input cho intersection) |
| `rerank_intersection_scores.pkl` | Reranker scores trên intersection pool (input cho LLM cuối + gen_answer) |
| `submission_3_1_decompose_bm25_top150.json` | BM25S v7 decompose top-150 (input tính intersection) |
| `fewshot_search_top30.pkl` | Top-30 few-shot QA tương tự mỗi query |
| `fewshot_top5_reranked.pkl` | **Few-shot top-5 đã rerank — input cho gen_answer** |
| `submission_3_1_llm8b_rerankerv2ck8k_top5.json` | Submission cấp điều luật (output bước 5, input cho bước 5b — sinh khi chạy reproduce) |

---

## 8. Link tải dữ liệu

Toàn bộ dữ liệu đã upload lên HuggingFace:
**https://huggingface.co/datasets/NghiemAbe/r2ai-legal-data**

| Thư mục | Dung lượng | Cách lấy |
|---------|-----------|----------|
| `vbpl_dataset/` | 6.6 GB | Tải từ [NghiemAbe/r2ai-legal-data](https://huggingface.co/datasets/NghiemAbe/r2ai-legal-data) |
| `data_final/` | 616 MB | Tải từ [NghiemAbe/r2ai-legal-data](https://huggingface.co/datasets/NghiemAbe/r2ai-legal-data) |
| `synthetic_qa/` | 939 MB | Tải từ [NghiemAbe/r2ai-legal-data](https://huggingface.co/datasets/NghiemAbe/r2ai-legal-data) |
| `data_final_hard_negatives.jsonl` | 259 MB | Tải từ [NghiemAbe/r2ai-legal-data](https://huggingface.co/datasets/NghiemAbe/r2ai-legal-data) |
| `retrieval_index_dense/` | 4.8 GB | Rebuild: `python3 code/retrieval_dense.py build-index` |
| `retrieval_index_bm25s_v7/` | 2.2 GB | Rebuild: `python3 code/retrieval_bm25s.py build-index` |
| `fewshot_index/` | 927 MB | Rebuild: `python3 code/build_fewshot_index.py` |

Đặt dữ liệu vào thư mục `data/` trong submission. Xem `data/README.md` để biết chi tiết.
