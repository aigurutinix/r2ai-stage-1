# Bộ tài liệu thuyết minh sản phẩm — R2AI 2026

Hồ sơ thuyết minh sản phẩm **AI Legal Assistant** (Truy hồi & Hỏi đáp văn bản pháp luật tiếng Việt cho SME). Kết quả chính thức: **ARTICLES_F2_MACRO = 0.5766**.

## 1. Danh mục tài liệu (đủ 4 hạng mục BTC yêu cầu)

| # | Hạng mục BTC | Tài liệu | Vị trí |
|---|---|---|---|
| 1 | **Mô tả dữ liệu** (nguồn, cấu trúc/định dạng, hướng dẫn truy cập, link share) | Mô tả dữ liệu | [01-mo-ta-du-lieu.md](01-mo-ta-du-lieu.md) |
| 2 | **Mô hình sử dụng** (thông tin model, phiên bản checkpoint, hướng dẫn tải/dùng, link share) | Mô hình sử dụng | [02-mo-hinh-su-dung.md](02-mo-hinh-su-dung.md) |
| 3 | **Mã nguồn** (toàn bộ code, dependencies, tệp cấu hình) | Mã nguồn & phụ thuộc | [03-ma-nguon-va-phu-thuoc.md](03-ma-nguon-va-phu-thuoc.md) |
| 4 | **Tài liệu hướng dẫn** (README đầy đủ: cài đặt, cấu hình, chạy lại từ đầu, tái hiện) | README | [../../README.md](../../README.md) |

- **Mã nguồn (toàn bộ):** repo GitHub `lehuubao2909/ai_legal_assistant` (branch `master`).
- **Dependencies:** [`requirements.txt`](../../requirements.txt) (gốc repo) + `backend/requirements-local.txt`.
- **Đề bài + metric + kiến trúc:** [competition-overview.md](../competition-overview.md) · [pipeline-architecture.md](../pipeline-architecture.md).

---

## 2. ⚠️ HÀNH ĐỘNG TRƯỚC KHI NỘP (bắt buộc)

Các tài liệu còn **placeholder `{{...}}`** — phải điền sau khi upload. BTC yêu cầu **kiểm tra kỹ tính truy cập của link trước khi nộp**.

### 2.1 Upload dữ liệu lên Google Drive → điền `{{GOOGLE_DRIVE_LINK_DATA}}`

Tạo 1 thư mục Drive (vd `R2AI-AILegalAssistant-Data`), upload **3 file của lần chạy 0.5766** (lấy từ **Kaggle Dataset** đã dùng cho notebook, hoặc build lại bằng `build_corpus.py`+`embed_corpus.py`):

| File | Nội dung | Kích thước |
|---|---|---|
| `corpus_articles.jsonl` | Corpus HF ~93K Điều (lọc SME) | ~91 MB |
| `corpus_emb.npy` | BGE-M3 embeddings float32 N×1024 (khớp dòng với jsonl) | ~381 MB |
| `stage1_questions.json` | 2000 câu test (BTC cấp) | ~0.5 MB |

> ⚠️ **KHÔNG** dùng `data/corpus_articles.jsonl` hiện trong repo cho bundle này — đó là bản **VBPL ~41K đã revert** (điểm tệ hơn). Bản tạo ra 0.5766 là **corpus HF ~93K** trên Kaggle. Nếu không còn, regenerate: `python backend/build_corpus.py` → `python backend/embed_corpus.py` (xem README §4 Cách B).

### 2.2 Pin phiên bản model

Sản phẩm dùng **3 model gốc public trên HuggingFace** (không có checkpoint tự train). Với mỗi model, ghi rõ trong `02-mo-hinh-su-dung.md` là dùng **bản mới nhất công bố trên HuggingFace (ghim theo thời điểm nộp)**. Ví dụ `AITeamVN/Vietnamese_Embedding`.

### 2.3 Điền placeholder nhanh (sau khi có link)

```bash
cd /Users/huubao/Documents/GOKU/Dev/DEV_AI/ai_legal_assistant
# macOS sed: thay <DÁN_LINK> bằng link Drive thật (đặt sharing = "Anyone with the link")
LINK_DATA='<DÁN_LINK_DRIVE_DATA>'
grep -rl '{{GOOGLE_DRIVE_LINK_DATA}}' README.md docs/submission/ | \
  xargs sed -i '' "s|{{GOOGLE_DRIVE_LINK_DATA}}|$LINK_DATA|g"
```

Kiểm tra còn sót placeholder: `grep -rn '{{' README.md docs/submission/0[123]*.md` (phải rỗng trước khi nộp; file `00-index` cố tình nhắc tên placeholder trong hướng dẫn nên bỏ qua).

---

## 3. Checklist trước khi nộp (≤ 17h30 30/06/2026)

- [ ] Upload `corpus_articles.jsonl` (93K) + `corpus_emb.npy` + `stage1_questions.json` lên Drive; sharing = **Anyone with the link**.
- [ ] Điền `{{GOOGLE_DRIVE_LINK_DATA}}` (5 chỗ: README + 01-mo-ta-du-lieu).
- [ ] Trong `02-mo-hinh-su-dung.md`: ghi rõ dùng bản mới nhất công bố trên HuggingFace (ghim theo thời điểm nộp); không có checkpoint tự train.
- [ ] `grep -rn '{{' README.md docs/submission/0[123]*.md` → **rỗng**.
- [ ] Mở thử mọi link (Drive + HF) ở chế độ ẩn danh → tải được.
- [ ] Repo GitHub ở chế độ truy cập được cho BTC (public hoặc mời reviewer).
- [ ] `requirements.txt` + `README.md` ở gốc repo, mở được.

---

## 4. Tóm tắt giải pháp (1 đoạn cho người duyệt)

Pipeline RAG offline: corpus ~93K Điều (HF `tmquan/vbpl-vn` lọc SME, 1 Điều/dòng) → embed BGE-M3 → mỗi câu hỏi truy hồi **Dense (Vietnamese_Embedding) ‖ BM25 → RRF có trọng số → cross-encoder rerank (Vietnamese_Reranker, RERANK_MAX 512)** → cutoff `t5m3` + `drop_superseded` → `relevant_docs/relevant_articles`; Qwen2.5-7B-Instruct sinh `answer` (QA). Tất cả model mở < 14B, công bố trước 01/03/2026. **ARTICLES_F2_MACRO = 0.5766**.
