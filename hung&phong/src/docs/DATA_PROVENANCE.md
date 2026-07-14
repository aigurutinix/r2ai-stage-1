# Nguồn gốc & xử lý dữ liệu (Data Provenance)

> Tài liệu truy xuất nguồn gốc corpus — phục vụ viết báo cáo cuộc thi. Ghi rõ: lấy data
> từ đâu, thu thập & xử lý thế nào, nộp cho BTC ra sao. Cập nhật 2026-06-11.

BTC **chỉ cung cấp test set** (2000 câu hỏi, không đáp án). Đội **tự thu thập corpus** từ
nguồn chính thống (thể lệ mục 5). Tài liệu này mô tả toàn bộ chuỗi data của hệ thống.

---

## 1. Nguồn dữ liệu (chính thống)

| Thuộc tính | Giá trị |
|---|---|
| **Nguồn gốc** | **vbpl.vn** — Cơ sở dữ liệu quốc gia về văn bản pháp luật, **Bộ Tư pháp** |
| **Dataset trung gian** | `tmquan/vbpl-vn` (HuggingFace) — bản scrape có cấu trúc từ vbpl.vn |
| **Quy mô gốc** | 158.822 văn bản (toàn quốc, mọi lĩnh vực, mọi cấp) |
| **Provenance từng VB** | `source_url` (link gốc vbpl.vn) + `text_hash` (toàn vẹn) — lưu trong manifest |
| **Cấp độ** | dùng `scope == "trung_uong"` (văn bản trung ương; bỏ địa phương để giảm nhiễu) |

Mỗi bản ghi gốc gồm: `doc_number` (số ký hiệu), `title`, `legal_type` (loại VB),
`issuing_authority`, `issue_date`/`year`, `legal_area` (lĩnh vực), `scope`,
`markdown` (toàn văn), `source_url`, `text_hash`.

> Lý do chọn nguồn này: vbpl.vn là CSDL **chính thống** của Bộ Tư pháp → đáp ứng yêu cầu
> "nguồn chính thống" của thể lệ; có `source_url` để kiểm chứng từng căn cứ pháp lý.

---

## 2. Lọc phạm vi (scope) — tiến hóa qua 3 phiên bản

Cuộc thi giới hạn **Luật Doanh nghiệp & văn bản liên quan SME** (thuế, lao động, hợp đồng...).
Lọc 158k VB về đúng phạm vi để (a) giảm nhiễu → tăng precision, (b) giảm thời gian embed.

| Bản lọc | Quy tắc | Kết quả |
|---|---|---|
| **v1** (cũ) | dataset `th1nhng0`, lọc thô | nhiều VB hết hiệu lực/bản cũ → F2 thấp |
| **v2** | `in_scope_vbpl`: trung_uong + loại lõi (Luật/NĐ/TT...) + **khớp từ khóa SME trên title/legal_area** | **16.008 VB · ~191.724 chunk** |
| **v3** (hiện tại) | v2 **+ vá lỗ hổng** (mục 3) | **+119 VB · +5.118 chunk → ~196.842 chunk** |

Từ khóa SME (v2): doanh nghiệp, đầu tư, thuế, lao động, hợp đồng, thương mại, kinh doanh,
kế toán, tài chính, phá sản, chứng khoán, sở hữu trí tuệ, cạnh tranh, BHXH, tiền lương,
đấu thầu, hải quan, hóa đơn, ngân sách, việc làm, hộ kinh doanh, HTX, đăng ký kinh doanh,
công đoàn, an toàn, đối tác công tư, quản lý thuế, khởi nghiệp, giá, đất đai, xây dựng.
(Định nghĩa: `ingest/scope.py`)

---

## 3. Mở rộng v2→v3: vá lỗ hổng luật nền tảng (2026-06-11)

**Phát hiện qua điều tra 2000 câu thi** (`docs/DIAGNOSIS.md`): corpus v2 **thiếu nhiều luật
nền tảng** mà ~262 câu cần, do 2 nguyên nhân:

1. **Lọc keyword trên title quá chặt** — loại oan luật mà title không chứa từ khóa SME
   (vd "Bộ luật Dân sự", "Luật Quảng cáo", "Luật Giao dịch điện tử").
2. **Bug dedup theo số ký hiệu đơn** — số hiệu Quốc hội KHÔNG unique giữa các loại VB
   (vd `91/2015/QH13` vừa là **Bộ luật Dân sự** vừa là một **Nghị quyết** giám sát QH).
   Dedup theo số hiệu đơn → giữ nhầm Nghị quyết, **che mất Bộ luật Dân sự**.

**Cách vá** (`scripts/expand_corpus.py`, bộ lọc `in_scope_v3`):
- Dedup theo **(số ký hiệu + loại VB)** thay vì số ký hiệu đơn.
- Giữ Luật/Bộ luật/Pháp lệnh khớp keyword nghiệp vụ **mở rộng** (+ dân sự, quảng cáo,
  người tiêu dùng, giao dịch điện tử, công nghệ số, trí tuệ nhân tạo, trọng tài, công chứng).
- Giữ NĐ/TT chỉ thuộc **domain mới còn thiếu** (quảng cáo, NTD, GDĐT, AI) — tránh kéo nhiễu.
- **Incremental**: chỉ embed VB mới (chưa có theo khóa (số hiệu,loại)) → upsert, KHÔNG recreate.

**Đã thêm (119 VB → 101 parse được điều → 5.118 chunk):** Bộ luật Dân sự 91/2015 (689 điều),
Luật Quảng cáo (16/2012, 75/2025), Luật Bảo vệ quyền lợi NTD (19/2023), Luật Giao dịch
điện tử (20/2023), **Luật Công nghiệp công nghệ số 71/2025 (chương AI)**, + NĐ/TT liên quan.
Provenance: `data/vbpl_v3_added_manifest.json` (mỗi VB kèm source_url).

---

## 3b. Mô hình embedding (cách lấy — yêu cầu thể lệ)

| | Cũ | **Mới (v8)** |
|---|---|---|
| Model | BAAI/bge-m3 (đa ngữ, BAAI 2024) | **AITeamVN/Vietnamese_Embedding_v2** |
| Nguồn | HuggingFace, qua Ollama | HuggingFace (sentence-transformers, GPU local) |
| Phát hành | 2024 | 17/03/2025 (Apache 2.0) |
| Base / dim | XLM-R / 1024 | finetune từ bge-m3 / 1024 |
| Tham số | 568M | 568M (<14B ✓) |

Lý do đổi: A/B trên eval_set_v2 (54 câu + gold) — AITeamVN-v2 **Recall@1 0.80 vs bge-m3 0.52**,
MRR 0.84 vs 0.64. Là bản **finetune chuyên tiếng Việt + pháp lý** trên chính bge-m3.
Re-embed sang collection `vbpl_aiteam` (giữ `vbpl_v2` bge-m3 để rollback); fp16 cho vừa
3060 12GB. Reranker vẫn bge-reranker-v2-m3 (sẽ A/B AITeamVN_Reranker/ViRanker sau).

## 4. Pipeline xử lý (data → index)

```
vbpl.vn / tmquan/vbpl-vn (markdown toàn văn)
  → parse_vbpl.py:   tách markdown → từng "Điều" (cổng số thứ tự, phân biệt header vs tham chiếu)
  → chunk.py:        1 Điều = 1 chunk (điều quá dài → tách theo Khoản, giữ header)
  → embed:           bge-m3 (1024-dim, local qua Ollama)
  → Qdrant:          collection `vbpl_v2` (embedded file://), payload kèm so_ky_hieu/dieu_so/
                     title/nam/source_url/tinh_trang_hieu_luc/text
Index truy hồi:
  → BM25  (data/bm25_vbpl_v2.pkl)            — lexical, giữ số hiệu luật khi tokenize
  → HNSW  (C:/Users/PHONG/vbpl_idx/...)      — ANN dense (hnswlib, cosine), tăng tốc retrieve
Truy hồi: hybrid (dense ∪ BM25) → rerank cross-encoder (bge-reranker-v2-m3) → gộp bản cũ → chọn
```

---

## 5. Dữ liệu trong `data/` (cập nhật đầy đủ)

| Đường dẫn | Nội dung |
|---|---|
| `data/corpus_vbpl_v2/documents.parquet` | Danh mục VB (số hiệu, loại, title, năm, lĩnh vực, source_url, n_dieu) |
| `data/corpus_vbpl_v2/articles.parquet` | Toàn bộ Điều (so_ky_hieu, dieu_so, tiêu đề, char_len) — đơn vị chấm |
| `data/corpus_vbpl_v2/README.md` + `inventory.md` | Mô tả + thống kê corpus |
| `data/vbpl_v2_manifest.json` | Manifest nguồn corpus v2 (16.008 VB) |
| `data/vbpl_v3_added_manifest.json` | Manifest 119 VB thêm ở v3 (kèm source_url) |
| `data/submission_v*.json` / `.zip` | Các bản nộp (xem `docs/SUBMISSIONS.md`) |
| `data/subqueries.json` | Câu hỏi đã phân rã (IMPROVE#2) |

> Sau mở rộng v3, parquet + inventory được **regenerate từ Qdrant** để phản ánh corpus mới.

---

## 6. Định dạng nộp cho Ban Tổ chức

Bài nộp: file JSON `results.json` (nén `.zip`), mỗi câu:
```json
{
  "id": 1,
  "question": "...",
  "answer": "... (có trích 'Điều X' rõ ràng) ...",
  "relevant_docs": ["59/2020/QH14|Luật Doanh nghiệp", ...],
  "relevant_articles": ["59/2020/QH14|Luật Doanh nghiệp|Điều 190", ...]
}
```
- `relevant_articles` (định danh `số hiệu | tên VB | Điều X`) là trường được chấm
  ARTICLES_F2; `relevant_docs` chấm DOCS_F2 (đã kiểm thực nghiệm: đổi trường này → điểm đổi).
- `answer` do Qwen3.5 sinh, grounding trên các Điều đã truy hồi.

---

## 7. Tái lập (reproduce)

```powershell
# 1. Ingest corpus v2 (lần đầu, ~8h embed)
$env:QDRANT_COLLECTION="vbpl_v2"; python -m ingest.run_vbpl --recreate
# 2. Mở rộng v3 (luật nền tảng, incremental ~17min)
python scripts/expand_corpus.py --apply
# 3. Rebuild index
python scripts/build_bm25.py ; python scripts/build_hnsw.py
# 4. Sinh submission (xem docs/SUBMISSIONS.md)
```
