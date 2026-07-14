# Corpus đang dùng cho chatbot — vbpl_v2

**Nguồn DUY NHẤT:** HuggingFace `tmquan/vbpl-vn` — scrape từ **vbpl.vn** (CSDL quốc gia
về VBPL, Bộ Tư pháp). Không trộn nguồn khác.

## Đã lọc gì (từ 158.822 VB gốc → 21,574 VB)
1. Cấp **trung ương** (bỏ văn bản địa phương).
2. **Loại lõi**: Luật, Bộ luật, Pháp lệnh, Nghị định, Thông tư, TTLT, VBHN, Nghị quyết, Quyết định.
3. **Từ khoá DN/SME** (DN, thuế, lao động, đấu thầu, hải quan, hoá đơn, ...).

## File trong thư mục
- `documents.parquet` — sổ đăng ký từng VĂN BẢN (metadata + provenance, không kèm text).
- `articles.parquet`  — từng ĐIỀU đã index (205,044 dòng) = đơn vị trong vector DB (kèm nội dung).
- `manifest.json`     — provenance đầy đủ (so_ky_hieu, source_url vbpl.vn, text_hash) từng VB.
- `inventory.md`      — thống kê theo loại / lĩnh vực + danh sách luật gốc.

## Tái tạo
```
python scripts/export_corpus_vbpl.py          # xuất lại thư mục này
python -m ingest.run_vbpl --recreate          # ingest lại vào Qdrant (collection vbpl_v2)
```
