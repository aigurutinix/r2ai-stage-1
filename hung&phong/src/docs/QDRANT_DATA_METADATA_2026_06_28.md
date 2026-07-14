# Qdrant Data Metadata Audit - 2026-06-28

## Active Collections

Qdrant is served by Docker container `qdrant` on `http://localhost:6333`.
Port `6333` is the REST API, not a rich web UI. Useful browser/API endpoints:

- `http://localhost:6333/collections`
- `http://localhost:6333/collections/vbpl_aiteam`
- `http://localhost:6333/collections/vbpl_aiteam_meta_20260628`

Collections after cleanup:

- `vbpl_aiteam`
  - Original competition collection used by recent submission scripts.
  - `points_count = 285020`
  - vector size `1024`, cosine, on-disk vectors.
  - Used by `scripts/build_full_v50_v47v49.py` and `scripts/build_v66_arch_pipeline.py` via:
    - `QDRANT_COLLECTION=vbpl_aiteam`
    - `BM25_INDEX_PATH=data/bm25_vbpl_aiteam.pkl`

- `vbpl_aiteam_issue_date`
  - Copy of `vbpl_aiteam`, created on 2026-06-28 from Qdrant snapshot.
  - `points_count = 285020`
  - Earlier enrichment target. Do not use for official submissions; this collection had queue/optimization confusion during the first attempt.

- `vbpl_aiteam_meta_20260628`
  - Correct working copy created from snapshot of the old `vbpl_aiteam`, then enriched in-place.
  - `points_count = 285020`
  - vector size `1024`, cosine, on-disk vectors.
  - Status after audit: `green`, `indexed_vectors_count = 285020`, optimizer `ok`.
  - Metadata enrichment:
    - `284565` chunks got `ngay_ban_hanh` from `tmquan/vbpl-vn.issue_date`.
    - `455` chunks from 11 manual `.docx` files got `ngay_ban_hanh` from the official document header date.
  - Matching BM25 index: `data/bm25_vbpl_aiteam_meta_20260628.pkl`.
  - This is the collection to use for the next metadata-aware pipeline experiment:
    - `QDRANT_COLLECTION=vbpl_aiteam_meta_20260628`
    - `BM25_INDEX_PATH=data/bm25_vbpl_aiteam_meta_20260628.pkl`

- `vbpl_aiteam_meta_parsefix_20260628`
  - Snapshot copy of `vbpl_aiteam_meta_20260628` with two conservative parser/chunk repairs.
  - `points_count = 285002`
  - Status after audit: `green`.
  - Patched documents:
    - `125/2020/NĐ-CP`: recovered missing `Điều 29`.
    - `41/2024/QH15`: recovered missing `Điều 38`.
  - Matching BM25 index: `data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl`.
  - Use for parsefix experiments:
    - `QDRANT_COLLECTION=vbpl_aiteam_meta_parsefix_20260628`
    - `BM25_INDEX_PATH=data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl`
  - Details: `docs/PARSE_CHUNK_FIX_2026_06_28.md`.

- `vbpl_aiteam_clean_20260628`
  - Clean rebuild created on 2026-06-28 from the current `tmquan/vbpl-vn` cached corpus and manual docs.
  - `points_count = 216148`
  - vector size `1024`, cosine, on-disk vectors.
  - Embedding model: `AITeamVN/Vietnamese_Embedding_v2` via sentence-transformers local GPU.
  - Build command used wrapper `scripts/run_vbpl_ingest_quiet.py` to suppress repeated HTTP request logs.
  - Matching BM25 index: `data/bm25_vbpl_aiteam_clean_20260628.pkl`.
  - Important: this was the wrong branch for the user's requested operation because it rebuilt from the current ingest corpus instead of copying the proven old `vbpl_aiteam`. Do not use it for submission comparison against v50/v56 unless explicitly testing a fresh corpus rebuild.

Deleted:

- `vbpl_clean`
  - Removed on 2026-06-28 to avoid collection confusion.

## Important Caveat

Before the backup-copy rule was clarified, an async enrichment command was started against `vbpl_aiteam`.
It added `ngay_ban_hanh` and marker `metadata_enriched_issue_date = "tmquan/vbpl-vn.issue_date"` to part of the original collection.

This did not change vectors, text, ids, or article metadata such as `so_ky_hieu` and `dieu_so`, but it means `vbpl_aiteam` is no longer strictly untouched at the payload level.

If we need to restore the original payload style for `vbpl_aiteam`, wait until `update_queue.length == 0`, then clear:

- `metadata_enriched_issue_date`
- `ngay_ban_hanh` back to empty string

for points carrying the marker.

## Payload Fields

Observed payload keys in Qdrant:

- `doc_id`
- `so_ky_hieu`
- `loai_van_ban`
- `co_quan_ban_hanh`
- `ngay_ban_hanh`
- `ngay_hieu_luc`
- `tinh_trang_hieu_luc`
- `linh_vuc`
- `title`
- `dieu_so`
- `dieu_tieu_de`
- `khoan_so`
- `text`
- `char_len`
- `source_url`
- `nguon`
- `nam`

Payload schema indexes currently include:

- `doc_id`
- `so_ky_hieu`
- `loai_van_ban`
- `co_quan_ban_hanh`
- `tinh_trang_hieu_luc`
- `linh_vuc`
- `dieu_so`
- `ngay_ban_hanh`

`source_url` was not indexed during the 2026-06-28 enrichment attempt. Updating by `source_url` is therefore slow because Qdrant has to scan.

## Metadata Coverage

The original exported corpus files:

- `data/corpus_vbpl_v2/articles.parquet`
- `data/corpus_vbpl_v2/documents.parquet`
- `data/corpus_vbpl_v2/manifest.json`

do not contain reliable effective-status fields.

`tmquan/vbpl-vn` HF cache has `issue_date`, and this can be used to fill `ngay_ban_hanh`.
The source does not provide:

- `ngay_hieu_luc`
- `ngay_het_hieu_luc`
- reliable `tinh_trang_hieu_luc`
- amendment/replacement graph
- original/supplement/consolidated-document relationship

Therefore we should not infer "current law as of March 2026" from Qdrant metadata alone.
For version-aware ranking we still need a separate version graph.

## 2026-06-28 Final Audit For `vbpl_aiteam_meta_20260628`

Source-copy integrity check against `vbpl_aiteam`:

- source points: `285020`
- copy points: `285020`
- point IDs missing in copy: `0`
- point IDs extra in copy: `0`
- distinct `so_ky_hieu` keys in source/copy: `20406` / `20406`
- missing `so_ky_hieu` keys in copy: `0`
- extra `so_ky_hieu` keys in copy: `0`
- documents with different chunk counts between source and copy: `0`

Scroll audit over all Qdrant points:

- points: `285020`
- unique `doc_id`: `21111`
- unique non-empty `source_url`: `21113`
- empty `text`: `0`
- missing required article/doc fields:
  - `doc_id`: `0`
  - `so_ky_hieu`: `0`
  - `loai_van_ban`: `0`
  - `title`: `0`
  - `dieu_so`: `0`
  - `text`: `0`
  - `ngay_ban_hanh`: `0`
  - `metadata_enriched_issue_date`: `0`
- `source_url` missing: `455`, all from the 11 manual `.docx` files. This is expected because those manual records were imported from local official docx files and did not carry real source URLs. Do not invent `source_url`; use `metadata_issue_date_source_file` / `nguon` for provenance.

Manual docs with blank `source_url`:

- `07/2022/NĐ-CP`: `80`
- `07/2022/QH15`: `66`
- `20/2026/TT-BTC`: `65`
- `125/2020/NĐ-CP`: `61`
- `81/2018/NĐ-CP`: `57`
- `157/2025/NĐ-CP`: `51`
- `68/2026/NĐ-CP`: `37`
- `132/2026/NĐ-CP`: `13`
- `135/2020/NĐ-CP`: `10`
- `122/2020/NĐ-CP`: `10`
- `141/2026/NĐ-CP`: `5`

Manual issue dates filled from document headers:

- `125/2020/NĐ-CP`: `2020-10-19`
- `122/2020/NĐ-CP`: `2020-10-15`
- `20/2026/TT-BTC`: `2026-03-12`
- `135/2020/NĐ-CP`: `2020-11-18`
- `157/2025/NĐ-CP`: `2025-06-25`
- `81/2018/NĐ-CP`: `2018-05-22`
- `07/2022/NĐ-CP`: `2022-01-10`
- `07/2022/QH15`: `2022-06-16`
- `68/2026/NĐ-CP`: `2026-03-05`
- `141/2026/NĐ-CP`: `2026-04-29`
- `132/2026/NĐ-CP`: `2026-04-06`

Chunk/data caveat:

- The copy preserves the old `vbpl_aiteam` exactly, then adds metadata. It does not fix inherited parser/chunk gaps.
- Known inherited chunk gaps from earlier audits still exist, especially legal docs where article splitting missed some `Điều` boundaries or merged content into adjacent chunks. Examples already noted in `docs/DATA_AUDIT.md` and `docs/POST_SUBMISSION_REVIEW.md`: `122/2021/NĐ-CP`, `17/2022/NĐ-CP`, `274/2025/NĐ-CP`, `125/2020/NĐ-CP` missing/mis-splitting around Điều 29, and `41/2024/QH15` around Điều 38.
- Therefore metadata coverage is now complete, but article-level chunk quality is not guaranteed 100%. Any big score jump above the current ceiling likely still needs parser/chunk repair or a version/effectiveness graph, not only metadata.

## Enrichment Scripts

Created:

- `scripts/copy_qdrant_collection.py`
  - Copies a Qdrant collection point-for-point.
  - Snapshot recovery was faster than scroll/upsert for `vbpl_aiteam -> vbpl_aiteam_issue_date`.

- `scripts/enrich_qdrant_issue_date.py`
  - Reads `issue_date` from the original `tmquan/vbpl-vn` Arrow cache.
  - Updates Qdrant payload field `ngay_ban_hanh`.
  - Adds marker `metadata_enriched_issue_date = "tmquan/vbpl-vn.issue_date"`.
  - Does not fill `ngay_hieu_luc` or `tinh_trang_hieu_luc`.

- `scripts/run_vbpl_ingest_quiet.py`
  - Thin wrapper around `ingest.run_vbpl.run`.
  - Keeps the same ingest logic but lowers noisy HTTP/client loggers.
  - Used to rebuild `vbpl_aiteam_clean_20260628`.

## Recommended Next Steps

1. Use the parsefix branch for the next retrieval experiment if we want the repaired `125/2020 Điều 29` and `41/2024 Điều 38`:
   - `QDRANT_COLLECTION=vbpl_aiteam_meta_parsefix_20260628`
   - `BM25_INDEX_PATH=data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl`
2. Use the metadata-only copy as rollback/comparison:
   - `QDRANT_COLLECTION=vbpl_aiteam_meta_20260628`
   - `BM25_INDEX_PATH=data/bm25_vbpl_aiteam_meta_20260628.pkl`
3. Keep `vbpl_aiteam` as source baseline.
4. Ignore `vbpl_aiteam_clean_20260628` and `vbpl_aiteam_issue_date` unless explicitly comparing alternate data branches.
5. For real score gains, add a separate legal version graph and repair high-impact chunk/parser gaps instead of relying only on `nam` or `ngay_ban_hanh`.
