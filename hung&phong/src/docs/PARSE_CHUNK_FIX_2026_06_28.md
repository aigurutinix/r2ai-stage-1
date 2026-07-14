# Parse/Chunk Fix - 2026-06-28

## Goal

Investigate inherited parser/chunk gaps in the active legal RAG data without replacing the proven `vbpl_aiteam` baseline blindly.

## Parser Fixes

Updated `ingest/parse_vbpl.py`:

- Accept article headers without punctuation after the article number, for example:
  - `Điều 29 Xử phạt hành vi...`
- Accept the common OCR typo `Điêu N.` as an article header.
- Keep guards against references such as `Điều 4 Nghị định này`, `Điều 51 Luật ...`, `Điều 6 và ...`.

## Conservative Corpus Branch

Created:

- `data/corpus_vbpl_v2_parsefix_20260628`

This branch starts from the full existing corpus `data/corpus_vbpl_v2`, then patches only two high-confidence documents:

- `125/2020/NĐ-CP`
  - Before: 46 distinct articles, missing `Điều 29`.
  - After: 47 distinct articles, no missing sequence from 1 to 47.
  - Root cause: source header was `Điều 29 Xử phạt...` without a dot.
- `41/2024/QH15`
  - Before: 140 distinct articles, missing `Điều 38`.
  - After: 141 distinct articles, no missing sequence from 1 to 141.
  - Root cause: source had OCR typo `Điêu 38.`.

Corpus size:

- old branch: `205044` article rows, `21574` documents.
- parsefix branch: `205046` article rows, `21574` documents.

Backup before export:

- `data/_corpus_vbpl_v2_backup_20260628_parsefix_before`

Temporary full re-export from the older export script:

- `data/corpus_vbpl_v2_parsefix_20260628_export`
- Not used as the main corpus because its scope is smaller: `14418` documents / `167602` article rows.

## Qdrant Branch

Created by snapshot-copying `vbpl_aiteam_meta_20260628`:

- `vbpl_aiteam_meta_parsefix_20260628`

Then replaced only the two patched documents:

- Deleted old chunks:
  - `125/2020/NĐ-CP`: 61 chunks.
  - `41/2024/QH15`: 163 chunks.
- Upserted new chunks:
  - `125/2020/NĐ-CP`: 53 chunks, 47 distinct articles.
  - `41/2024/QH15`: 153 chunks, 141 distinct articles.

Final collection audit:

- points: `285002`
- status: `green`
- required metadata missing: `0`
- `125/2020/NĐ-CP Điều 29`: present.
- `41/2024/QH15 Điều 38`: present.

Matching BM25:

- `data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl`

Use this pair for parsefix experiments:

- `QDRANT_COLLECTION=vbpl_aiteam_meta_parsefix_20260628`
- `BM25_INDEX_PATH=data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl`

## Retrieval Smoke Test

With hybrid retrieval against the parsefix branch:

- Query: `hành vi lập gửi thông báo báo cáo về hóa đơn quá hạn bị xử phạt theo điều nào`
  - Top 1: `125/2020/NĐ-CP Điều 29`.
- Query: `chậm đóng bảo hiểm xã hội bắt buộc bảo hiểm thất nghiệp được quy định tại điều nào của Luật BHXH 2024`
  - Top 1: `41/2024/QH15 Điều 38`.

## Data Poke Check

Direct Qdrant inspection after patch:

- `125/2020/NĐ-CP Điều 28/29/30/47`: present, clean article boundaries, issue date `2020-10-19`.
- `41/2024/QH15 Điều 37/38/39/141`: present, clean article boundaries, issue date `2024-06-29`.
- First 5000-point sample:
  - missing required metadata: `0`
  - empty text: `0`
  - `char_len` mismatch: `0`
  - article header mismatch: `0`

Remaining inherited issue:

- The parsefix branch still has `92` chunks with `char_len >= 12000`, inherited from old data outside the two patched documents.
- Examples: `334/QĐ-BTC Điều 25`, `19/2015/TT-NHNN Điều 10`, `31/1999/QĐ-BGD Điều 3`, `99/2021/NĐ-CP Điều 56`.
- These need a broader re-chunk/re-embed pass if they are relevant to scoring. They are not caused by this parsefix patch.

## Not Patched Yet

Not included in the conservative patch:

- `17/2022/NĐ-CP`
- `70/2025/NĐ-CP`

Reason: these are amendment documents with many embedded amended-article headers inside a small number of amending articles. Splitting them into standalone `dieu_so` values may improve retrieval for some questions, but it changes citation semantics. This needs a separate design: either keep citation as the amending article and add embedded target-article metadata, or create an amendment-aware side index.

## Important Caveat

The original `data/corpus_vbpl_v2` and `vbpl_aiteam_meta_20260628` were restored/kept as baseline. The parsefix changes are isolated in new branch names.
