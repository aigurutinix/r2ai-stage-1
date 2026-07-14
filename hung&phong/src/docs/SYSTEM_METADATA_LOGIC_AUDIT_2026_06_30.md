# System Metadata + Logic Audit - 2026-06-30

## Why This Audit Exists

Recent work drifted into submission-level answer mutation. That is not the right long-term path.

The system should improve because retrieval, metadata, reranking, and selection are structurally better. This audit separates:

- what metadata the system already has;
- what metadata is missing;
- what logic is currently too weak;
- what should be changed in the real pipeline.

## Current Active Data

Active collection:

```text
vbpl_aiteam_meta_parsefix_20260628
```

Active BM25:

```text
data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl
```

Payload keys observed in Qdrant:

```text
char_len
co_quan_ban_hanh
dieu_so
dieu_tieu_de
doc_id
khoan_so
linh_vuc
loai_van_ban
metadata_enriched_issue_date
nam
ngay_ban_hanh
ngay_hieu_luc
nguon
so_ky_hieu
source_url
text
tinh_trang_hieu_luc
title
```

## Metadata Coverage Audit

Full scroll over `285002` points:

| Field | Empty points |
|---|---:|
| `ngay_hieu_luc` | 285002 |
| `tinh_trang_hieu_luc` | 285002 |
| `dieu_tieu_de` | 12169 |
| `khoan_so` | 178654 |
| `source_url` | 447 |

Important finding:

`ngay_ban_hanh` is present, but `ngay_hieu_luc`, `ngay_het_hieu_luc`, and reliable `tinh_trang_hieu_luc` are absent. Therefore the system currently cannot know the current legal state from Qdrant alone.

## Cutoff Risk

If the benchmark ground truth is updated only to March 2026, documents after `2026-03-31` should not receive recency boost.

Current collection contains chunks after that cutoff:

| Document | Issue date | Title |
|---|---|---|
| `132/2026` | 2026-04-06 | Sửa đổi, bổ sung một số điều của Nghị định số 41/2018/NĐ-CP... |
| `141/2026` | 2026-04-29 | Sửa đổi, bổ sung một số điều của Nghị định số 68/2026/NĐ-CP... |

Current v72 metadata logic boosts by year only. That is too coarse. It can treat after-cutoff 2026 documents as preferred even when BTC should not.

## Metadata Graph Artifact

Created:

```text
scripts/build_doc_metadata_graph.py
data/doc_metadata_graph_20260630.json
```

The graph is built from Qdrant and contains:

- document prefix;
- title;
- legal type;
- domain;
- issue date;
- article count;
- `after_eval_cutoff`;
- `amendment_kind`: `base`, `amendment`, or `consolidated`;
- document numbers referenced in title;
- weak successor edges inferred from title references.

Build result:

| Metric | Value |
|---|---:|
| unique document prefixes | 9073 |
| amendment-like docs | 1304 |
| consolidated docs | 30 |
| successor sources inferred from title refs | 63 |
| docs after `2026-03-31` | 2 |

Caveat:

This is not an authoritative legal-effect graph. Title references miss major replacements such as `60/2005 -> 59/2020` or `68/2014 -> 59/2020`. It should be used as ranking evidence only.

## Root Cause: Metadata

The system has enough metadata for basic tie-breaking:

- issue date;
- legal type;
- domain;
- article number/title;
- source;
- document title.

But it lacks the metadata needed for high precision:

1. `effective_from`
2. `effective_to`
3. `status_as_of_cutoff`
4. `replaced_by`
5. `replaces`
6. `amended_by`
7. `amends`
8. `consolidates`
9. `is_current_as_of_2026_03_31`
10. `target_articles_modified` for amendment documents
11. `article_role`: definition, scope, procedure, sanction, right, duty, remedy, transition, enforcement
12. `article_scope`: general rule vs direct operative rule

Without these, the selector has to guess from title and text, which explains unstable old/new document selection.

## Root Cause: Logic

Current v72 logic is mostly deterministic and does not apply LLM selector. The problem is not mainly prompt.

Important current logic limitations:

1. Recency uses year-level boost, not benchmark cutoff.
   - A 2026 document after March can be boosted.

2. Version knowledge is hardcoded and partial.
   - `OBSOLETE_PREFIXES`, `CURRENT_BOOSTS`, and `SUCCESSORS` help a few cases but are incomplete.
   - These maps should become data artifacts, not scattered constants.

3. Amendment documents are not handled structurally.
   - An amendment article may be the correct citation, but often it only modifies another legal base.
   - The system lacks `target_doc` and `target_article` metadata.

4. Selector under-selects article-level complements.
   - Audit against v50 showed `937/1490` v50-extra articles were already in the v72 retrieval pool.
   - This means retrieval often finds the candidate but selector does not choose it.

5. Same-doc expansion improves article recall but loses precision.
   - v79d kept docset identical and raised article recall, proving docs are not the only problem.
   - But precision dropped, proving article selection within a correct document is still weak.

6. Multi-hop queries need clause state, not just top-k thresholding.
   - For questions with multiple legal asks, the selector must track which clause is already covered.

## What Should Be Added Next

### Required Metadata

Priority 1:

- `status_as_of_2026_03_31`
- `effective_from`
- `effective_to`
- `replaced_by`
- `replaces`
- `amended_by`
- `amends`
- `is_after_eval_cutoff`

Priority 2:

- `article_role`
- `article_is_generic`
- `article_is_direct_operational`
- `article_amends_target_doc`
- `article_amends_target_article`

Priority 3:

- `domain_taxonomy_v2`
- `document_family`
- `legal_hierarchy_rank`
- `consolidated_source_docs`

### Required Logic Changes

1. Replace year-only recency boost with cutoff-aware status:

```text
if issue_date > 2026-03-31:
    penalize unless question explicitly references that date/document
```

2. Replace hardcoded `CURRENT_BOOSTS/SUCCESSORS` with a data-loaded version graph:

```text
data/legal_effect_graph_*.json
```

3. Add article role gating:

```text
definition/scope/general principle articles are not selected unless the question asks definition/scope/principle.
```

4. Add clause coverage state:

```text
split question into legal clauses -> each selected article must cover an uncovered clause or be dropped.
```

5. Treat same-doc expansion as a selector feature, not postprocess:

```text
same doc additions are allowed only when heading/snippet matches an uncovered clause.
```

6. Add amendment-aware retrieval:

```text
if candidate is an amendment document, expose both:
- cited amendment article
- target document/article being amended
```

7. Use LLM only for diagnostics:

```text
LLM may label uncovered clauses and explain candidate fit.
LLM should not directly mutate final articles.
```

## Current Conclusion

The current ceiling is mainly caused by missing legal-effect metadata and selector architecture, not by the embedding model alone.

The next real-system improvement should be:

1. Build or import legal-effect graph.
2. Add cutoff-aware graph scoring to retrieval/rerank.
3. Add clause-level selector with article-role gating.
4. Only then run a full pipeline and submit.

## Selector Fix Applied

Patched `scripts/build_v66_arch_pipeline.py` after smoke-auditing cases where the correct candidate was retrieved but not selected.

Changes:

1. Added low-raw guard in `acceptable()`:
   - candidates with near-zero reranker score can no longer be selected only because they receive current-version metadata boost;
   - exception is allowed only when the article heading/title has direct lexical evidence for the question.

2. Added domain guards:
   - block company-governance articles such as `Ban kiểm soát`, `Hội đồng quản trị`, `thành viên hợp danh`, `phần vốn góp` unless the question is actually about company governance;
   - block `trọng tài lao động` when the question asks ordinary commercial arbitration and does not mention labor;
   - block aviation arbitration/articles unless the question is about aviation.

3. Fixed overly broad pre-2015 penalty:
   - old-but-current laws are no longer treated as obsolete only because their year is before 2015.
   - added `STILL_CURRENT_OLD_PREFIXES` for important still-current laws:
     - `36/2005` Luật Thương mại
     - `50/2005` Luật Sở hữu trí tuệ
     - `54/2010` Luật Trọng tài thương mại
     - `54/2014` Luật Hải quan
     - `91/2015` Bộ luật Dân sự
     - `88/2015` Luật Kế toán

4. Added a few clearly obsolete customs prefixes to `OBSOLETE_PREFIXES`:
   - `101/2001`
   - `102/2001`
   - `154/2005`

Smoke results after patch:

- ID `1008`: no longer selects `59/2020 Luật Doanh nghiệp Điều 170`; selects promotion/advertising basis.
- ID `1035`: prefers `54/2014 Luật Hải quan Điều 76` and suppresses old customs decrees.
- ID `1058`: selects `54/2010 Luật Trọng tài thương mại Điều 44` instead of labor arbitration or aviation arbitration.
- Regression cases `1001`, `1016`, `1053`, `1055`, `1251`, `1795`, `1860`, `1999` still produce the expected multi-hop/core legal bases.
