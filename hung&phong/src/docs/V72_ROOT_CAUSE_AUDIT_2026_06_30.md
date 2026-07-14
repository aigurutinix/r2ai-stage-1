# V72 Root-Cause Audit - 2026-06-30

## Goal

Find why v72 is stuck around:

- `ARTICLES_F2MACRO = 0.6319`
- `ARTICLES_PRECISION = 0.5193`
- `ARTICLES_RECALL = 0.7303`
- `DOCS_F2MACRO = 0.6753`
- `DOCS_PRECISION = 0.6167`
- `DOCS_RECALL = 0.7267`

Target is not small postprocess gains; target is a real pipeline path toward `>0.7`.

## Local Score History

Best local submissions in `Downloads/scoring_result*.zip`:

| Version | ART_F2 | ART_P | ART_R | DOC_F2 | DOC_P | DOC_R |
|---|---:|---:|---:|---:|---:|---:|
| v50 | 0.6567 | 0.5627 | 0.7270 | 0.6709 | 0.5850 | 0.7200 |
| v36/v49 | 0.6530 | 0.5607 | 0.7230 | 0.6834 | 0.6050 | 0.7300 |
| v51 | 0.6499 | 0.5207 | 0.7497 | 0.6798 | 0.5757 | 0.7500 |
| v72 | 0.6319 | 0.5193 | 0.7303 | 0.6753 | 0.6167 | 0.7267 |

No local scoring artifact currently shows `>0.7` for article/doc F2.

## Pipeline Difference

v72:

- collection: `vbpl_aiteam_meta_parsefix_20260628`
- embedding: `AITeamVN/Vietnamese_Embedding_v2`
- reranker: `AITeamVN/Vietnamese_Reranker`
- BM25: `data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl`
- no LLM selector by default
- deterministic selector from `build_v66_arch_pipeline.py`
- metadata date boost is only a tie-breaker

v50:

- old collection: `vbpl_aiteam`
- retrieval + subquery + adaptive
- optional judge
- penalty whitelist
- concept-additive expansion
- version collapse
- v47 domain/version filter
- v49 known-old blacklist

The practical difference is not the prompt. v72 mostly avoids prompt mutation. v50 is broader and adds more complementary candidates.

## Count Comparison

For IDs 1001-2000:

| Version | avg articles | avg docs |
|---|---:|---:|
| v49 | 3.412 | 2.515 |
| v50 | 3.124 | 2.399 |
| v72 | 2.565 | 1.913 |
| v76b | 2.924 | 2.073 |

v72 is much more conservative. That explains better DOC precision, but it also drops article recall opportunities.

## Selector vs Retrieval Diagnosis

Compared v50 extras against v72 sidecar pools:

- v50 extra articles over v72: `1490`
- those already present in v72 pool: `937`
- ratio in pool: `62.9%`
- same-doc in pool: `231`
- new-doc in pool: `706`

Interpretation:

The main bottleneck is selector/decision logic, not corpus/index coverage. Retrieval already surfaces many candidates that v72 does not select.

## Observed Failure Modes

1. Under-selection of complementary articles
   - Example: ID 1005 retrieves `BLDS Äiá»u 536` with heading `nghÄ©a vá»¥ cá»§a bÃªn thuÃª váº­n chuyá»ƒn`, but v72 does not select it.
   - Cause: threshold/coverage logic is too strict for secondary-but-direct provisions.

2. Over-penalty for older but still direct instruments
   - Example: ID 1034 retrieves `11/2015/TT-BKHCN Äiá»u 13/14` with very high raw scores for trademark/trade-name infringement, but v72 prefers newer `65/2023` candidates.
   - Metadata/currentness helps, but it can suppress direct legal bases.

3. Document-level expansion is risky
   - v76b added many new docs and increased `ARTICLES_RECALL`, but `DOCS_PRECISION` dropped.
   - Therefore broad expansion cannot be the solution.

4. Prompt/LLM selector is not reliable as the final mutator
   - v75 prompt selector cut too much and dropped recall.
   - Qwen 9B can help diagnose missing coverage, but should not own final keep/drop.

## Root Cause

Current ceiling is mostly architecture/selector:

- data/index is sufficient for many misses because candidates are already in pool;
- retrieval still has misses, but it is not the first bottleneck;
- prompt is not the main issue in v72 because v72 does not apply LLM selection;
- algorithm is too conservative for article-level coverage and too blunt about old/new document priors.

## Next Architecture Direction

Build an evidence-balanced selector:

1. Keep v72 deterministic core for precision.
2. Add secondary candidates only when they satisfy evidence gates:
   - candidate is already in top retrieval pool;
   - heading/snippet directly matches a legal cue in the question;
   - candidate covers a clause not covered by selected articles;
   - prefer same-doc additions because they do not hurt DOC precision;
   - allow new-doc additions only at higher confidence.
3. Do not let LLM directly choose final articles.
4. Use LLM only as an audit signal: identify uncovered question clauses and explain whether pool contains a candidate.
5. Reduce blanket old-document penalties where raw reranker score is very high and heading is directly relevant.
