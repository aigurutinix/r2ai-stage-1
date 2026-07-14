# V67 Root Cause Audit

Date: 2026-06-28

## Why v66 Lost Score

v66 improved a few audited multi-hop cases, but broad 1000-question scoring dropped because the selector became overconfident when retrieval was weak.

Main evidence from `data/v66_audit_candidates.csv`:

- `895/1000` second-half rows differ from v50.
- `104` rows collapsed to one article.
- `40` rows have all selected articles with `max(score_raw) < 0.1`.
- `31` selected articles have `score_raw < 0.05`.
- `62` rows dropped at least 3 articles compared with v50.

Manual audit examples:

- ID 1175: labor contract/táº¡m hoÃ£n/cháº¥m dá»©t question, v66 chose `168/2025/NÄ-CP Äiá»u 28` about social enterprise registration. This is wrong-domain and has raw score `0.0003`.
- ID 1008: promotion/advertising question, v66 chose `59/2020/QH14 Äiá»u 170` about board of supervisors. Wrong-domain, raw `0.0001`.
- ID 1058: arbitration question, v66 chose `45/2019/QH14 Äiá»u 193` about labor arbitration, while `54/2010/QH12 Äiá»u 44` was in the pool with raw `0.6626`.
- ID 1997: customs/IP control question, v66 chose `59/2020/QH14 Äiá»u 105` about board of supervisors, while the correct-looking customs/IP articles were in the pool.

## Root Causes

1. Current-version boost was allowed to dominate raw retrieval score.
2. Global year penalty damaged valid older-but-current laws such as Commercial Law 2005, IP Law 2005, Arbitration Law 2010.
3. Fallback behavior was unsafe: when confidence was low, v66 still selected `primary[0]` instead of falling back to the baseline.
4. Multi-hop expansion helped some cases but created wrong-domain noise when subquery terms were ambiguous.

## V67 Strategy

Use v66 only when confidence is acceptable. Fall back to v50 when v66 shows high-risk symptoms:

- selected articles all have very low raw retrieval score;
- newly selected article has near-zero raw score;
- v66 collapses a multi-article v50 answer to one low-confidence article;
- v66 drops trusted older-but-current law families;
- v66 selects enterprise/governance articles with near-zero raw score due to boost.

This is not a score hack; it is a real uncertainty policy: if the new selector is not confident, keep the safer baseline.

## Generated Variants

- `data/submission_v67_confidence_fallback.zip`
  - fallback: 52 second-half rows
  - conservative rescue of the most obvious v66 failures.

- `data/submission_v67_strict_fallback.zip`
  - fallback: 59 second-half rows
  - recommended first submit: rescues low-raw/drop/trusted-old-law cases while retaining most v66 multi-hop improvements.

- `data/submission_v67_very_strict_fallback.zip`
  - fallback: 86 second-half rows
  - safer precision, but may remove useful v66 expansions.

All variants validated:

- 2000 rows;
- IDs 1-2000 in order;
- no empty `relevant_articles` or `relevant_docs`;
- `relevant_docs` matches rebuilt docs from `relevant_articles`;
- zip contains only `results.json`.
