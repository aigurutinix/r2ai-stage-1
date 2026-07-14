# V66 Architecture Pipeline Notes

Date: 2026-06-27

## Goal

Build a real retrieval-selection pipeline for the second 1000 questions (IDs 1001-2000), not an answer-hacking postprocess:

- retrieve from original question plus safe subqueries for complex questions;
- rerank with current-version awareness;
- select by clause coverage and direct legal relevance;
- keep prompt/LLM verification as a diagnostic layer, not as an unsafe final selector.

## What Changed

- Fixed `head_norm()` so headings like `Äiá»u 44. NguyÃªn táº¯c...` are normalized correctly.
- Narrowed generic filtering:
  - do not reject every `NguyÃªn táº¯c...` article;
  - reject broad penalty-principle articles when a direct penalty article exists.
- Fixed `off_topic()` to use document title and article heading instead of incidental words inside article body.
  - Example: `38/2019/QH14 Äiá»u 67` mentions `tá»• chá»©c tÃ­n dá»¥ng` in its body but is still a tax article and must not be dropped for tax dissolution questions.
- Added topic complements for multi-hop coverage:
  - dissolution and remaining tax debt;
  - delayed tax refund handling;
  - tax extension há»“ sÆ¡/trÃ¬nh tá»±;
  - representative authorization and business registration;
  - SME startup support selection.
- Added redundant filtering:
  - if `125/2020/NÄ-CP Äiá»u 16` is present for tax under-declaration penalties, drop broad/older tax penalty framework articles such as `38/2019 Äiá»u 142/138/136`.
- Added `--llm-verify` prompt audit:
  - Qwen reviews keep/add/drop/missing over retrieved candidates;
  - default is diagnostic only and writes to sidecar;
  - applying Qwen changes requires explicit `--llm-apply`.

## Prompt Layer Decision

Qwen with thinking is useful for explanation, but current tests show it is not reliable enough to mutate final retrieval:

- ID 1053: Qwen selected `19/2021/TT-BTC Äiá»u 31` about electronic tax transactions instead of direct `80/2021/TT-BTC Äiá»u 24` on tax extension há»“ sÆ¡/trÃ¬nh tá»±.
- ID 1999: Qwen still preferred a broad/new tax principle article over the direct penalty decree in one review.

Therefore v66 full run should not use `--llm-apply`. Use deterministic output for submission and keep `--llm-verify` only for audit.

## Smoke Audit

Command:

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; python scripts/build_v66_arch_pipeline.py --ids 1001,1016,1053,1055,1251,1795,1860,1999 --out data/_v66_smoke_audit6.json --sidecar data/_v66_smoke_audit6_sidecar.json --zip
```

Observed final articles:

- 1001: `99/2025 Äiá»u 22`, `88/2015 Äiá»u 47`, `88/2015 Äiá»u 37`.
- 1016: `108/2025 Äiá»u 42`, `108/2025 Äiá»u 18`.
- 1053: `38/2019 Äiá»u 67`, `38/2019 Äiá»u 75`, `80/2021 Äiá»u 27`, `80/2021 Äiá»u 24`.
- 1055: `80/2021 Äiá»u 24`, `126/2020 Äiá»u 19`, `38/2019 Äiá»u 64`.
- 1251: `45/2019 Äiá»u 27`.
- 1795: `168/2025 Äiá»u 54`, `59/2020 Äiá»u 14`.
- 1860: `04/2017 Äiá»u 17`, `06/2022 Äiá»u 14`, `80/2021 Äiá»u 22`.
- 1999: `125/2020 Äiá»u 16`, `12/2022 Äiá»u 38`, `12/2022 Äiá»u 4`.

## Full Run Recommendation

Run deterministic v66 for IDs 1001-2000:

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONPATH='.'; $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; python scripts/build_v66_arch_pipeline.py --lo 1001 --hi 2000 --out data/submission_v66_arch_pipeline.json --sidecar data/v66_arch_pipeline_sidecar.json --checkpoint-every 25 --zip
```
