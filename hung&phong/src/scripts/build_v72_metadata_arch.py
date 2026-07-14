"""v72: v66 deterministic architecture on parsefix data, with metadata-aware scoring.

This is meant to test the new Qdrant data branch cleanly:

- first 1000 rows are kept from `data/submission_v50_full.json`;
- ids 1001-2000 are rebuilt from retrieval on `vbpl_aiteam_meta_parsefix_20260628`;
- no concept expansion and no LLM judge by default, to avoid v71-style drift;
- use `ngay_ban_hanh` metadata as a small tie-breaker, not as a relevance replacement.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import date

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("QDRANT_COLLECTION", "vbpl_aiteam_meta_parsefix_20260628")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("EMBED_BACKEND", "st")
os.environ.setdefault("EMBED_ST_MODEL", "AITeamVN/Vietnamese_Embedding_v2")
os.environ.setdefault("HYBRID_SEARCH", "true")
os.environ.setdefault("USE_RERANKER", "true")
os.environ.setdefault("RERANKER_MODEL", "AITeamVN/Vietnamese_Reranker")
os.environ.setdefault("BM25_INDEX_PATH", "data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl")

import build_v66_arch_pipeline as v66  # noqa: E402
from tests.build_submission_v12 import cand_from_hit as _base_cand_from_hit  # noqa: E402


_V66_BASE_SCORE = v66.base_score
_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})")


def issue_year(payload: dict) -> int:
    raw = str(payload.get("ngay_ban_hanh") or payload.get("nam") or "")
    match = _DATE_RE.search(raw)
    if match:
        return int(match.group(1))
    match = re.search(r"(19|20)\d{2}", raw)
    return int(match.group(0)) if match else 0


def date_boost(year: int) -> float:
    """Small recency prior from enriched metadata.

    Keep this deliberately modest: retrieval/reranker relevance must dominate.
    The goal is to break ties between old/current legal bases, not to promote
    unrelated new documents.
    """
    if year >= 2026:
        return 0.045
    if year == 2025:
        return 0.040
    if year == 2024:
        return 0.030
    if year >= 2020:
        return 0.015
    if 0 < year < 2015:
        return -0.040
    return 0.0


def metadata_base_score(c: dict) -> float:
    score = _V66_BASE_SCORE(c)
    year = int(c.get("issue_year") or 0)
    raw = float(c.get("score_raw") or 0.0)

    # Metadata is only a tie-breaker. Do not rescue weak candidates just because
    # they are new.
    if raw >= 0.18:
        score += date_boost(year)
    elif year < 2015 and year:
        score -= 0.020
    return score


def metadata_candidate_from_hit(hit: dict, source: str, source_rank: int) -> dict | None:
    c = _base_cand_from_hit(hit)
    if not c:
        return None
    payload = hit.get("payload", {}) or {}
    c["text"] = str(payload.get("text") or "")
    c["title"] = str(payload.get("title") or "")
    c["prefix"] = v66.prefix(c["art"])
    c["source"] = source
    c["source_rank"] = source_rank
    c["score_raw"] = float(c.get("rr") or 0.0)
    c["ngay_ban_hanh"] = str(payload.get("ngay_ban_hanh") or "")
    c["issue_year"] = issue_year(payload)
    c["score_arch"] = metadata_base_score(c) + (0.04 if source == "orig" else 0.0)
    return c


def main() -> None:
    v66.base_score = metadata_base_score
    v66.candidate_from_hit = metadata_candidate_from_hit
    sys.argv[0] = "build_v72_metadata_arch.py"
    v66.main()


if __name__ == "__main__":
    main()
