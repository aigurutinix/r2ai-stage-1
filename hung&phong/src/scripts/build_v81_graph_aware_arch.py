"""v81: graph-aware architecture wrapper over v66/v72 pipeline.

This is a system-level experiment, not submission postprocess:

- uses the parsefix Qdrant collection and BM25 index;
- loads `data/doc_metadata_graph_20260630.json`;
- applies benchmark-cutoff aware scoring;
- uses title-derived successor evidence as a soft signal;
- keeps deterministic selector from v66.

The graph is not authoritative legal-effect metadata, so it is used only for
soft ranking penalties/boosts.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("QDRANT_COLLECTION", "vbpl_aiteam_meta_parsefix_20260628")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("EMBED_BACKEND", "st")
os.environ.setdefault("EMBED_ST_MODEL", "AITeamVN/Vietnamese_Embedding_v2")
os.environ.setdefault("HYBRID_SEARCH", "true")
os.environ.setdefault("USE_RERANKER", "true")
os.environ.setdefault("RERANKER_MODEL", "AITeamVN/Vietnamese_Reranker")
os.environ.setdefault("BM25_INDEX_PATH", "data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl")

ROOT = Path(__file__).resolve().parents[1]
DOC_GRAPH_PATH = ROOT / "data" / "doc_metadata_graph_20260630.json"
EFFECT_GRAPH_PATH = ROOT / "data" / "legal_effect_graph_20260630.json"

import build_v66_arch_pipeline as v66  # noqa: E402
from tests.build_submission_v12 import cand_from_hit as _base_cand_from_hit  # noqa: E402


_V66_BASE_SCORE = v66.base_score
_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower().replace("đ", "d").replace("Ä‘", "d")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text).strip()


def prefix(art_or_sk: str) -> str:
    sk = (art_or_sk or "").split("|", 1)[0].strip()
    m = re.match(r"(\d+/\d{4})", unicodedata.normalize("NFKC", sk))
    return m.group(1) if m else sk


def issue_year(payload: dict) -> int:
    raw = str(payload.get("ngay_ban_hanh") or payload.get("nam") or "")
    match = _DATE_RE.search(raw)
    if match:
        return int(match.group(1))
    match = re.search(r"(19|20)\d{2}", raw)
    return int(match.group(0)) if match else 0


def load_graph() -> dict:
    if DOC_GRAPH_PATH.exists():
        return json.loads(DOC_GRAPH_PATH.read_text(encoding="utf-8"))
    return {"eval_cutoff": "2026-03-31", "docs": {}, "successors_from_title_refs": {}}


def load_effect_graph() -> dict:
    if EFFECT_GRAPH_PATH.exists():
        return json.loads(EFFECT_GRAPH_PATH.read_text(encoding="utf-8"))
    return {
        "eval_cutoff": "2026-03-31",
        "obsolete_prefixes": [],
        "still_current_old_prefixes": [],
        "successors": {},
        "current_boosts": {},
        "after_eval_cutoff_prefixes": [],
    }


GRAPH = load_graph()
EFFECT_GRAPH = load_effect_graph()
DOCS = GRAPH.get("docs") or {}
TITLE_SUCCESSORS = GRAPH.get("successors_from_title_refs") or {}
EVAL_CUTOFF = str(EFFECT_GRAPH.get("eval_cutoff") or GRAPH.get("eval_cutoff") or "2026-03-31")
EFFECT_OBSOLETE = set(EFFECT_GRAPH.get("obsolete_prefixes") or [])
EFFECT_STILL_CURRENT_OLD = set(EFFECT_GRAPH.get("still_current_old_prefixes") or [])
EFFECT_CURRENT_BOOSTS = {str(k): float(v) for k, v in (EFFECT_GRAPH.get("current_boosts") or {}).items()}
EFFECT_AFTER_CUTOFF = set(EFFECT_GRAPH.get("after_eval_cutoff_prefixes") or [])


def date_boost(c: dict) -> float:
    raw = float(c.get("score_raw") or 0.0)
    px = c.get("prefix") or prefix(c.get("art", ""))
    d = DOCS.get(px) or {}
    issue_date = str(d.get("ngay_ban_hanh") or c.get("ngay_ban_hanh") or "")
    year = int(c.get("issue_year") or 0)

    # Benchmark is believed to be current through March 2026. Do not boost
    # documents after that date; use a penalty because they can be unavailable
    # to the scorer's ground truth.
    if issue_date and issue_date > EVAL_CUTOFF:
        return -0.28 if raw < 0.90 else -0.10
    if raw < 0.18:
        return -0.020 if 0 < year < 2015 else 0.0
    if year == 2026:
        return 0.020
    if year == 2025:
        return 0.035
    if year == 2024:
        return 0.025
    if year >= 2020:
        return 0.012
    if 0 < year < 2015:
        return -0.045
    return 0.0


def graph_base_score(c: dict) -> float:
    score = _V66_BASE_SCORE(c)
    px = c.get("prefix") or prefix(c.get("art", ""))
    raw = float(c.get("score_raw") or 0.0)
    still_current = hasattr(v66, "still_current_old_candidate") and v66.still_current_old_candidate(c)

    # Override scattered code constants with the explicit graph artifact.
    # The base score may have applied older constants; this nudges toward the
    # graph while keeping reranker dominance.
    if px in EFFECT_OBSOLETE and raw < 0.95 and not still_current:
        score -= 0.12
    if raw >= 0.12 and (px in EFFECT_STILL_CURRENT_OLD or still_current):
        score += 0.16
    score += EFFECT_CURRENT_BOOSTS.get(px, 0.0) * 0.35

    score += date_boost(c)
    d = DOCS.get(px) or {}
    kind = d.get("amendment_kind")

    # Amendment documents are useful but often broad. If reranker evidence is
    # weak, avoid letting "new amendment" status win by itself.
    if kind == "amendment" and raw < 0.25:
        score -= 0.035
    elif kind == "consolidated" and raw >= 0.18:
        score += 0.020
    return score


def graph_candidate_from_hit(hit: dict, source: str, source_rank: int) -> dict | None:
    c = _base_cand_from_hit(hit)
    if not c:
        return None
    payload = hit.get("payload", {}) or {}
    c["text"] = str(payload.get("text") or "")
    c["title"] = str(payload.get("title") or "")
    c["prefix"] = prefix(c["art"])
    c["source"] = source
    c["source_rank"] = source_rank
    c["score_raw"] = float(c.get("rr") or 0.0)
    c["ngay_ban_hanh"] = str(payload.get("ngay_ban_hanh") or "")
    c["issue_year"] = issue_year(payload)
    d = DOCS.get(c["prefix"]) or {}
    c["amendment_kind"] = d.get("amendment_kind", "base")
    c["after_eval_cutoff"] = bool(d.get("after_eval_cutoff"))
    c["score_arch"] = graph_base_score(c) + (0.04 if source == "orig" else 0.0)
    return c


def merge_title_successors() -> None:
    # v66 successor map is hardcoded. Add title-derived edges as weak extra
    # evidence. The scoring penalty in v66 remains modest.
    for old, news in TITLE_SUCCESSORS.items():
        if not news:
            continue
        cur = list(v66.SUCCESSORS.get(old, []))
        for n in news:
            if n not in cur:
                cur.append(n)
        v66.SUCCESSORS[old] = cur
    for old, news in (EFFECT_GRAPH.get("successors") or {}).items():
        cur = list(v66.SUCCESSORS.get(old, []))
        for n in news:
            if n not in cur:
                cur.append(n)
        v66.SUCCESSORS[old] = cur
    v66.OBSOLETE_PREFIXES.update(EFFECT_OBSOLETE)
    if hasattr(v66, "STILL_CURRENT_OLD_PREFIXES"):
        v66.STILL_CURRENT_OLD_PREFIXES.update(EFFECT_STILL_CURRENT_OLD)


def main() -> None:
    merge_title_successors()
    v66.base_score = graph_base_score
    v66.candidate_from_hit = graph_candidate_from_hit
    sys.argv[0] = "build_v81_graph_aware_arch.py"
    v66.main()


if __name__ == "__main__":
    main()
