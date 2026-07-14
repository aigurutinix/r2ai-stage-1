from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_v66_arch_pipeline import add_clause_evidence, doc_part  # noqa: E402


def _cand(key: str, art: str, text: str, raw: float = 0.5, arch: float = 0.5) -> dict:
    return {
        "_key": key,
        "art": art,
        "title": art.split("|")[1].strip() if "|" in art else art,
        "text": text,
        "prefix": art.split("/", 1)[0] + "/" + art.split("/", 2)[1].split("|", 1)[0] if "/" in art else art,
        "score_raw": raw,
        "score_arch": arch,
    }


def test_clause_evidence_adds_same_document_article() -> None:
    question = "Thu tuc hoan thue va thoi han nop ho so hoan thue duoc quy dinh nhu the nao?"
    selected = [
        _cand(
            "a1",
            "80/2021/TT-BTC | Thong tu quan ly thue | Dieu 28",
            "Dieu 28. Hoan thue. Nguoi nop thue duoc hoan thue theo quy dinh.",
        )
    ]
    same_doc = _cand(
        "a2",
        "80/2021/TT-BTC | Thong tu quan ly thue | Dieu 34",
        "Dieu 34. Thu tuc, ho so va thoi han giai quyet hoan thue.",
        raw=0.18,
        arch=0.2,
    )

    out = add_clause_evidence(question, selected, [selected[0], same_doc], max_k=4)

    assert same_doc["_key"] in {c["_key"] for c in out}


def test_clause_evidence_does_not_open_new_document() -> None:
    question = "Thu tuc hoan thue va thoi han nop ho so hoan thue duoc quy dinh nhu the nao?"
    selected = [
        _cand(
            "a1",
            "80/2021/TT-BTC | Thong tu quan ly thue | Dieu 28",
            "Dieu 28. Hoan thue. Nguoi nop thue duoc hoan thue theo quy dinh.",
        )
    ]
    new_doc = _cand(
        "b1",
        "126/2020/ND-CP | Nghi dinh quan ly thue | Dieu 40",
        "Dieu 40. Thu tuc, ho so va thoi han giai quyet hoan thue.",
        raw=0.9,
        arch=0.9,
    )

    out = add_clause_evidence(question, selected, [selected[0], new_doc], max_k=4)

    assert {doc_part(c["art"]) for c in out} == {doc_part(selected[0]["art"])}
