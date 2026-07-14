"""v47 — lọc nhiễu trên v46 (precision tune phần thêm, KHÔNG re-run). Recall-safe.

v46 thêm ~1.56 điều/câu (R↑, P↓ nhẹ). Bỏ nhiễu rõ trong phần thêm: sai-domain (đối chiếu câu hỏi)
+ bản-cũ (cùng loại VB, cùng chủ đề, có bản mới trong đáp án). Recall-safe: chỉ bỏ khi an toàn.

Chạy: PYTHONUTF8=1 PYTHONPATH=. python scripts/build_v47_filter.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_v39_domain import is_offdomain, rebuild_docs  # noqa: E402
from build_v40_version import topic_tokens, year_of, is_amend, instrument, jacc  # noqa: E402


def version_drop(arts: list[str]) -> list[str]:
    """Bỏ bản cũ recall-safe (cùng LOẠI VB, cùng chủ đề, có bản mới hơn trong CÙNG đáp án)."""
    info = []
    for a in arts:
        p = a.split("|"); nm = p[1].strip() if len(p) >= 2 else ""
        info.append({"art": a, "yr": year_of(p[0].strip()), "tok": topic_tokens(nm),
                     "amend": is_amend(nm), "inst": instrument(nm), "sk": p[0].strip()})
    drop = set()
    for i, x in enumerate(info):
        if x["amend"] or x["yr"] == 0 or x["inst"] == "other":
            continue
        for j, y in enumerate(info):
            if i == j or y["amend"] or y["sk"] == x["sk"]:
                continue
            if y["inst"] == x["inst"] and y["yr"] > x["yr"] and jacc(x["tok"], y["tok"]) >= 0.75:
                drop.add(i); break
    return [info[i]["art"] for i in range(len(info)) if i not in drop]


def main() -> None:
    src = json.loads((ROOT / "data/submission_v46.json").read_text(encoding="utf-8"))
    out = []; n_dom = n_ver = 0
    for r in src:
        q = r["question"]; arts = r["relevant_articles"]
        # 1) domain filter (recall-safe: chỉ bỏ off-domain khi còn >1 điều)
        kept = []; dropped = 0
        for a in arts:
            if is_offdomain(a, q) and (len(arts) - dropped) > 1:
                dropped += 1; n_dom += 1; continue
            kept.append(a)
        if not kept:
            kept = arts[:1]
        # 2) version filter
        before = len(kept); kept = version_drop(kept); n_ver += before - len(kept)
        if not kept:
            kept = arts[:1]
        nr = dict(r); nr["relevant_articles"] = kept; nr["relevant_docs"] = rebuild_docs(kept)
        out.append(nr)
    json.dump(out, open(ROOT / "data/submission_v47.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    import statistics
    print(f"v47: bỏ {n_dom} off-domain + {n_ver} bản-cũ | TB v46→v47 "
          f"{statistics.mean(len(r['relevant_articles']) for r in src):.2f} → "
          f"{statistics.mean(len(r['relevant_articles']) for r in out):.2f} điều/câu")


if __name__ == "__main__":
    main()
