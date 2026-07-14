"""v43 — reranker DISTILL (học từ Claude) re-score pool v37 top-20 → keep → submission OPEN hợp lệ.

Mục tiêu: kéo selection open lên gần Claude (v42=0.653) bằng model <14B hợp lệ.
Re-score (query, text) bằng reranker_distill → keep ngưỡng (calib ~2.6 điều/câu) + sàn top-1.

Chạy: RERANKER_MODEL=models/reranker_distill ... PYTHONUTF8=1 PYTHONPATH=. python scripts/build_v43_distill.py
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")

import json
import sys
import statistics
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from exp_v24 import collapse_versions  # noqa: E402


def rebuild_docs(arts):
    docs, seen = [], set()
    for a in arts:
        p = a.split("|"); d = f"{p[0].strip()}|{p[1].strip()}" if len(p) >= 2 else p[0].strip()
        if d not in seen:
            seen.add(d); docs.append(d)
    return docs


def main() -> None:
    cands = json.loads((ROOT / "data/_claude_cands.json").read_text(encoding="utf-8"))
    ans = {r["id"]: r.get("answer", "") for r in json.loads((ROOT / "data/submission_v20_clean.json").read_text(encoding="utf-8"))}

    from FlagEmbedding import FlagReranker
    rr = FlagReranker(os.environ.get("RERANKER_MODEL", "models/reranker_distill"), use_fp16=True)

    # re-score tất cả
    scored = []
    for rec in cands:
        cl = rec["cands"]
        if not cl:
            scored.append((rec, [])); continue
        s = rr.compute_score([[rec["question"], c["text"]] for c in cl], normalize=True)
        if not isinstance(s, list):
            s = [s]
        scored.append((rec, s))

    def build(thresh, floor, name):
        out = []
        for rec, s in scored:
            cl = rec["cands"]
            order = sorted(range(len(cl)), key=lambda i: s[i], reverse=True)
            keep = [cl[order[0]]["art"]] if order else []           # sàn top-1
            for i in order[1:]:
                if len(keep) >= 5:
                    break
                if s[i] >= thresh:
                    keep.append(cl[i]["art"])
            arts = collapse_versions(keep)
            out.append({"id": rec["id"], "question": rec["question"], "answer": ans.get(rec["id"], ""),
                        "relevant_docs": rebuild_docs(arts), "relevant_articles": arts})
        json.dump(out, open(ROOT / f"data/submission_v43_{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"{name}: thresh {thresh} | TB {statistics.mean(len(r['relevant_articles']) for r in out):.3f} điều/câu", flush=True)

    for thr, nm in [(0.3, "t30"), (0.5, "t50"), (0.7, "t70")]:
        build(thr, 1, nm)


if __name__ == "__main__":
    main()
