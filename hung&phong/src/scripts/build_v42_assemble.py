"""Ráp lựa chọn của Claude-selector (id→indices) → submission_v42. Chạy sau workflow.

data/_claude_sel.json = [{id, selected:[idx]}] (lưu từ kết quả workflow).
"""
from __future__ import annotations

import json
import sys
import statistics
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]


def rebuild_docs(arts):
    docs, seen = [], set()
    for a in arts:
        p = a.split("|"); d = f"{p[0].strip()}|{p[1].strip()}" if len(p) >= 2 else p[0].strip()
        if d not in seen:
            seen.add(d); docs.append(d)
    return docs


def main() -> None:
    cands = {r["id"]: r["cands"] for r in json.loads((ROOT / "data/_claude_cands.json").read_text(encoding="utf-8"))}
    sel = {r["id"]: r["selected"] for r in json.loads((ROOT / "data/_claude_sel.json").read_text(encoding="utf-8"))}
    ans = {r["id"]: r.get("answer", "") for r in json.loads((ROOT / "data/submission_v20_clean.json").read_text(encoding="utf-8"))}
    qs = json.loads(Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json").read_text(encoding="utf-8"))

    out = []; n_empty = 0
    for q in qs:
        qid = q["id"]; clist = cands.get(qid, [])
        idxs = sel.get(qid, [])
        arts = []
        for i in idxs:
            if isinstance(i, int) and 0 <= i < len(clist):
                arts.append(clist[i]["art"])
        # dedup giữ thứ tự
        seen = set(); arts = [a for a in arts if not (a in seen or seen.add(a))]
        if not arts and clist:    # fallback top-1
            arts = [clist[0]["art"]]; n_empty += 1
        out.append({"id": qid, "question": q["question"], "answer": ans.get(qid, ""),
                    "relevant_docs": rebuild_docs(arts), "relevant_articles": arts})
    json.dump(out, open(ROOT / "data/submission_v42_claude.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"v42 (Claude-selector): {len(out)} câu | TB {statistics.mean(len(r['relevant_articles']) for r in out):.3f} điều/câu "
          f"| fallback rỗng: {n_empty} | thiếu selection: {sum(1 for q in qs if q['id'] not in sel)}")


if __name__ == "__main__":
    main()
