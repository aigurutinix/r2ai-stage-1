"""Distill Claude→reranker: dựng train data từ lựa chọn Claude trên 2000 câu BTC.

Claude-chọn (trong pool v37 top-20) = POSITIVE thật; Claude-bỏ = HARD-NEGATIVE thật
(điều reranker xếp cao mà Claude phán sai). Nhãn relevance THẬT trên CÂU THẬT của BTC
→ hơn hẳn synthetic (chunk=positive). Format cho finetune_reranker.py: {query, pos:[t], neg:[...]}.

Chạy: PYTHONUTF8=1 python scripts/build_distill_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cands = {r["id"]: r for r in json.loads((ROOT / "data/_claude_cands.json").read_text(encoding="utf-8"))}
    sel = {r["id"]: r["selected"] for r in json.loads((ROOT / "data/_claude_sel.json").read_text(encoding="utf-8"))}

    rows = []
    n_q = n_pos = 0
    for qid, selected in sel.items():
        rec = cands.get(qid)
        if not rec:
            continue
        clist = rec["cands"]; q = rec["question"]
        sel_set = {i for i in selected if isinstance(i, int) and 0 <= i < len(clist)}
        if not sel_set:
            continue
        pos = [clist[i]["text"] for i in sel_set if clist[i]["text"]]
        neg = [clist[i]["text"] for i in range(len(clist)) if i not in sel_set and clist[i]["text"]]
        if not pos or len(neg) < 5:
            continue
        n_q += 1
        for pt in pos:                       # 1 row / positive (multi-pos → nhiều row)
            rows.append({"query": q, "pos": [pt], "neg": neg})
            n_pos += 1
    out = ROOT / "data/_distill_train.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import statistics
    print(f"Distill data: {len(rows)} row (từ {n_q} câu, {n_pos} positive) → {out}")
    print(f"  TB neg/row: {statistics.mean(len(r['neg']) for r in rows):.1f} | TB pos/câu: {n_pos/max(n_q,1):.2f}")


if __name__ == "__main__":
    main()
