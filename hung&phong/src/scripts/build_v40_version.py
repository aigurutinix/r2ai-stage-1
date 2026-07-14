"""BƯỚC 2 — Version filter mở rộng (post-process v39, FREE). Diệt noise_ban_cu (~26%).

Bắt cặp bản-cũ-vs-mới CÙNG chủ đề mà version-map (134 họ) bỏ sót (khác tiêu đề). RECALL-SAFE:
chỉ bỏ bản cũ KHI trong CÙNG đáp án có bản mới cùng chủ đề (Jaccard topic >= 0.75, năm mới hơn).
KHÔNG bỏ bản cũ đứng một mình. Loại trừ "Luật sửa đổi/bổ sung" (đồng tồn bản gốc).

Chạy: PYTHONUTF8=1 PYTHONPATH=. python scripts/build_v40_version.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

_STOP = {"luat", "bo", "nghi", "dinh", "thong", "tu", "quyet", "mot", "so", "dieu", "cua",
         "ve", "quy", "va", "cac", "doi", "voi", "trong", "linh", "vuc", "huong", "dan",
         "chi", "tiet", "thi", "hanh", "sua", "bo", "sung", "ban", "hanh"}


def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def topic_tokens(name: str) -> set:
    toks = re.findall(r"[a-z0-9]+", _norm(name))
    return {t for t in toks if t not in _STOP and len(t) > 2 and not t.isdigit()}


def year_of(sk: str) -> int:
    m = re.search(r"/(\d{4})", sk) or re.search(r"\b(19|20)(\d{2})\b", sk)
    if not m:
        return 0
    return int(m.group(1)) if len(m.group(1)) == 4 else int(m.group(0))


def is_amend(name: str) -> bool:
    n = _norm(name)
    return "sua doi" in n or "bo sung" in n


def instrument(name: str) -> str:
    """Loại văn bản — version chỉ tính trong CÙNG loại (Luật↔Luật...).
    Luật vs Nghị định cùng chủ đề = CO-CITATION (cả 2 gold), KHÔNG drop."""
    n = _norm(name)
    if n.startswith("bo luat") or n.startswith("luat"):
        return "luat"
    if n.startswith("nghi dinh"):
        return "nd"
    if n.startswith("thong tu"):  # gồm thông tư liên tịch
        return "tt"
    if n.startswith("quyet dinh"):
        return "qd"
    if n.startswith("phap lenh"):
        return "pl"
    return "other"


def jacc(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rebuild_docs(arts):
    docs, seen = [], set()
    for a in arts:
        p = a.split("|"); d = f"{p[0].strip()}|{p[1].strip()}" if len(p) >= 2 else p[0].strip()
        if d not in seen:
            seen.add(d); docs.append(d)
    return docs


def main() -> None:
    src = json.loads((ROOT / "data/submission_v39_domain.json").read_text(encoding="utf-8"))
    out = []; n_drop = 0; examples = []
    for r in src:
        arts = r["relevant_articles"]
        # gom thông tin doc của từng điều
        info = []
        for a in arts:
            p = a.split("|")
            sk = p[0].strip(); nm = p[1].strip() if len(p) >= 2 else ""
            info.append({"art": a, "sk": sk, "name": nm, "yr": year_of(sk),
                         "tok": topic_tokens(nm), "amend": is_amend(nm), "inst": instrument(nm)})
        drop_idx = set()
        for i, x in enumerate(info):
            if x["amend"] or x["yr"] == 0 or x["inst"] == "other":
                continue
            for j, y in enumerate(info):
                if i == j or y["amend"] or y["sk"] == x["sk"]:
                    continue
                # y mới hơn x, CÙNG LOẠI văn bản, cùng chủ đề → bỏ x (bản cũ thật)
                if y["inst"] == x["inst"] and y["yr"] > x["yr"] and jacc(x["tok"], y["tok"]) >= 0.75:
                    drop_idx.add(i); break
        kept = [info[i]["art"] for i in range(len(info)) if i not in drop_idx]
        if not kept and arts:
            kept = [arts[0]]
        dropped = [info[i] for i in drop_idx]
        n_drop += len(dropped)
        if dropped and len(examples) < 12:
            examples.append((r["question"][:65], [(d["name"], d["yr"]) for d in dropped]))
        nr = dict(r); nr["relevant_articles"] = kept; nr["relevant_docs"] = rebuild_docs(kept)
        out.append(nr)
    json.dump(out, open(ROOT / "data/submission_v40_version.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    import statistics
    print(f"Bỏ {n_drop} bản cũ | TB v39→v40: {statistics.mean(len(r['relevant_articles']) for r in src):.3f} → "
          f"{statistics.mean(len(r['relevant_articles']) for r in out):.3f} điều/câu | "
          f"câu bị động: {sum(1 for s,o in zip(src,out) if len(s['relevant_articles'])!=len(o['relevant_articles']))}")
    print("\n=== VÍ DỤ bản cũ bị bỏ (có bản mới cùng chủ đề trong đáp án) ===")
    for q, drs in examples:
        print(f"Q: {q}")
        for nm, yr in drs:
            print(f"   ✗ bỏ ({yr}): {nm[:75]}")


if __name__ == "__main__":
    main()
