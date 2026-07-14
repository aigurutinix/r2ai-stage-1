"""Internal F2 evaluation for the retrieval task (uses data/gold_dev.json).

Matches a predicted article to gold on (mã văn bản, số Điều) — names ignored,
mirroring the competition's "Điều X" + law_id join. Reports macro Precision,
Recall, F2 (F2 = 5PR / (4P+R), recall-weighted).

Run:  python eval_f2.py            # scores ../results.json
      python eval_f2.py path.json
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
import local_models_config as cfg

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _key(code: str, article: str):
    code = re.sub(r"\s+", "", code).upper()
    m = re.search(r"Điều\s*(\d+)", article) or re.search(r"(\d+)", article)
    return (code, m.group(1)) if m else None


def _pred_keys(rel_articles):
    out = set()
    for s in rel_articles:
        parts = s.split("|")
        if len(parts) >= 3:
            k = _key(parts[0], parts[-1])
            if k:
                out.add(k)
    return out


def _gold_keys(gold_articles):
    out = set()
    for s in gold_articles:
        parts = s.split("|")
        if len(parts) >= 2:
            k = _key(parts[0], parts[-1])
            if k:
                out.add(k)
    return out


def f2(p, r):
    return (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0.0


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "results.json")
    with open(cfg.GOLD_DEV, encoding="utf-8") as f:
        gold = {g["id"]: _gold_keys(g["gold_articles"]) for g in json.load(f)["items"]}
    with open(results_path, encoding="utf-8") as f:
        preds = {r["id"]: _pred_keys(r.get("relevant_articles", [])) for r in json.load(f)}

    Ps, Rs, F2s = [], [], []
    print(f"{'id':>3} {'P':>5} {'R':>5} {'F2':>5}  matched/pred/gold")
    for qid, g in sorted(gold.items()):
        p_set = preds.get(qid, set())
        correct = len(p_set & g)
        P = correct / len(p_set) if p_set else 0.0
        R = correct / len(g) if g else 0.0
        F = f2(P, R)
        Ps.append(P); Rs.append(R); F2s.append(F)
        print(f"{qid:>3} {P:5.2f} {R:5.2f} {F:5.2f}  {correct}/{len(p_set)}/{len(g)}")

    n = len(F2s) or 1
    print("\n" + "=" * 40)
    print(f"MACRO  Precision={sum(Ps)/n:.4f}  Recall={sum(Rs)/n:.4f}  F2={sum(F2s)/n:.4f}")
    print(f"(ALQAC 2024 best retrieval F2 ≈ 0.87 — reference)")
    print("=" * 40)


if __name__ == "__main__":
    main()
