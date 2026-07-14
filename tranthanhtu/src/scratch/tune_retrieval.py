"""Tune retrieval cutoff (top_k / min_score / margin) on the 20-mock dev set.

Loads the engine ONCE, caches top-10 reranked candidates per question, then sweeps
threshold configs offline (fast) and reports configs by macro F2.
Run: python tune_retrieval.py
"""
import json, os, sys
from itertools import product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.insert(0, os.path.dirname(__file__))

import local_models_config as cfg
from local_rag_engine import LocalLegalRAGEngine
from eval_f2 import _key, f2

# Gold + questions
gold = {g["id"]: {_key(s.split("|")[0], s.split("|")[-1]) for s in g["gold_articles"]}
        for g in json.load(open(cfg.GOLD_DEV, encoding="utf-8"))["items"]}
questions = json.load(open(cfg.TEST_QUESTIONS, encoding="utf-8"))

# Cache top-10 reranked candidates per question (the expensive part, done once)
print("Caching candidates (load model + retrieve 20 q) ...")
eng = LocalLegalRAGEngine(use_reranker=True)
cache = {}
for q in questions:
    cache[int(q["id"])] = eng._candidates_scored(q["question"], pool=10)
print("cached.\n")


def score_config(top_k, min_score, margin):
    Ps, Rs, F2s = [], [], []
    for qid, cand in cache.items():
        g = gold.get(qid, set())
        sel = LocalLegalRAGEngine._cutoff(cand, top_k, min_score, margin)
        pred = {_key(d["doc_number"], d["article"]) for d in sel if d.get("doc_number") and d.get("article")}
        correct = len(pred & g)
        P = correct / len(pred) if pred else 0.0
        R = correct / len(g) if g else 0.0
        Ps.append(P); Rs.append(R); F2s.append(f2(P, R))
    n = len(F2s) or 1
    return sum(Ps)/n, sum(Rs)/n, sum(F2s)/n


TOP_KS = [2, 3, 5, 8]
MIN_SCORES = [None, -2, -1, 0, 1]
MARGINS = [None, 2, 3, 4]

rows = []
for tk, ms, mg in product(TOP_KS, MIN_SCORES, MARGINS):
    P, R, F = score_config(tk, ms, mg)
    rows.append((F, P, R, tk, ms, mg))
rows.sort(key=lambda r: r[0], reverse=True)   # sort by F2 only (tuple has None values)

print(f"{'F2':>6} {'P':>6} {'R':>6} | top_k  min_score  margin")
print("-" * 50)
for F, P, R, tk, ms, mg in rows[:12]:
    print(f"{F:6.3f} {P:6.3f} {R:6.3f} |   {tk:<4} {str(ms):>8}   {str(mg):>6}")

bestF, bestP, bestR, btk, bms, bmg = rows[0]
print("\n" + "=" * 50)
print(f"BEST: F2={bestF:.3f} (P={bestP:.3f} R={bestR:.3f})")
print(f"  RETRIEVE_TOP_K = {btk}")
print(f"  RETRIEVE_MIN_SCORE = {bms}")
print(f"  RETRIEVE_MARGIN = {bmg}")
print("=" * 50)
print(f"(baseline hiện tại: top_k=5, no cutoff → F2≈0.586)")
