"""Zalo-2021 eval harness — thước đo F2 LOCAL, hết bay mù trên leaderboard 50 câu ẩn.

Zalo AI 2021 Legal Text Retrieval: ~3.2K câu hỏi tiếng Việt CÓ GÁN NHÃN điều luật
(law_id + article_id → map thẳng sang khóa (mã văn bản, số Điều) của ta). Dataset chỉ
làm THƯỚC ĐO offline — không nạp vào corpus/pipeline (tránh rủi ro "external data" §9).

Lưu ý đọc số: phân bố Zalo ≠ 2000 câu SME của BTC → tin CHÊNH LỆCH TƯƠNG ĐỐI giữa các
config, đừng tin số tuyệt đối. Chỉ chấm trên câu mà TOÀN BỘ gold nằm trong corpus ta
(coverage-adjusted) để tách "retrieval kém" khỏi "corpus thiếu".

Modes:
    prep            tải HF → data/zalo_eval.json (questions + gold) + báo coverage
    run  [--limit]  chạy retrieval (engine local, MPS/CPU) → data/zalo_retrieved.json
    score [--verified verified.json]   chấm P/R/F2 macro cho các config cutoff
"""
import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
import local_models_config as cfg
from retrieval_cutoff import apply_cutoff, drop_superseded

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_JSON = os.path.join(cfg.DATA_DIR, "zalo_eval.json")
RETR_JSON = os.path.join(cfg.DATA_DIR, "zalo_retrieved.json")
HF_ID = "minhnguyent546/zalo-ai-legal-text-retrieval-2021"


def _key(code, article):
    """(mã văn bản chuẩn hóa, số Điều) — y hệt eval_f2/grader join."""
    code = re.sub(r"\s+", "", str(code or "")).upper()
    m = re.search(r"(\d+)", str(article or ""))
    return (code, m.group(1)) if code and m else None


def _corpus_keys():
    keys = set()
    with open(cfg.CORPUS_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                k = _key(r.get("doc_number"), r.get("article"))
                if k:
                    keys.add(k)
    return keys


def _load_cfg(name):
    """load 1 config với retry (mạng yếu)."""
    from datasets import load_dataset
    for attempt in range(5):
        try:
            return load_dataset(HF_ID, name)
        except Exception as e:
            print(f"  {name} retry {attempt+1}: {str(e)[:90]}"); time.sleep(4)
    sys.exit(f"Không tải được config {name}.")


def prep():
    # BEIR-style: queries(query_id, question) + qrels(query_id, corpus_id="law#art", score)
    queries, qrels = _load_cfg("queries"), _load_cfg("qrels")
    qtext = {}
    for sp in queries:
        for r in queries[sp]:
            qtext[r["query_id"]] = r["question"]
    gold_by_q = {}
    for sp in qrels:
        for r in qrels[sp]:
            law, _, art = (r["corpus_id"] or "").partition("#")
            k = _key(law, art)
            if k and r["query_id"] in qtext:
                gold_by_q.setdefault(r["query_id"], []).append(f"{k[0]}|Điều {k[1]}")
    items = [{"id": i + 1, "question": qtext[qid], "gold": sorted(set(g))}
             for i, (qid, g) in enumerate(sorted(gold_by_q.items()))]
    print(f"loaded: {len(items)} câu có gold (queries {len(qtext)}, qrels splits {list(qrels.keys())})")

    ck = _corpus_keys()
    n_all = n_any = 0
    for it in items:
        ks = [_key(*g.split("|")) for g in it["gold"]]
        hit = [k in ck for k in ks]
        it["cover_all"] = all(hit)
        it["cover_any"] = any(hit)
        n_all += it["cover_all"]; n_any += it["cover_any"]
    json.dump(items, open(EVAL_JSON, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"coverage vs corpus 93K: ALL-gold-in-corpus {n_all}/{len(items)} ({100*n_all/len(items):.0f}%) "
          f"| ANY {n_any}/{len(items)} ({100*n_any/len(items):.0f}%)")
    print(f"→ {EVAL_JSON} (chấm mặc định trên nhóm ALL = đo retrieval thuần, không lẫn lỗi thiếu corpus)")


def run(limit, pool):
    items = [it for it in json.load(open(EVAL_JSON, encoding="utf-8")) if it["cover_all"]]
    random.Random(42).shuffle(items)
    items = items[:limit] if limit else items
    print(f"retrieval trên {len(items)} câu (cover_all, seed 42, pool={pool})")

    from local_rag_engine import LocalLegalRAGEngine   # nặng — import tại đây
    eng = LocalLegalRAGEngine(use_reranker=True)
    out, t0 = {}, time.time()
    for n, it in enumerate(items):
        cands = eng._candidates_scored(it["question"], pool=pool)
        out[str(it["id"])] = [
            {"doc_number": d.get("doc_number"), "clean_name": d.get("clean_name"),
             "article": d.get("article"), "score": float(d.get("rerank_score", 0.0))}
            for d in cands
        ]
        if (n + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(items)} | {el/ (n+1):.1f}s/câu | ETA {(len(items)-n-1)*el/(n+1)/60:.0f}p")
            json.dump(out, open(RETR_JSON, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(out, open(RETR_JSON, "w", encoding="utf-8"), ensure_ascii=False)
    print("→", RETR_JSON)


def _f2(p, r):
    return (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0.0


def score(verified_path):
    gold = {str(it["id"]): {_key(*g.split("|")) for g in it["gold"]}
            for it in json.load(open(EVAL_JSON, encoding="utf-8"))}
    cache = json.load(open(RETR_JSON, encoding="utf-8"))
    if verified_path:
        flags = json.load(open(verified_path, encoding="utf-8"))
        for qid, cands in cache.items():
            f = flags.get(qid, [])
            for i, c in enumerate(cands):
                c["llm_ok"] = bool(f[i]) if i < len(f) else False

    # (tag, top_k, margin, verified_mode)
    grid = [("t3m15", 3, 1.5, False), ("t3m2", 3, 2.0, False), ("t5m3", 5, 3.0, False),
            ("t8m6", 8, 6.0, False), ("t12flat", 12, None, False)]
    if verified_path:
        grid += [("v_k4", 4, None, True), ("v_k6", 6, None, True), ("v_k8", 8, None, True)]

    print(f"chấm trên {len(cache)} câu (cover_all)\n{'tag':9} {'P':>6} {'R':>6} {'F2':>6} {'art/q':>6}")
    for tag, tk, mg, ver in grid:
        Ps, Rs, Fs, na = [], [], [], 0
        for qid, cands in cache.items():
            g = gold.get(qid, set())
            if not g:
                continue
            cs = drop_superseded(cands)
            if ver:
                ctx = (cs[:1] + [c for c in cs[1:] if c.get("llm_ok")])[:tk] if cs else []
            else:
                ctx = apply_cutoff(cs, tk, None, mg, score_key="score")
            pred = {k for k in (_key(c.get("doc_number"), c.get("article")) for c in ctx) if k}
            na += len(pred)
            c = len(pred & g)
            P = c / len(pred) if pred else 0.0
            R = c / len(g) if g else 0.0
            Ps.append(P); Rs.append(R); Fs.append(_f2(P, R))
        n = len(Fs) or 1
        print(f"{tag:9} {sum(Ps)/n:6.3f} {sum(Rs)/n:6.3f} {sum(Fs)/n:6.3f} {na/n:6.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "run", "score"])
    ap.add_argument("--limit", type=int, default=300, help="số câu chạy retrieval (0=hết)")
    ap.add_argument("--pool", type=int, default=12)
    ap.add_argument("--verified", default=None, help="verified.json (Phase V) để chấm chế độ v_*")
    a = ap.parse_args()
    if a.mode == "prep":
        prep()
    elif a.mode == "run":
        run(a.limit, a.pool)
    else:
        score(a.verified)
