"""v48 — judge phần THÊM của v46 (precision tune). Qwen chấm CÓ/KHÔNG từng điều v46 thêm,
giữ v33 base + điều-thêm-CÓ. Recall-safe cho base (không đụng v33), lọc nhiễu phần thêm.

Song song nhiều worker (cần OLLAMA_NUM_PARALLEL>=workers). Judge MỌI điều thêm (kể cả câu thêm 1).

Chạy: OLLAMA_NUM_PARALLEL=3 QDRANT_COLLECTION=vbpl_aiteam ... PYTHONUTF8=1 PYTHONPATH=. \
  python scripts/build_v48_judge.py --workers 3
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


def rebuild_docs(arts):
    docs, seen = [], set()
    for a in arts:
        p = a.split("|"); d = f"{p[0].strip()}|{p[1].strip()}" if len(p) >= 2 else p[0].strip()
        if d not in seen:
            seen.add(d); docs.append(d)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cache", default="data/judge_cache_v48.json")
    ap.add_argument("--out", default="data/submission_v48.json")
    args = ap.parse_args()

    from backend.llm import LLMClient
    from llm_judge import judge_question, load_text_lookup
    from exp_v24 import collapse_versions

    ji = {r["id"]: r for r in json.loads((ROOT / "data/_judgein_v48.json").read_text(encoding="utf-8"))}
    v46 = {r["id"]: r for r in json.loads((ROOT / "data/submission_v46.json").read_text(encoding="utf-8"))}
    v33 = {r["id"]: r["relevant_articles"] for r in json.loads((ROOT / "data/submission_v33_v24map.json").read_text(encoding="utf-8"))}

    def k(a):
        p = a.split("|"); return (p[0].strip(), p[-1].strip())

    print("Nạp text lookup...", flush=True)
    texts = load_text_lookup(os.environ.get("QDRANT_COLLECTION", "vbpl_aiteam"))
    llm = LLMClient()
    cache_path = ROOT / args.cache
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    lock = Lock()

    def judge_one(r):
        qid = str(r["id"]); arts = r["relevant_articles"]
        with lock:
            c = cache.get(qid)
        if c is not None and len(c) == len(arts):
            return qid, c
        try:
            v = judge_question(r["question"], arts, texts, llm)
        except Exception as e:  # noqa: BLE001
            v = [True] * len(arts)  # lỗi → giữ (recall-safe)
        if len(v) != len(arts):
            v = [True] * len(arts)
        with lock:
            cache[qid] = v
        return qid, v

    todo = list(ji.values())
    print(f"Judge {sum(len(r['relevant_articles']) for r in todo)} điều thêm / {len(todo)} câu | {args.workers} workers", flush=True)
    verdicts = {}; done = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, r): r for r in todo}
        for fut in as_completed(futs):
            qid, v = fut.result(); verdicts[qid] = v; done += 1
            if done % 25 == 0:
                with lock:
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print(f"  [{done}/{len(todo)}] {done/(time.time()-t0):.2f} câu/s ETA {(len(todo)-done)/(done/(time.time()-t0))/60:.0f}p", flush=True)

    # combine: v33 base + điều thêm CÓ
    out = []; n_keep = n_drop = 0
    for qid, r in v46.items():
        base = list(v33.get(qid, []))
        if qid not in ji:
            out.append(r); continue
        added = ji[qid]["relevant_articles"]; v = verdicts.get(str(qid), [True] * len(added))
        for a, keep in zip(added, v):
            if keep:
                base.append(a); n_keep += 1
            else:
                n_drop += 1
        arts = collapse_versions(base)
        nr = dict(r); nr["relevant_articles"] = arts; nr["relevant_docs"] = rebuild_docs(arts)
        out.append(nr)
    out.sort(key=lambda r: r["id"])
    json.dump(out, open(ROOT / args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    import statistics
    print(f"XONG: giữ {n_keep} điều thêm, BỎ {n_drop} (judge=KHÔNG) → {args.out}", flush=True)
    print(f"  TB điều/câu: {statistics.mean(len(r['relevant_articles']) for r in out):.2f} (v46 2.93)", flush=True)


if __name__ == "__main__":
    main()
