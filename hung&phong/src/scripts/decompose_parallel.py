"""Pha A: phân rã SÂU câu tình huống → data/subqueries_deep.json (song song, resumable).

Khác decompose_questions.py: chạy SONG SONG (ThreadPool Ollama), prompt sâu hơn (cap 6,
tách vế ẩn, think=True), phủ mọi câu DÀI (>= min-len) không cần gate theo điểm cũ.

Chạy: python scripts/decompose_parallel.py --workers 4 --min-len 120
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.decompose import decompose
from backend.llm import LLMClient

QFILE = Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json")
OUT = ROOT / "data/subqueries_deep.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-len", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    qs = json.loads(QFILE.read_text(encoding="utf-8"))
    cache = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    targets = [q for q in qs if len(q["question"]) >= args.min_len and str(q["id"]) not in cache]
    if args.limit:
        targets = targets[: args.limit]
    print(f"Câu cần phân rã (>= {args.min_len} ký tự, chưa cache): {len(targets)} "
          f"| đã cache {len(cache)} | workers {args.workers}", flush=True)

    llm = LLMClient()
    lock = threading.Lock()
    t0 = time.time()
    done = 0

    def work(q: dict):
        subs = decompose(q["question"], llm)
        return str(q["id"]), subs

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, q): q for q in targets}
        for fut in as_completed(futs):
            qid, subs = fut.result()
            with lock:
                cache[qid] = subs
                done += 1
                if done % 20 == 0:
                    OUT.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
                    rate = done / (time.time() - t0)
                    eta = (len(targets) - done) / rate / 60 if rate else 0
                    print(f"  [{done}/{len(targets)}] {rate:.2f} câu/s · ETA {eta:.0f} phút", flush=True)

    OUT.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    import statistics
    multi = [len(v) for v in cache.values() if len(v) >= 2]
    print(f"XONG: {len(cache)} câu cache | {len(multi)} câu ≥2 vế | TB vế {statistics.mean(multi):.2f} "
          f"| max {max(len(v) for v in cache.values())} → {OUT}", flush=True)


if __name__ == "__main__":
    main()
