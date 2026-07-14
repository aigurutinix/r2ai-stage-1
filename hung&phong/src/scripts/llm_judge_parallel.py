"""LLM-judge SONG SONG: lọc điều nhiễu khỏi submission, gọi Qwen đồng thời nhiều luồng.

Khác `llm_judge.py` (tuần tự) ở chỗ dùng ThreadPoolExecutor → cắt thời gian từ ~5h xuống
<~1.5h. Logic judge y hệt (tái dùng helper). Judge chỉ lọc `relevant_articles`, KHÔNG đụng
`answer`. Cache RIÊNG theo input (cache cũ lệch độ dài → judge thành no-op).

Lưu ý: để Ollama xử lý song song thật, cần `OLLAMA_NUM_PARALLEL>=workers` khi khởi động
Ollama. Nếu Ollama đang NUM_PARALLEL=1 thì các request bị xếp hàng (không nhanh hơn).

Chạy:
  QDRANT_COLLECTION=vbpl_aiteam QDRANT_URL=http://localhost:6333
  python scripts/llm_judge_parallel.py --in data/submission_v15_dedup.json \
     --out data/submission_v16_judge.json --cache data/judge_cache_v15.json --workers 6
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
sys.path.insert(0, str(ROOT / "scripts"))

from llm_judge import judge_question, load_text_lookup  # noqa: E402


def _rebuild_docs(kept: list[str]) -> list[str]:
    docs, seen = [], set()
    for a in kept:
        parts = a.split("|")
        doc = "|".join(parts[:2]) if len(parts) >= 2 else parts[0]
        if doc not in seen:
            seen.add(doc)
            docs.append(doc)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/submission_v15_dedup.json")
    ap.add_argument("--out", default="data/submission_v16_judge.json")
    ap.add_argument("--cache", default="data/judge_cache_v15.json")
    ap.add_argument("--collection", default="vbpl_aiteam")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import os
    os.environ.setdefault("QDRANT_COLLECTION", args.collection)
    from backend.llm import LLMClient

    llm = LLMClient()
    texts = load_text_lookup(args.collection)

    data = json.loads(Path(ROOT / args.inp).read_text(encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]
    cache_path = ROOT / args.cache
    cache: dict[str, list[bool]] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    )
    lock = threading.Lock()
    print(f"Tổng {len(data)} câu | cache {len(cache)} | workers {args.workers}", flush=True)

    def judge_one(row: dict):
        qid = str(row["id"])
        arts = row.get("relevant_articles", [])
        with lock:
            cached = cache.get(qid)
        if cached is not None and len(cached) == len(arts):
            return qid, cached
        if len(arts) <= 1:
            v = [True] * len(arts)
        else:
            v = judge_question(row["question"], arts, texts, llm)
        with lock:
            cache[qid] = v
        return qid, v

    verdicts: dict[str, list[bool]] = {}
    t0 = time.time()
    done = 0
    todo = [r for r in data if r.get("relevant_articles")]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, r): r for r in todo}
        for fut in as_completed(futs):
            qid, v = fut.result()
            verdicts[qid] = v
            done += 1
            if done % 25 == 0:
                with lock:
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                rate = done / (time.time() - t0)
                eta = (len(todo) - done) / rate / 60 if rate else 0
                print(f"  [{done}/{len(todo)}] {rate:.2f} câu/s · ETA {eta:.0f} phút", flush=True)

    # Build output theo THỨ TỰ GỐC
    out, n_filtered, n_fallback = [], 0, 0
    for row in data:
        qid = str(row["id"])
        arts = row.get("relevant_articles", [])
        v = verdicts.get(qid, [True] * len(arts))
        if len(v) != len(arts):
            v = [True] * len(arts)
        kept = [a for a, k in zip(arts, v) if k]
        if arts and not kept:
            kept = arts[:1]
            n_fallback += 1
        elif len(kept) < len(arts):
            n_filtered += 1
        out.append({
            "id": row["id"], "question": row["question"], "answer": row["answer"],
            "relevant_docs": _rebuild_docs(kept), "relevant_articles": kept,
        })

    Path(ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tot_in = sum(len(r.get("relevant_articles", [])) for r in data)
    tot_out = sum(len(r["relevant_articles"]) for r in out)
    print(f"XONG: {len(out)} câu | article {tot_in}→{tot_out} | lọc {n_filtered} câu | "
          f"fallback {n_fallback} | {time.time()-t0:.0f}s → {args.out}", flush=True)


if __name__ == "__main__":
    main()
