"""Pha D: sinh CÂU TRẢ LỜI grounded từ các điều đã chọn (song song, resumable).

Với mỗi câu: lấy toàn văn các điều trong relevant_articles → Qwen viết câu trả lời pháp lý
NGẮN GỌN, chính xác, bám điều luật (cho phần QA: chính xác/đầy đủ/thực tiễn/rõ ràng).

Chạy: QDRANT_COLLECTION=vbpl_aiteam QDRANT_URL=http://localhost:6333
      python scripts/gen_answers_parallel.py --in data/submission_xx.json \
        --out data/submission_final.json --workers 4
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

from llm_judge import load_text_lookup  # noqa: E402

_SYSTEM = (
    "Bạn là chuyên gia pháp luật Việt Nam. Trả lời câu hỏi CHÍNH XÁC dựa trên các điều luật "
    "được cung cấp, không bịa thông tin ngoài điều luật."
)
_USER = """Câu hỏi: {q}

Các điều luật liên quan:
{block}

Hãy trả lời câu hỏi trên NGẮN GỌN, chính xác, đúng trọng tâm, dựa trên nội dung các điều luật.
- Nếu câu hỏi có nhiều vế, trả lời lần lượt từng vế.
- Dẫn số điều / số ký hiệu văn bản khi nêu căn cứ (vd "theo Điều 4 Luật 04/2017/QH14").
- Viết liền mạch, không gạch đầu dòng dài dòng, không lặp lại câu hỏi.
- KHÔNG thêm thông tin không có trong các điều luật trên."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="data/answer_cache.json")
    ap.add_argument("--collection", default="vbpl_aiteam")
    ap.add_argument("--workers", type=int, default=4)
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
    cache: dict[str, str] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    lock = threading.Lock()
    print(f"Sinh đáp án {len(data)} câu | cache {len(cache)} | workers {args.workers}", flush=True)

    def block_of(arts: list[str]) -> str:
        parts = []
        for a in arts:
            t = texts.get(a, "")
            parts.append(f"[{a}]\n{t[:1500]}")
        return "\n\n".join(parts)

    def work(row: dict):
        qid = str(row["id"])
        with lock:
            if qid in cache and cache[qid]:
                return qid, cache[qid]
        arts = row.get("relevant_articles", [])
        if not arts:
            return qid, row.get("answer", "")
        try:
            ans = llm.complete(_SYSTEM, _USER.format(q=row["question"], block=block_of(arts)), think=False).strip()
        except Exception as e:  # noqa: BLE001
            ans = row.get("answer", "")
            print(f"  lỗi id={qid}: {repr(e)[:60]}", flush=True)
        with lock:
            cache[qid] = ans
        return qid, ans

    answers: dict[str, str] = {}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r for r in data}
        for fut in as_completed(futs):
            qid, ans = fut.result()
            answers[qid] = ans
            done += 1
            if done % 25 == 0:
                with lock:
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                rate = done / (time.time() - t0)
                print(f"  [{done}/{len(data)}] {rate:.2f} câu/s · ETA {(len(data)-done)/rate/60:.0f} phút", flush=True)

    out = [{**r, "answer": answers.get(str(r["id"]), r.get("answer", ""))} for r in data]
    Path(ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"XONG: {len(out)} câu có đáp án ({time.time()-t0:.0f}s) → {args.out}", flush=True)


if __name__ == "__main__":
    main()
