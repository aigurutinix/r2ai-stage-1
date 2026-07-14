"""PASS 1: phân rã câu hỏi tình huống → cache data/subqueries.json (resumable).

Chỉ phân rã câu DÀI ở "vùng bán phần" (truy hồi 1-shot chưa chắc đủ). Câu ngắn
đã tốt (rr~0.95) → bỏ qua, giữ single-query. Câu rr quá thấp thường là thiếu data
(AI) → decompose không cứu được, vẫn tách nhưng ưu tiên thấp.

Usage:
  python scripts/decompose_questions.py --limit 200          # chạy thử 200 câu
  python scripts/decompose_questions.py                       # full câu dài
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
from backend.decompose import decompose, LLMClient

QFILE = Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json")
OUT = ROOT / "data/subqueries.json"


def top1(sc):
    cs = sc.get("candidates", []) if sc else []
    return max((c["rr"] for c in cs), default=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)       # 0 = không giới hạn
    ap.add_argument("--min-len", type=int, default=150)
    ap.add_argument("--lo", type=float, default=0.0)      # ngưỡng rr dưới
    ap.add_argument("--hi", type=float, default=0.90)     # ngưỡng rr trên (bỏ câu đã quá tốt)
    ap.add_argument("--scored", default="data/submission_v8_scored.json")  # gate theo pipeline hiện tại
    args = ap.parse_args()

    qs = json.loads(QFILE.read_text(encoding="utf-8"))
    scored = {r["id"]: r for r in json.loads((ROOT / args.scored).read_text(encoding="utf-8"))}
    cache = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    # chọn id cần phân rã
    targets = []
    for q in qs:
        if len(q["question"]) < args.min_len:
            continue
        t1 = top1(scored.get(q["id"]))
        if args.lo <= t1 <= args.hi:
            targets.append(q)
    targets.sort(key=lambda q: top1(scored.get(q["id"])))   # khó (rr thấp) trước
    if args.limit:
        targets = targets[:args.limit]
    todo = [q for q in targets if str(q["id"]) not in cache]
    print(f"Cần phân rã: {len(targets)} câu (đã cache {len(targets)-len(todo)}, còn {len(todo)})", flush=True)

    llm = LLMClient()
    t0 = time.time()
    for i, q in enumerate(todo, 1):
        subs = decompose(q["question"], llm)
        cache[str(q["id"])] = subs
        if i % 10 == 0 or i == len(todo):
            OUT.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [{i}/{len(todo)}] {i/(time.time()-t0):.2f} câu/s", flush=True)
    OUT.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    n_multi = sum(1 for v in cache.values() if len(v) >= 2)
    print(f"XONG: {len(cache)} câu trong cache | {n_multi} câu tách ≥2 vế → {OUT}")


if __name__ == "__main__":
    main()
