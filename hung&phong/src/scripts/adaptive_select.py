"""Chọn điều THÍCH ỨNG từ submission_v4_scored (theo điểm reranker) — mỗi câu một
số điều khác nhau tùy độ tự tin. Đây là điểm khác biệt vs đối thủ (cố định K).

Quy tắc: giữ điều có rr >= max(T_abs, ratio*top_rr), trong [min_k, max_k].
  - Câu chỉ 1 điều rõ ràng (top cao, còn lại tụt) → giữ ít → precision cao.
  - Câu nhiều điều ngang nhau (đều cao) → giữ nhiều → recall cao.

Usage: python scripts/adaptive_select.py --t-abs 0.5 --ratio 0.6 --min 1 --max 8 \
       --out data/submission_v4.json   (in phân bố số điều/câu)
"""
import argparse, json, statistics, sys
from collections import Counter
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")


def select(cands, t_abs, ratio, min_k, max_k):
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: c["rr"], reverse=True)
    top = cands[0]["rr"]
    cut = max(t_abs, ratio * top)
    chosen = [c for c in cands if c["rr"] >= cut][:max_k]
    if len(chosen) < min_k:
        chosen = cands[:min_k]
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="data/submission_v4_scored.json")
    ap.add_argument("--out", default="data/submission_v4.json")
    ap.add_argument("--t-abs", type=float, default=0.5)
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--min", type=int, default=1)
    ap.add_argument("--max", type=int, default=8)
    args = ap.parse_args()

    d = json.loads(Path(args.scored).read_text(encoding="utf-8"))
    out, counts = [], []
    for r in d:
        chosen = select(r["candidates"], args.t_abs, args.ratio, args.min, args.max)
        arts, docs, seen_d = [], [], set()
        for c in chosen:
            arts.append(c["art"])
            if c["doc"] not in seen_d:
                seen_d.add(c["doc"]); docs.append(c["doc"])
        counts.append(len(arts))
        out.append({"id": r["id"], "question": r["question"], "answer": r["answer"],
                    "relevant_docs": docs, "relevant_articles": arts})
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    dist = Counter(counts)
    print(f"T_abs={args.t_abs} ratio={args.ratio} min={args.min} max={args.max}")
    print(f"  TB {statistics.mean(counts):.2f} điều/câu · median {statistics.median(counts)} · "
          f"min {min(counts)} max {max(counts)}")
    print(f"  Phân bố: " + " ".join(f"{k}đ:{dist[k]}" for k in sorted(dist)))
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()
