"""Đánh giá submission local bằng pseudo-gold (không cần nộp BTC).

Chạy trong vài giây. Kết quả là estimate — không hoàn toàn khớp BTC vì:
  1. Pseudo-gold chưa hoàn hảo (chỉ có ~60-70% câu, bỏ qua câu không chắc)
  2. BTC có thể dùng 50 câu khác nhau mỗi lần chấm

Nhưng đủ để so sánh tương đối giữa các version trước khi nộp thật.

Chạy:
  python scripts/eval_local.py data/submission_v9.json
  python scripts/eval_local.py data/submission_v11.json --gold data/pseudo_gold.json
  python scripts/eval_local.py data/submission_v9.json data/submission_v11.json  # so sánh 2 bản
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = ROOT / "data" / "pseudo_gold.json"


def art_key(art: str) -> str:
    parts = art.split("|")
    sk = parts[0].strip()
    dieu = parts[-1].strip().replace("Điều ", "").strip() if len(parts) >= 3 else "0"
    return f"{sk}#{dieu}"


def f2(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 5 * p * r / (4 * p + r)


def score_submission(sub_path: str, gold: dict) -> dict:
    rows = json.loads((ROOT / sub_path if not Path(sub_path).is_absolute()
                       else Path(sub_path)).read_text(encoding="utf-8"))
    sub = {str(r["id"]): r for r in rows}

    total_f2 = total_p = total_r = 0.0
    n = 0
    zero_recall = 0

    details = []
    for qid, gdata in gold.items():
        if qid not in sub:
            continue
        gold_keys = {art_key(a) for a in gdata["articles"]}
        pred_arts = sub[qid].get("relevant_articles") or []
        pred_keys = {art_key(a) for a in pred_arts}

        tp = len(gold_keys & pred_keys)
        precision = tp / len(pred_keys) if pred_keys else 0.0
        recall = tp / len(gold_keys) if gold_keys else 0.0
        f2_q = f2(precision, recall)

        total_f2 += f2_q
        total_p += precision
        total_r += recall
        n += 1
        if recall == 0:
            zero_recall += 1
        details.append({"id": int(qid), "f2": f2_q, "p": precision, "r": recall,
                         "conf": gdata["confidence"], "sources": gdata["sources"]})

    if n == 0:
        return {"error": "Không có câu nào khớp"}

    results = {
        "n_questions": n,
        "ART_F2_est": round(total_f2 / n, 4),
        "ART_P_est": round(total_p / n, 4),
        "ART_R_est": round(total_r / n, 4),
        "zero_recall_pct": round(zero_recall / n * 100, 1),
        "details": sorted(details, key=lambda x: x["f2"]),
    }
    return results


def print_results(label: str, r: dict):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    if "error" in r:
        print(f"  LỖI: {r['error']}")
        return
    print(f"  Câu eval  : {r['n_questions']}")
    print(f"  ART_F2_est: {r['ART_F2_est']:.4f}")
    print(f"  ART_P_est : {r['ART_P_est']:.4f}")
    print(f"  ART_R_est : {r['ART_R_est']:.4f}")
    print(f"  Zero-recall: {r['zero_recall_pct']:.1f}%")

    # Top 5 câu tệ nhất (f2=0)
    worst = [d for d in r["details"] if d["f2"] == 0][:5]
    if worst:
        print(f"\n  -- {len([d for d in r['details'] if d['f2']==0])} câu recall=0 (top 5):")
        for d in worst:
            print(f"     ID {d['id']:4d} | conf={d['conf']:.2f} | src={d['sources']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submissions", nargs="+", help="1 hoặc nhiều file submission để so sánh")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="Chỉ dùng câu pseudo-gold có confidence >= ngưỡng này")
    ap.add_argument("--worst", type=int, default=10,
                    help="In N câu tệ nhất (f2 thấp nhất)")
    args = ap.parse_args()

    gold_path = ROOT / args.gold if not Path(args.gold).is_absolute() else Path(args.gold)
    if not gold_path.exists():
        print(f"Chưa có {gold_path}. Chạy: python scripts/build_pseudo_gold.py", flush=True)
        sys.exit(1)

    gold_all = json.loads(gold_path.read_text(encoding="utf-8"))
    gold = {k: v for k, v in gold_all.items() if v["confidence"] >= args.min_conf}
    print(f"Pseudo-gold: {len(gold)}/{len(gold_all)} câu (min_conf={args.min_conf})", flush=True)

    all_results = {}
    for sub in args.submissions:
        label = Path(sub).name
        r = score_submission(sub, gold)
        all_results[label] = r
        print_results(label, r)

    # So sánh nếu nhiều hơn 1 submission
    if len(args.submissions) >= 2:
        print(f"\n{'='*55}")
        print("  SO SÁNH")
        print(f"{'='*55}")
        items = [(k, v) for k, v in all_results.items() if "ART_F2_est" in v]
        items.sort(key=lambda x: x[1]["ART_F2_est"], reverse=True)
        for rank, (label, r) in enumerate(items, 1):
            print(f"  #{rank} {label:40s} F2={r['ART_F2_est']:.4f}  "
                  f"P={r['ART_P_est']:.4f}  R={r['ART_R_est']:.4f}")

    # In worst-N nếu 1 submission
    if len(args.submissions) == 1 and args.worst > 0:
        r = list(all_results.values())[0]
        if "details" in r:
            worst = sorted(r["details"], key=lambda x: x["f2"])[:args.worst]
            print(f"\n  -- {args.worst} câu F2 thấp nhất:")
            for d in worst:
                print(f"     ID {d['id']:4d} | f2={d['f2']:.3f} "
                      f"p={d['p']:.2f} r={d['r']:.2f} | conf={d['conf']:.2f}")


if __name__ == "__main__":
    main()
