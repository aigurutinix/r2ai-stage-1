"""Xây pseudo-gold từ TẤT CẢ các lần nộp đã có (v2–v11).

Nguyên lý voting: điều luật được nhiều version chất lượng cao đồng thuận
→ confidence cao → pseudo-gold. Hai loại nguồn:

  - scored file: có rr (rerank score) → dùng rr × weight
  - submission file: chỉ có relevant_articles → dùng 1.0 × weight

Confidence tổng hợp = sum(score × weight) / sum(weight_max_per_version)
→ trong khoảng [0, 1] → dễ đặt threshold.

Chạy:
  python scripts/build_pseudo_gold.py          # dùng mặc định v2-v10
  python scripts/build_pseudo_gold.py --add-sub v11 data/submission_v11.json 1.1
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "data" / "pseudo_gold.json"

# ── Nguồn mặc định ────────────────────────────────────────────────────────────
# (tên, đường dẫn, weight, loại: "scored" hoặc "sub")
# weight phản ánh chất lượng version: cao hơn = tin cậy hơn
DEFAULT_SOURCES = [
    # scored files (có rr riêng từng ứng viên — signal mạnh hơn)
    ("v4sc", "data/submission_v4_scored.json",  0.70, "scored"),
    ("v7sc", "data/submission_v7_scored.json",  0.90, "scored"),
    ("v8sc", "data/submission_v8_scored.json",  1.00, "scored"),
    # submission files (chỉ có relevant_articles — signal nhị phân)
    ("v2",   "data/submission_v2.json",         0.30, "sub"),
    ("v3",   "data/submission_v3.json",         0.20, "sub"),   # precision kém
    ("v4a",  "data/submission_v4_adaptive.json",0.65, "sub"),
    ("v5",   "data/submission_v5.json",         0.75, "sub"),
    ("v7",   "data/submission_v7.json",         0.85, "sub"),
    ("v8",   "data/submission_v8.json",         0.95, "sub"),
    ("v9",   "data/submission_v9.json",         0.90, "sub"),
    ("v10",  "data/submission_v10.json",        0.85, "sub"),
]

_CONF_THRESH  = 0.55   # confidence tối thiểu để đưa vào pseudo-gold
_RATIO_THRESH = 0.82   # top-2 phải ≥ 82% top-1 mới giữ (2 điều đều gold)


def art_key(art: str) -> str:
    """'so_ky_hieu|name|Điều N' → 'so_ky_hieu#N'"""
    p = art.split("|")
    sk   = p[0].strip()
    dieu = p[-1].strip().replace("Điều ", "").strip() if len(p) >= 3 else "0"
    return f"{sk}#{dieu}"


def load_scored(path: Path, weight: float, agg: dict, ver: str):
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        qid = row["id"]
        for c in (row.get("candidates") or []):
            art = c.get("art", "")
            rr  = float(c.get("rr") or c.get("adj") or 0.0)
            if not art or rr <= 0:
                continue
            k = art_key(art)
            d = agg[qid].setdefault(k, {"art": art, "score": 0.0, "src": []})
            d["score"] += rr * weight
            if ver not in d["src"]:
                d["src"].append(ver)
    return len(rows)


def load_sub(path: Path, weight: float, agg: dict, ver: str):
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        qid = row["id"]
        for art in (row.get("relevant_articles") or []):
            k = art_key(art)
            d = agg[qid].setdefault(k, {"art": art, "score": 0.0, "src": []})
            d["score"] += 1.0 * weight      # binary: xuất hiện = 1.0
            if ver not in d["src"]:
                d["src"].append(ver)
    return len(rows)


def build(sources: list[tuple]) -> dict:
    # agg[qid][art_key] = {art, score, src}
    agg: dict = defaultdict(dict)
    total_weight = sum(w for _, _, w, _ in sources)

    for ver, rel_path, weight, kind in sources:
        path = ROOT / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        if not path.exists():
            print(f"  ⚠  {ver}: bỏ qua (không tìm thấy {path.name})", flush=True)
            continue
        if kind == "scored":
            n = load_scored(path, weight, agg, ver)
        else:
            n = load_sub(path, weight, agg, ver)
        print(f"  {ver:6s} [{kind:6s}] weight={weight:.2f}  {n} câu", flush=True)

    # Normalize score → [0,1] bằng cách chia cho total_weight × 1.0 (max rr=1)
    for qid in agg:
        for k in agg[qid]:
            agg[qid][k]["score"] /= total_weight

    # Chọn gold cho mỗi câu
    pseudo: dict[str, dict] = {}
    n_q1 = n_q2 = n_skip = 0

    for qid, arts in agg.items():
        ranked = sorted(arts.values(), key=lambda x: x["score"], reverse=True)
        if not ranked or ranked[0]["score"] < _CONF_THRESH:
            n_skip += 1
            continue

        top     = ranked[0]
        chosen  = [top["art"]]

        if (len(ranked) >= 2
                and ranked[1]["score"] >= _RATIO_THRESH * top["score"]):
            chosen.append(ranked[1]["art"])
            n_q2 += 1
        else:
            n_q1 += 1

        pseudo[str(qid)] = {
            "articles":   chosen,
            "confidence": round(top["score"], 4),
            "sources":    top["src"],
        }

    print(f"\nPseudo-gold: {len(pseudo)} câu  "
          f"| 1-article={n_q1}  2-article={n_q2}  skip={n_skip}", flush=True)
    return pseudo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-scored", nargs=3,
                    metavar=("NAME","PATH","WEIGHT"), action="append", default=[])
    ap.add_argument("--add-sub",    nargs=3,
                    metavar=("NAME","PATH","WEIGHT"), action="append", default=[])
    ap.add_argument("--thresh", type=float, default=_CONF_THRESH,
                    help="Confidence threshold (default %(default)s)")
    args = ap.parse_args()

    sources = list(DEFAULT_SOURCES)
    for name, path, weight in args.add_scored:
        sources.append((name, path, float(weight), "scored"))
    for name, path, weight in args.add_sub:
        sources.append((name, path, float(weight), "sub"))

    print(f"Tổng {len(sources)} nguồn:", flush=True)
    pseudo = build(sources)
    OUT.write_text(json.dumps(pseudo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {OUT}", flush=True)

    confs = sorted([v["confidence"] for v in pseudo.values()], reverse=True)
    if confs:
        print(f"Confidence: top-10%={confs[len(confs)//10]:.3f}  "
              f"median={confs[len(confs)//2]:.3f}  "
              f"bottom-10%={confs[-len(confs)//10]:.3f}", flush=True)


if __name__ == "__main__":
    main()
