"""Tune KEEP + lọc version OFFLINE từ sidecar v36 (KHÔNG rebuild). Sửa lỗi v34: keep quá chặt.

Điểm reranker FT PHÂN CỰC (hoặc ~1 hoặc ~0) → ngưỡng tuyệt đối 0.40 vô dụng. Dùng keep
THEO RANK / RATIO-tương-đối (robust với điểm thấp). + collapse_versions (recall-safe) lọc bản cũ.

Sinh nhiều biến thể keep, in điều/câu + recall-proxy (v24) để chọn → pack nộp.

Chạy: PYTHONUTF8=1 PYTHONPATH=. python scripts/build_v36_variants.py
"""
from __future__ import annotations

import json
import sys
import statistics
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from exp_v24 import collapse_versions  # noqa: E402

QFILE = Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json")
V20 = ROOT / "data/submission_v20_clean.json"


def rebuild_docs(arts):
    docs, seen = [], set()
    for a in arts:
        p = a.split("|"); doc = f"{p[0].strip()}|{p[1].strip()}" if len(p) >= 2 else p[0].strip()
        if doc not in seen:
            seen.add(doc); docs.append(doc)
    return docs


def keep_ratio(cands, ratio, max_k, min_k):
    """Giữ top-1 + ứng viên có rr >= ratio*top_rr (tương đối → robust điểm phân cực)."""
    if not cands:
        return []
    top = cands[0]["rr"] or 0.0
    thr = ratio * top if top > 0 else -1
    out = [cands[0]["art"]]
    for c in cands[1:max_k]:
        if (c["rr"] or 0) >= thr and (c["rr"] or 0) > 0:
            out.append(c["art"])
    while len(out) < min_k and len(out) < len(cands):
        out.append(cands[len(out)]["art"])
    return out


def keep_topk(cands, k):
    return [c["art"] for c in cands[:k]]


def main() -> None:
    side = json.loads((ROOT / "data/v36_cands.json").read_text(encoding="utf-8"))
    qs = json.loads(QFILE.read_text(encoding="utf-8"))
    ans = {r["id"]: r.get("answer", "") for r in json.loads(V20.read_text(encoding="utf-8"))}
    v24 = {str(r["id"]): r["relevant_articles"]
           for r in json.loads((ROOT / "data/submission_v24_penalty.json").read_text(encoding="utf-8"))}

    def akey(a):
        p = a.split("|"); return (p[0].strip(), p[-1].strip())

    VARIANTS = {
        "topk2":      lambda c: keep_topk(c, 2),
        "topk3":      lambda c: keep_topk(c, 3),
        "ratio50_k4": lambda c: keep_ratio(c, 0.50, 4, 2),
        "ratio35_k5": lambda c: keep_ratio(c, 0.35, 5, 2),
        "ratio65_k4": lambda c: keep_ratio(c, 0.65, 4, 2),
    }

    for name, fn in VARIANTS.items():
        out = []
        g = hit = npick = 0
        for q in qs:
            qid = q["id"]
            cands = side.get(str(qid), [])
            arts = collapse_versions(fn(cands))   # lọc bản cũ recall-safe
            out.append({"id": qid, "question": q["question"], "answer": ans.get(qid, ""),
                        "relevant_docs": rebuild_docs(arts), "relevant_articles": arts})
            gold = {akey(a) for a in v24.get(str(qid), [])}
            sel = {akey(a) for a in arts}
            g += len(gold); hit += len(gold & sel); npick += len(arts)
        avg = npick / len(out)
        json.dump(out, open(ROOT / f"data/submission_v36_{name}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"{name:12s} | TB {avg:.2f} điều/câu | recall-proxy(v24) {hit/g:.1%} → submission_v36_{name}.json", flush=True)


if __name__ == "__main__":
    main()
