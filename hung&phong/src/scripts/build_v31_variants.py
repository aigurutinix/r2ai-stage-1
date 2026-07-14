"""Dựng các BIẾN THỂ nộp từ sidecar của build_listwise — KHÔNG chạy lại Qwen.

Sidecar (data/submission_v31_listwise_diag.json):
  { "<qid>": {"cands":[{"art","rr"},...] (đã xếp hạng reranker, sau collapse),
              "picked":[index Qwen chọn]} }

Biến thể:
  A (thuần)  : đúng những điều Qwen chọn (≈1.82 điều/câu, precision-play).
  B (+sàn)   : A + SÀN RECALL theo gap map (DIEU_KHUNG + VE_THU_TUC = 46%):
               - nếu chọn CÓ luật mà THIẾU NĐ/TT → thêm NĐ/TT điểm cao nhất (rr>=thresh)
               - nếu chọn CÓ NĐ/TT mà THIẾU luật → thêm luật điểm cao nhất (rr>=thresh)
               - nếu tổng <2 điều → thêm ứng viên kế tiếp điểm cao nhất (đảm bảo cặp).
               → khôi phục cặp luật↔NĐ mà listwise thuần hay bỏ sót.

Chạy (KHÔNG cần GPU):
  PYTHONUTF8=1 PYTHONPATH=. python scripts/build_v31_variants.py \
    --diag data/submission_v31_listwise_diag.json --floor-thresh 0.5
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from exp_cocite import loai_of  # noqa: E402  'law' | 'nd' | ''
from exp_v24 import collapse_versions  # noqa: E402

QFILE = Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json")
V20 = ROOT / "data/submission_v20_clean.json"


def rebuild_docs(arts: list[str]) -> list[str]:
    docs, seen = [], set()
    for a in arts:
        p = a.split("|")
        doc = f"{p[0].strip()}|{p[1].strip()}" if len(p) >= 2 else p[0].strip()
        if doc not in seen:
            seen.add(doc)
            docs.append(doc)
    return docs


def variant_pure(cands: list[dict], picked: list[int]) -> list[str]:
    return [cands[i]["art"] for i in picked if 0 <= i < len(cands)]


def variant_floor(cands: list[dict], picked: list[int], thresh: float) -> list[str]:
    """A + sàn recall: bù vế bổ trợ (luật↔NĐ) + đảm bảo >=2 điều."""
    arts = [cands[i]["art"] for i in picked if 0 <= i < len(cands)]
    have = set(arts)
    loais = {loai_of(a) for a in arts}
    # cands đã xếp theo rr giảm dần (thứ tự reranker)
    def add_first(want_loai: str) -> None:
        for c in cands:
            if c["art"] in have:
                continue
            if loai_of(c["art"]) == want_loai and float(c.get("rr", 0) or 0) >= thresh:
                arts.append(c["art"]); have.add(c["art"]); return

    if "law" in loais and "nd" not in loais:
        add_first("nd")            # có luật, thiếu NĐ/TT hướng dẫn → bù
    elif "nd" in loais and "law" not in loais:
        add_first("law")           # có NĐ/TT, thiếu luật mẹ → bù

    # đảm bảo tối thiểu 2 điều — nhưng CHỈ khi ứng viên kế tiếp đủ tự tin (rr>=thresh),
    # để câu ĐƠN-VẾ thật (vd khám thai → 1 điều BHXH) không bị nhồi nhiễu.
    if len(arts) < 2:
        for c in cands:
            if c["art"] not in have and float(c.get("rr", 0) or 0) >= thresh:
                arts.append(c["art"]); have.add(c["art"]); break

    return collapse_versions(arts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", default="data/submission_v31_listwise_diag.json")
    ap.add_argument("--floor-thresh", type=float, default=0.5)
    ap.add_argument("--out-a", default="data/submission_v31a_pure.json")
    ap.add_argument("--out-b", default="data/submission_v31b_floor.json")
    args = ap.parse_args()

    diag = json.loads((ROOT / args.diag).read_text(encoding="utf-8"))
    qs = json.loads(QFILE.read_text(encoding="utf-8"))
    ans = {r["id"]: r.get("answer", "") for r in json.loads(V20.read_text(encoding="utf-8"))}

    out_a, out_b = [], []
    n_floored = 0
    for q in qs:
        qid = q["id"]
        d = diag.get(str(qid))
        if not d:
            continue
        cands, picked = d["cands"], d["picked"]
        a = collapse_versions(variant_pure(cands, picked))
        b = variant_floor(cands, picked, args.floor_thresh)
        if len(b) > len(a):
            n_floored += 1
        base = {"id": qid, "question": q["question"], "answer": ans.get(qid, "")}
        out_a.append({**base, "relevant_docs": rebuild_docs(a), "relevant_articles": a})
        out_b.append({**base, "relevant_docs": rebuild_docs(b), "relevant_articles": b})

    (ROOT / args.out_a).write_text(json.dumps(out_a, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / args.out_b).write_text(json.dumps(out_b, ensure_ascii=False, indent=2), encoding="utf-8")
    avg_a = statistics.mean(len(r["relevant_articles"]) for r in out_a) if out_a else 0
    avg_b = statistics.mean(len(r["relevant_articles"]) for r in out_b) if out_b else 0
    print(f"A (thuần): {len(out_a)} câu | TB {avg_a:.3f} điều/câu → {args.out_a}")
    print(f"B (+sàn) : {len(out_b)} câu | TB {avg_b:.3f} điều/câu | bù sàn {n_floored} câu → {args.out_b}")


if __name__ == "__main__":
    main()
