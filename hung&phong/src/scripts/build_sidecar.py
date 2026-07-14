"""Build LƯU SIDECAR — retrieve + FT reranker, lưu top-N ứng viên/câu → tune keep + filter OFFLINE.

Mục tiêu (sau mổ v34): KEEP quá chặt (ngưỡng tuyệt đối 0.40 vô dụng với điểm phân cực) là lỗi
recall #1. Lưu ứng viên đã rerank → thử mọi cách keep (rank-based / ratio) + lọc version/domain
offline tức thì, không phải rebuild 4h mỗi lần.

Dense-only (HYBRID_SEARCH=false) cho nhanh (~100 phút). FT reranker xếp hạng. Lưu top-30.

Chạy: RERANKER_MODEL=models/reranker_vbpl_v2 HYBRID_SEARCH=false QDRANT_COLLECTION=vbpl_aiteam ... \
  PYTHONUTF8=1 PYTHONPATH=. python scripts/build_sidecar.py --out data/submission_v36.json
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

QFILE = Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json")
V20 = ROOT / "data/submission_v20_clean.json"


def rebuild_docs(arts):
    docs, seen = [], set()
    for a in arts:
        p = a.split("|"); doc = f"{p[0].strip()}|{p[1].strip()}" if len(p) >= 2 else p[0].strip()
        if doc not in seen:
            seen.add(doc); docs.append(doc)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/submission_v36.json")
    ap.add_argument("--sidecar", default="data/v36_cands.json")
    ap.add_argument("--topk", type=int, default=30, help="số ứng viên lưu/câu")
    ap.add_argument("--keep", type=int, default=3, help="keep top-K mặc định cho submission tạm")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from backend.rag import RAGPipeline
    from tests.build_submission_v12 import hits_to_cands
    from exp_v24 import collapse_versions

    qs = json.loads(QFILE.read_text(encoding="utf-8"))
    if args.limit:
        qs = qs[: args.limit]
    ans = {r["id"]: r.get("answer", "") for r in json.loads(V20.read_text(encoding="utf-8"))}

    rag = RAGPipeline()
    out_path = ROOT / args.out
    side_path = ROOT / args.sidecar
    out, done, side = [], set(), {}
    if out_path.exists():
        try:
            out = json.loads(out_path.read_text(encoding="utf-8")); done = {r["id"] for r in out}
            if side_path.exists():
                side = json.loads(side_path.read_text(encoding="utf-8"))
            print(f"RESUME: {len(done)} câu.", flush=True)
        except Exception:  # noqa: BLE001
            out, done, side = [], set(), {}

    t0 = time.time(); n = 0
    for q in qs:
        qid = q["id"]
        if qid in done:
            continue
        # retrieve dư rồi LỌC Điều-0 (rác không chấm được article-level) → giữ topk
        _hits = [h for h in rag.retrieve(q["question"], top_k=args.topk + 12)
                 if str((h.get("payload") or {}).get("dieu_so")) not in ("0", "None")]
        cands = hits_to_cands(_hits)[: args.topk]  # đã xếp theo adj_score
        side[str(qid)] = [{"art": c["art"], "rr": round(float(c.get("rr", 0) or 0), 4)} for c in cands]
        arts = collapse_versions([c["art"] for c in cands[: args.keep]])  # keep tạm = top-3 rank-based
        out.append({"id": qid, "question": q["question"], "answer": ans.get(qid, ""),
                    "relevant_docs": rebuild_docs(arts), "relevant_articles": arts})
        n += 1
        if n % 25 == 0:
            import torch, gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        if n % 50 == 0:
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            side_path.write_text(json.dumps(side, ensure_ascii=False), encoding="utf-8")
            r = n / (time.time() - t0)
            print(f"  [{len(out)}/{len(qs)}] {r:.2f} q/s ETA {(len(qs)-len(out))/r/60:.0f}p", flush=True)

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    side_path.write_text(json.dumps(side, ensure_ascii=False), encoding="utf-8")
    print(f"XONG {len(out)} câu ({time.time()-t0:.0f}s) | sidecar {len(side)} câu → {args.sidecar}", flush=True)


if __name__ == "__main__":
    main()
