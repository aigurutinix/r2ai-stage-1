"""Build PRECISION — single-query + FT reranker + giữ CHẶT + map. Không decomp, không judge.

Mục tiêu mới (sau khi biết leader hơn ở PRECISION 0.69 vs 0.47, recall đã sát):
chọn ÍT mà ĐÚNG. FT reranker xếp gold lên top → giữ top-2-3 là phần lớn gold.

Nhanh: dense (bỏ BM25 pickle chậm) → retrieve top-K → FT reranker đã xếp → adaptive keep
chặt → collapse_versions (map bỏ bản cũ). Resume + sample.

Chạy sample: RERANKER_MODEL=models/reranker_vbpl_v2 HYBRID_SEARCH=false QDRANT_COLLECTION=vbpl_aiteam ... \
  PYTHONUTF8=1 PYTHONPATH=. python scripts/build_precision.py --sample 40 --max-k 3
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
    ap.add_argument("--out", default="data/submission_v35_prec.json")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-k", type=int, default=3)
    ap.add_argument("--t-abs", type=float, default=0.40)
    ap.add_argument("--ratio", type=float, default=0.55)
    ap.add_argument("--min-k", type=int, default=1)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from backend.rag import RAGPipeline
    from tests.build_submission_v12 import hits_to_cands, adaptive
    from exp_v24 import collapse_versions

    qs = json.loads(QFILE.read_text(encoding="utf-8"))
    if args.limit:
        qs = qs[: args.limit]
    if args.sample:
        step = max(1, len(qs) // args.sample); qs = qs[::step][: args.sample]
    ans = {r["id"]: r.get("answer", "") for r in json.loads(V20.read_text(encoding="utf-8"))}
    v24 = {r["id"]: r["relevant_articles"]
           for r in json.loads((ROOT / "data/submission_v24_penalty.json").read_text(encoding="utf-8"))}

    def akey(a):
        p = a.split("|"); return (p[0].strip(), p[-1].strip())

    rag = RAGPipeline()
    out_path = ROOT / args.out
    out, done = [], set()
    if out_path.exists() and not args.sample:
        try:
            out = json.loads(out_path.read_text(encoding="utf-8")); done = {r["id"] for r in out}
            print(f"RESUME: {len(done)} câu → bỏ qua.", flush=True)
        except Exception:  # noqa: BLE001
            out, done = [], set()

    t0 = time.time(); n = 0
    # thống kê sample (so gold-proxy v24)
    g = hitp = npick = 0
    for q in qs:
        qid = q["id"]
        if qid in done:
            continue
        cands = hits_to_cands(rag.retrieve(q["question"], top_k=args.topk))  # đã xếp theo adj_score
        chosen = adaptive(cands, t_abs=args.t_abs, ratio=args.ratio, min_k=args.min_k, max_k=args.max_k)
        arts = collapse_versions([c["art"] for c in chosen])
        out.append({"id": qid, "question": q["question"], "answer": ans.get(qid, ""),
                    "relevant_docs": rebuild_docs(arts), "relevant_articles": arts})
        n += 1
        if args.sample:
            gold = {akey(a) for a in v24.get(qid, [])}
            sel = {akey(a) for a in arts}
            g += len(gold); hitp += len(gold & sel); npick += len(arts)
        if n % 25 == 0:
            import torch, gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        if n % 50 == 0 and not args.sample:
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [{len(out)}/{len(qs)}] {n/(time.time()-t0):.2f} q/s ETA {(len(qs)-len(out))/(n/(time.time()-t0))/60:.0f}p", flush=True)

    if not args.sample:
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    import statistics
    avg = statistics.mean(len(r["relevant_articles"]) for r in out) if out else 0
    print(f"\nXONG {len(out)} câu ({time.time()-t0:.0f}s) | TB {avg:.3f} điều/câu", flush=True)
    if args.sample and g:
        print(f"[SAMPLE so gold-proxy v24] giữ TB {npick/n:.2f} điều/câu | "
              f"recall-proxy {hitp}/{g}={hitp/g:.1%} | (precision: ít điều + đúng = tốt)", flush=True)


if __name__ == "__main__":
    main()
