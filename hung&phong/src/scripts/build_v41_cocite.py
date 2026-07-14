"""BƯỚC 3 — Co-citation 2-hop (recall lever). Phá trần retriever (gold co-citation bị chôn >rank40).

Hop-1: đáp án v33 (anchor confident). Xác định LỚP THIẾU:
  - có LUẬT, thiếu NĐ/TT → hop-2 tìm nghị định/thông tư hướng dẫn
  - có NĐ/TT, thiếu LUẬT → hop-2 tìm luật mẹ
Hop-2: query rewrite theo loại VB → dense retrieve sâu (topk) → rerank v3 → CỔNG rr>=thresh
  → thêm partner đúng loại, chưa có trong đáp án. (Cổng chống noise như v28 blanket fail.)

Chạy sample: RERANKER_MODEL=models/reranker_vbpl_v3 HYBRID_SEARCH=false QDRANT_COLLECTION=vbpl_aiteam ... \
  PYTHONUTF8=1 PYTHONPATH=. python scripts/build_v41_cocite.py --sample 40 --thresh 0.5
Full: bỏ --sample, thêm --out data/submission_v41.json
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


def instrument(name: str) -> str:
    import unicodedata
    n = "".join(c for c in unicodedata.normalize("NFD", name.lower()) if unicodedata.category(c) != "Mn")
    if n.startswith("bo luat") or n.startswith("luat") or n.startswith("phap lenh"):
        return "luat"
    if n.startswith("nghi dinh") or n.startswith("thong tu") or n.startswith("quyet dinh"):
        return "sub"
    return "other"


def rebuild_docs(arts):
    docs, seen = [], set()
    for a in arts:
        p = a.split("|"); d = f"{p[0].strip()}|{p[1].strip()}" if len(p) >= 2 else p[0].strip()
        if d not in seen:
            seen.add(d); docs.append(d)
    return docs


def akey(a):
    p = a.split("|"); return (p[0].strip(), p[-1].strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/submission_v41.json")
    ap.add_argument("--thresh", type=float, default=0.5, help="cổng rerank để thêm partner")
    ap.add_argument("--topk", type=int, default=40, help="độ sâu hop-2 retrieve")
    ap.add_argument("--max-add", type=int, default=1)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    from backend.rag import RAGPipeline
    from tests.build_submission_v12 import hits_to_cands

    src = json.loads((ROOT / "data/submission_v33_v24map.json").read_text(encoding="utf-8"))
    if args.sample:
        src = src[:: max(1, len(src) // args.sample)][: args.sample]
    rag = RAGPipeline()

    out = []; n_add = 0; n_q = 0; examples = []
    for r in src:
        arts = list(r["relevant_articles"]); q = r["question"]
        insts = {instrument(a.split("|")[1] if "|" in a else a) for a in arts}
        seek = None
        if "luat" in insts and "sub" not in insts:
            seek = "sub"; hint = "Nghị định, Thông tư quy định chi tiết, hướng dẫn thi hành"
        elif "sub" in insts and "luat" not in insts:
            seek = "luat"; hint = "Luật, Bộ luật, Pháp lệnh quy định về"
        if seek:
            n_q += 1
            cands = hits_to_cands(rag.retrieve(f"{hint} {q}", top_k=args.topk))
            have = {akey(a) for a in arts}
            added = []
            for c in cands:
                if len(added) >= args.max_add:
                    break
                nm = c["art"].split("|")[1] if "|" in c["art"] else c["art"]
                if instrument(nm) == seek and akey(c["art"]) not in have and (c.get("rr") or 0) >= args.thresh:
                    arts.append(c["art"]); added.append((nm, round(c["rr"], 3))); n_add += 1
            if added and len(examples) < 15:
                examples.append((q[:60], seek, added))
        out.append({"id": r["id"], "question": q, "answer": r.get("answer", ""),
                    "relevant_docs": rebuild_docs(arts), "relevant_articles": arts})

    import statistics
    print(f"\nCâu thiếu-lớp (cần hop-2): {n_q}/{len(src)} | thêm {n_add} partner (thresh {args.thresh}) | "
          f"TB điều/câu: {statistics.mean(len(r['relevant_articles']) for r in out):.3f}", flush=True)
    print("=== VÍ DỤ partner thêm (sanity: đúng lớp thiếu + liên quan?) ===")
    for q, seek, added in examples:
        print(f"Q: {q}  [tìm {seek}]")
        for nm, rr in added:
            print(f"   + ({rr}) {nm[:78]}")
    if not args.sample:
        json.dump(out, open(ROOT / args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"→ {args.out}", flush=True)


if __name__ == "__main__":
    main()
