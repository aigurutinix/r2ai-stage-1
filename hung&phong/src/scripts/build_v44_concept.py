"""v44 — controlled-perturbation: thay CHỈ nhóm câu retrieval-trượt trong v33 bằng concept-augmented.

Ý tưởng (user): v33=0.603 đã biết. Thay đáp án nhóm câu max-rr thấp (semantic-gap) bằng
retrieve(query + CHẾ ĐỊNH do Qwen trích) → nộp → delta điểm = tác động THẬT của concept-extraction.
Giữ nguyên các câu khác → cô lập sạch.

Chạy: RERANKER_MODEL=AITeamVN/Vietnamese_Reranker QDRANT_COLLECTION=vbpl_aiteam ... \
  PYTHONUTF8=1 PYTHONPATH=. python scripts/build_v44_concept.py --max-rr 0.2 --keep 3
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

_SYS = ("Bạn là chuyên gia pháp luật VN. Cho câu hỏi TÌNH HUỐNG, liệt kê 3-6 CHẾ ĐỊNH/THUẬT NGỮ "
        "pháp lý cốt lõi mà tình huống thuộc về (vd: biện pháp bảo đảm thực hiện nghĩa vụ, hiệu lực "
        "đối kháng, tạm ngừng kinh doanh, xử phạt vi phạm hành chính). Cách nhau dấu chấm phẩy. "
        "KHÔNG giải thích, KHÔNG câu dẫn.")


def rebuild_docs(arts):
    docs, seen = [], set()
    for a in arts:
        p = a.split("|"); d = f"{p[0].strip()}|{p[1].strip()}" if len(p) >= 2 else p[0].strip()
        if d not in seen:
            seen.add(d); docs.append(d)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rr", type=float, default=0.2, help="ngưỡng max-rr để coi là retrieval-trượt")
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default="data/submission_v44_concept.json")
    args = ap.parse_args()

    from backend.rag import RAGPipeline
    from backend.llm import LLMClient
    from tests.build_submission_v12 import hits_to_cands
    from exp_v24 import collapse_versions

    v33 = {r["id"]: r for r in json.loads((ROOT / "data/submission_v33_v24map.json").read_text(encoding="utf-8"))}
    side = json.loads((ROOT / "data/v37_cands.json").read_text(encoding="utf-8"))
    affected = {int(qid) for qid, v in side.items() if v and max((x["rr"] or 0) for x in v) < args.max_rr}
    print(f"Thay {len(affected)} câu (max-rr<{args.max_rr}); giữ nguyên {len(v33)-len(affected)} câu v33.", flush=True)

    rag = RAGPipeline(); llm = LLMClient()
    out = []; n_chg = 0; t0 = time.time()
    for qid in sorted(v33):
        base = v33[qid]
        if qid not in affected:
            out.append(base); continue
        q = base["question"]
        con = (llm.complete(_SYS, "Câu hỏi: " + q + "\nCHẾ ĐỊNH:", think=True) or "").strip().replace("\n", " ")[:300]
        # union: pool gốc + pool (gốc+chế định), rr = max
        pool = {}
        for query in (q, q + " " + con):
            for c in hits_to_cands(rag.retrieve(query, top_k=args.topk)):
                a = c["art"]; rr = c.get("rr") or 0
                if a not in pool or rr > pool[a]:
                    pool[a] = rr
        ranked = sorted(pool, key=lambda a: pool[a], reverse=True)
        arts = collapse_versions(ranked[: args.keep])
        nb = dict(base); nb["relevant_articles"] = arts; nb["relevant_docs"] = rebuild_docs(arts)
        out.append(nb); n_chg += 1
        if n_chg % 25 == 0:
            print(f"  [{n_chg}/{len(affected)}] {n_chg/(time.time()-t0):.2f} câu/s", flush=True)
    out.sort(key=lambda r: r["id"])
    json.dump(out, open(ROOT / args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"XONG: thay {n_chg} câu → {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
