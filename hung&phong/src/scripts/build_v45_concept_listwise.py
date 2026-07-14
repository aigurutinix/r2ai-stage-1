"""v45 — concept-fetch + listwise-select, nhắm vùng id 1001-2000 (câu phức tạp, v33 yếu f2~0.37).

Phát hiện binary-search: id 1001-2000 = câu tình huống phức tạp, v33 đáp tệ → mỏ vàng.
Mỗi câu target: Qwen trích CHẾ ĐỊNH → retrieve(gốc) ∪ retrieve(gốc+chế định) → Qwen LISTWISE
chọn (đọc nội dung, chộp gold vừa surface). Thay vào v33; câu ngoài range giữ nguyên v33.

Chạy: RERANKER_MODEL=AITeamVN/Vietnamese_Reranker QDRANT_COLLECTION=vbpl_aiteam ... \
  PYTHONUTF8=1 PYTHONPATH=. python scripts/build_v45_concept_listwise.py --lo 1001 --hi 1200
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

_CON_SYS = ("Bạn là chuyên gia pháp luật VN. Cho câu hỏi TÌNH HUỐNG, liệt kê 3-6 CHẾ ĐỊNH/THUẬT NGỮ "
            "pháp lý cốt lõi (vd: biện pháp bảo đảm thực hiện nghĩa vụ, hiệu lực đối kháng, tạm ngừng "
            "kinh doanh, xử phạt vi phạm hành chính). Cách nhau dấu chấm phẩy. KHÔNG giải thích.")


def rebuild_docs(arts):
    docs, seen = [], set()
    for a in arts:
        p = a.split("|"); d = f"{p[0].strip()}|{p[1].strip()}" if len(p) >= 2 else p[0].strip()
        if d not in seen:
            seen.add(d); docs.append(d)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=1001)
    ap.add_argument("--hi", type=int, default=1200)
    ap.add_argument("--out", default="data/submission_v45.json")
    args = ap.parse_args()

    from backend.rag import RAGPipeline
    from backend.llm import LLMClient
    from tests.build_submission_v12 import hits_to_cands
    from exp_v24 import collapse_versions
    from test_listwise import listwise_select

    v33 = {r["id"]: r for r in json.loads((ROOT / "data/submission_v33_v24map.json").read_text(encoding="utf-8"))}
    target = {qid for qid in v33 if args.lo <= qid <= args.hi}
    print(f"Thay {len(target)} câu (id {args.lo}-{args.hi}); giữ {len(v33)-len(target)} câu v33.", flush=True)

    rag = RAGPipeline(); llm = LLMClient()
    out = []; n = 0; t0 = time.time()
    for qid in sorted(v33):
        base = v33[qid]
        if qid not in target:
            out.append(base); continue
        q = base["question"]
        con = (llm.complete(_CON_SYS, "Câu hỏi: " + q + "\nCHẾ ĐỊNH:", think=True) or "").strip().replace("\n", " ")[:300]
        # union pool (giữ text cho listwise), rr = max
        pool = {}
        for query in (q, q + " " + con):
            for h in rag.retrieve(query, top_k=12):
                c = hits_to_cands([h])
                if not c:
                    continue
                art = c[0]["art"]; rr = c[0].get("rr") or 0
                txt = (h.get("payload") or {}).get("text") or ""
                if art not in pool or rr > pool[art]["rr"]:
                    pool[art] = {"art": art, "text": txt, "rr": rr}
        cands = sorted(pool.values(), key=lambda x: x["rr"], reverse=True)[:15]
        keep = set(collapse_versions([c["art"] for c in cands]))
        cands = [c for c in cands if c["art"] in keep]
        if not cands:
            out.append(base); n += 1; continue
        picked = listwise_select(llm, q, cands)
        arts = collapse_versions([cands[i]["art"] for i in picked if i < len(cands)]) or [cands[0]["art"]]
        nb = dict(base); nb["relevant_articles"] = arts; nb["relevant_docs"] = rebuild_docs(arts)
        out.append(nb); n += 1
        if n % 20 == 0:
            print(f"  [{n}/{len(target)}] {n/(time.time()-t0):.3f} câu/s · ETA {(len(target)-n)/(n/(time.time()-t0))/60:.0f}p", flush=True)
    out.sort(key=lambda r: r["id"])
    json.dump(out, open(ROOT / args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"XONG: thay {n} câu → {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
