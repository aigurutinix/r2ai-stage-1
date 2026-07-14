"""v46 — concept-ADDITIVE cho id 1001-2000 (câu phức tạp v33 yếu f2~0.37).

Giữ NGUYÊN đáp án v33 (phần nó đúng) + THÊM điều mà concept-rewrite surface (lớp thiếu:
luật cốt lõi / NĐ hướng dẫn). Recall-lean (F2 thưởng recall gấp đôi). Hợp năng lực Qwen
(KHÔNG listwise — Qwen under-pick). concept = rewriting chống loạn domain.

Chạy: RERANKER_MODEL=AITeamVN/Vietnamese_Reranker QDRANT_COLLECTION=vbpl_aiteam ... \
  PYTHONUTF8=1 PYTHONPATH=. python scripts/build_v46_additive.py --lo 1001 --hi 2000 --add 2
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
            "pháp lý cốt lõi (vd: biện pháp bảo đảm thực hiện nghĩa vụ, hiệu lực đối kháng, bảo vệ dữ "
            "liệu cá nhân, hợp đồng vận chuyển tài sản, nhượng quyền thương mại). Cách nhau dấu chấm "
            "phẩy. KHÔNG giải thích.")


def instrument(name: str) -> str:
    import unicodedata
    n = "".join(c for c in unicodedata.normalize("NFD", name.lower()) if unicodedata.category(c) != "Mn")
    if n.startswith("bo luat") or n.startswith("luat") or n.startswith("phap lenh"):
        return "luat"
    if n.startswith("nghi dinh") or n.startswith("thong tu") or n.startswith("quyet dinh"):
        return "sub"
    return "other"


def akey(a):
    p = a.split("|"); return (p[0].strip(), p[-1].strip())


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
    ap.add_argument("--hi", type=int, default=2000)
    ap.add_argument("--add", type=int, default=2, help="số điều thêm tối đa/câu")
    ap.add_argument("--rr", type=float, default=0.03, help="sàn rr tối thiểu để thêm")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--out", default="data/submission_v46.json")
    args = ap.parse_args()

    from backend.rag import RAGPipeline
    from backend.llm import LLMClient
    from tests.build_submission_v12 import hits_to_cands
    from exp_v24 import collapse_versions

    v33 = {r["id"]: r for r in json.loads((ROOT / "data/submission_v33_v24map.json").read_text(encoding="utf-8"))}
    target = {qid for qid in v33 if args.lo <= qid <= args.hi}
    print(f"Concept-additive {len(target)} câu (id {args.lo}-{args.hi}); giữ {len(v33)-len(target)} v33.", flush=True)

    rag = RAGPipeline(); llm = LLMClient()
    out_path = ROOT / args.out
    done = {}
    if out_path.exists():
        try:
            done = {r["id"]: r for r in json.loads(out_path.read_text(encoding="utf-8"))
                    if r["id"] in target and len(r["relevant_articles"]) > len(v33[r["id"]]["relevant_articles"])}
            print(f"RESUME: {len(done)} câu target đã xong → bỏ qua.", flush=True)
        except Exception:  # noqa: BLE001
            done = {}
    out = []; n = 0; n_add = 0; t0 = time.time()
    for qid in sorted(v33):
        base = v33[qid]
        if qid not in target:
            out.append(base); continue
        if qid in done:                       # đã xử lý ở lần trước
            out.append(done[qid]); n += 1; continue
        q = base["question"]; cur = list(base["relevant_articles"])
        try:
            con = (llm.complete(_CON_SYS, "Câu hỏi: " + q + "\nCHẾ ĐỊNH:", think=True) or "").strip().replace("\n", " ")[:250]
        except Exception as e:  # noqa: BLE001 — Qwen think timeout → bỏ concept câu này, giữ v33
            print(f"  [skip concept id {qid}] {repr(e)[:50]}", flush=True); con = ""
        cands = hits_to_cands(rag.retrieve(q + " " + con, top_k=args.topk))
        have = {akey(a) for a in cur}
        cur_inst = {instrument(a.split("|")[1] if "|" in a else a) for a in cur}
        # ưu tiên lớp THIẾU: nếu có luật mà thiếu NĐ → thêm sub; có sub thiếu luật → thêm luật
        want = None
        if "luat" in cur_inst and "sub" not in cur_inst:
            want = "sub"
        elif "sub" in cur_inst and "luat" not in cur_inst:
            want = "luat"
        added = []
        for pref in ([want] if want else []) + ["luat", "sub"]:
            for c in cands:
                if len(added) >= args.add:
                    break
                nm = c["art"].split("|")[1] if "|" in c["art"] else c["art"]
                if (instrument(nm) == pref and akey(c["art"]) not in have
                        and (c.get("rr") or 0) >= args.rr):
                    cur.append(c["art"]); have.add(akey(c["art"])); added.append(c["art"]); n_add += 1
            if len(added) >= args.add:
                break
        arts = collapse_versions(cur)
        nb = dict(base); nb["relevant_articles"] = arts; nb["relevant_docs"] = rebuild_docs(arts)
        out.append(nb); n += 1
        if n % 50 == 0:
            # ghi incremental (out đang theo thứ tự id) → resume được nếu crash
            tmp = out + [v33[i] for i in sorted(v33) if i > qid]   # phần chưa xử lý = v33 gốc
            out_path.write_text(json.dumps(sorted(tmp, key=lambda r: r["id"]), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [{n}/{len(target)}] {n/(time.time()-t0):.2f} câu/s · thêm TB {n_add/n:.2f}/câu", flush=True)
    out.sort(key=lambda r: r["id"])
    json.dump(out, open(ROOT / args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    import statistics
    print(f"XONG: {n} câu, thêm {n_add} điều (TB {n_add/max(n,1):.2f}/câu) → {args.out} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  TB điều/câu vùng target: {statistics.mean(len(r['relevant_articles']) for r in out if args.lo<=r['id']<=args.hi):.2f}", flush=True)


if __name__ == "__main__":
    main()
