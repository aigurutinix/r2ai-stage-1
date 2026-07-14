"""WIDE-POOL LISTWISE — thí nghiệm SẠCH: giữ pool recall-cao của v24, chỉ thay judge → listwise.

v31 confound: vừa thu hẹp retrieval (top-8/1-query) vừa thay judge. Recall chặn 0.604.
Ở đây ISOLATE tầng lọc: retrieval Y HỆT v24 (orig_topk=14 + subqueries decomp) → merge →
collapse(họ) → collapse_versions(phiên bản) → TOP-N theo rr (rộng hơn, mặc định 15) →
Qwen LISTWISE chọn subset. Pool chứa đủ gold (R tiềm năng 0.7153) → listwise (P=0.57) chọn
gold → kỳ vọng R cao + P cao = hồ sơ đội đầu.

TEST trước: --sample 60 (đọc + so v31/v24), rồi mới full.
  RERANKER_MODEL=AITeamVN/Vietnamese_Reranker QDRANT_COLLECTION=vbpl_aiteam ... \
  PYTHONUTF8=1 PYTHONPATH=. python scripts/build_listwise_wide.py --sample 60
Full:
  ... python scripts/build_listwise_wide.py --out data/submission_v32_widelistwise.json
"""
from __future__ import annotations

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
        p = a.split("|")
        doc = f"{p[0].strip()}|{p[1].strip()}" if len(p) >= 2 else p[0].strip()
        if doc not in seen:
            seen.add(doc)
            docs.append(doc)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/submission_v32_widelistwise.json")
    ap.add_argument("--orig-topk", type=int, default=14)
    ap.add_argument("--sub-topk", type=int, default=10)
    ap.add_argument("--topn", type=int, default=15, help="cắt pool đưa vào Qwen (rộng hơn v31=8)")
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--subq", default="data/subqueries.json")
    ap.add_argument("--sample", type=int, default=0, help="lấy trải đều N câu để TEST")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import re as _re
    from backend.rag import RAGPipeline
    from backend.llm import LLMClient
    from tests.build_submission_v12 import hits_to_cands, merge_cands, collapse
    from exp_v24 import collapse_versions
    from exp_cocite import KNOWN_OLD
    from test_listwise import listwise_select

    _OBSOLETE_FMT = _re.compile(r"^\d+[/-]CP$|/19\d\d/")  # format "/CP" (trước 2001) hoặc thập niên 199x

    def is_dead(art: str) -> bool:
        sk = art.split("|")[0].strip()
        return sk in KNOWN_OLD or bool(_OBSOLETE_FMT.search(sk))

    qs = json.loads(QFILE.read_text(encoding="utf-8"))
    if args.limit:
        qs = qs[: args.limit]
    if args.sample:
        step = max(1, len(qs) // args.sample)
        qs = qs[::step][: args.sample]
    subq = json.loads((ROOT / args.subq).read_text(encoding="utf-8"))
    ans = {r["id"]: r.get("answer", "") for r in json.loads(V20.read_text(encoding="utf-8"))}
    # v24 làm gold-proxy để so độ phủ pool (chỉ dùng lúc TEST)
    v24 = {r["id"]: r["relevant_articles"]
           for r in json.loads((ROOT / "data/submission_v24_penalty.json").read_text(encoding="utf-8"))}

    def akey(a):
        p = a.split("|"); return (p[0].strip(), p[-1].strip())

    rag = RAGPipeline()
    llm = LLMClient()
    out_path = ROOT / args.out
    diag_path = ROOT / "data" / (Path(args.out).stem + "_diag.json")
    out: list[dict] = []
    done: set = set()
    diag: dict = {}
    if out_path.exists() and not args.sample:
        try:
            out = json.loads(out_path.read_text(encoding="utf-8"))
            done = {r["id"] for r in out}
            if diag_path.exists():
                diag = json.loads(diag_path.read_text(encoding="utf-8"))
            print(f"RESUME: đã có {len(done)} câu → bỏ qua.", flush=True)
        except Exception:  # noqa: BLE001
            out, done, diag = [], set(), {}

    t0 = time.time()
    n_proc = 0
    # thống kê TEST
    pool_has = pool8_has = pick_has = n_g = 0  # đếm gold-proxy (v24) phủ pool15 / pool8 / được chọn
    lines = []
    for q in qs:
        qid, question = q["id"], q["question"]
        if qid in done:
            continue
        lists = [hits_to_cands(rag.retrieve(question, top_k=args.orig_topk))]
        subs = subq.get(str(qid)) or []
        if len(subs) >= 2:
            for s in subs:
                lists.append(hits_to_cands(rag.retrieve(s, top_k=args.sub_topk)))
        merged = collapse(merge_cands(lists))
        # gộp phiên bản (họ) + LỌC bản cũ chết (KNOWN_OLD/199x/"/CP") TRƯỚC Qwen → tránh Qwen
        # với tay vào bản cũ khi ép completeness. Recall-safe: chỉ bỏ văn bản CHẮC CHẮN hết hiệu lực.
        keep = set(collapse_versions([c["art"] for c in merged]))
        merged = [c for c in merged if c["art"] in keep and not is_dead(c["art"])]
        merged.sort(key=lambda c: c["rr"], reverse=True)
        pool = merged[: args.topn]
        for c in pool:
            c["text"] = c.get("text", "")
        if not pool:
            arts = []
            picked = []
        else:
            picked = listwise_select(llm, question, pool, max_k=args.max_k)
            arts = collapse_versions([pool[i]["art"] for i in picked])

        # đo độ phủ gold-proxy (v24)
        gold = {akey(a) for a in v24.get(qid, [])}
        if gold:
            pkeys15 = {akey(c["art"]) for c in pool}
            pkeys8 = {akey(c["art"]) for c in merged[:8]}
            sel = {akey(a) for a in arts}
            pool_has += len(gold & pkeys15); pool8_has += len(gold & pkeys8)
            pick_has += len(gold & sel); n_g += len(gold)

        diag[str(qid)] = {"cands": [{"art": c["art"], "rr": round(float(c.get("rr", 0) or 0), 4)} for c in pool],
                          "picked": picked}
        out.append({"id": qid, "question": question, "answer": ans.get(qid, ""),
                    "relevant_docs": rebuild_docs(arts), "relevant_articles": arts})
        n_proc += 1
        if args.sample:
            lines.append(f"id {qid} [{len(arts)}đ, pool {len(pool)}]: " +
                         " | ".join(a.split("|")[0].strip() + " " + a.split("|")[-1].replace("Điều ", "Đ") for a in arts))
        if n_proc % 10 == 0:
            import torch, gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        if n_proc % 30 == 0 and not args.sample:
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            diag_path.write_text(json.dumps(diag, ensure_ascii=False), encoding="utf-8")
            rate = n_proc / (time.time() - t0)
            print(f"  +{n_proc} ({len(out)}/{len(qs)}) · {rate:.2f} q/s · ETA {(len(qs)-len(out))/rate/60:.0f}p", flush=True)

    if not args.sample:
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    diag_path.write_text(json.dumps(diag, ensure_ascii=False), encoding="utf-8")  # luôn lưu diag (cả sample)
    import statistics
    avg = statistics.mean(len(r["relevant_articles"]) for r in out) if out else 0
    print(f"\nXONG {len(out)} câu ({time.time()-t0:.0f}s) | TB {avg:.3f} điều/câu", flush=True)
    if n_g:
        print(f"[ĐỘ PHỦ gold-proxy v24] pool15 chứa {pool_has}/{n_g}={pool_has/n_g:.1%} · "
              f"pool8 {pool8_has/n_g:.1%} · listwise CHỌN {pick_has}/{n_g}={pick_has/n_g:.1%}", flush=True)
    if args.sample:
        (ROOT / "data/diag/listwise_wide_test.txt").write_text("\n".join(lines), encoding="utf-8")
        print("→ data/diag/listwise_wide_test.txt", flush=True)


if __name__ == "__main__":
    main()
