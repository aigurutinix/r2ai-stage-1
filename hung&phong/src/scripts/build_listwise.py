"""BUILD full 2000 câu — Qwen LISTWISE reranker (tầng 3) + collapse_versions.

Cascade: dense(AITeamVN) + BM25 → AITeamVN/Vietnamese_Reranker (top-N) → Qwen listwise chọn
subset → collapse_versions (vá lỗi ưu-tiên-bản-cũ phát hiện ở test 50 câu).
RESUME được (ghi mỗi 30 câu; kill+restart bỏ qua câu đã xong → chống degrade run dài).

Chạy: RERANKER_MODEL=AITeamVN/Vietnamese_Reranker QDRANT_COLLECTION=vbpl_aiteam ... \
  PYTHONUTF8=1 PYTHONPATH=. python scripts/build_listwise.py --out data/submission_v31_listwise.json
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
    ap.add_argument("--out", default="data/submission_v31_listwise.json")
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--max-k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from backend.rag import RAGPipeline
    from backend.llm import LLMClient
    from tests.build_submission_v12 import hits_to_cands
    from test_listwise import listwise_select
    from exp_v24 import collapse_versions

    qs = json.loads(QFILE.read_text(encoding="utf-8"))
    if args.limit:
        qs = qs[: args.limit]
    ans = {r["id"]: r.get("answer", "") for r in json.loads(V20.read_text(encoding="utf-8"))}

    rag = RAGPipeline()
    llm = LLMClient()
    out_path = ROOT / args.out
    out: list[dict] = []
    done: set = set()
    if out_path.exists():
        try:
            out = json.loads(out_path.read_text(encoding="utf-8"))
            done = {r["id"] for r in out}
            print(f"RESUME: đã có {len(done)} câu → bỏ qua, chạy tiếp.", flush=True)
        except Exception:  # noqa: BLE001
            out, done = [], set()

    # SIDECAR: lưu ứng viên (đã xếp hạng reranker, sau collapse) + index Qwen chọn,
    # để dựng OFFLINE các biến thể (thuần / +sàn recall) mà KHÔNG chạy lại Qwen.
    diag_path = ROOT / (Path(args.out).stem + "_diag.json")
    diag: dict = {}
    if diag_path.exists():
        try:
            diag = json.loads(diag_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            diag = {}

    t0 = time.time()
    n_proc = 0
    for q in qs:
        qid, question = q["id"], q["question"]
        if qid in done:
            continue
        hits = rag.retrieve(question, top_k=args.topn)
        cands = []
        for h in hits:
            c = hits_to_cands([h])
            if c:
                c[0]["text"] = h.get("payload", {}).get("text", "")
                cands.append(c[0])
        # GỘP PHIÊN BẢN ở tầng ỨNG VIÊN (deterministic) — KHỚP test_listwise:
        # bỏ bản cũ TRƯỚC khi Qwen thấy → Qwen không chọn được bản cũ + bớt nhiễu
        # → giúp Qwen chọn đúng bản hiện hành (đã kiểm chứng id281 ở test 50 câu).
        keep = set(collapse_versions([c["art"] for c in cands]))
        cands = [c for c in cands if c["art"] in keep]
        if cands:
            picked = listwise_select(llm, question, cands, max_k=args.max_k)
            arts = [cands[i]["art"] for i in picked]
        else:
            picked = []
            arts = []
        diag[str(qid)] = {
            "cands": [{"art": c["art"], "rr": round(float(c.get("rr", 0) or 0), 4)} for c in cands],
            "picked": picked,
        }
        out.append({"id": qid, "question": question, "answer": ans.get(qid, ""),
                    "relevant_docs": rebuild_docs(arts), "relevant_articles": arts})
        n_proc += 1
        if n_proc % 10 == 0:
            import torch, gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        if n_proc % 30 == 0:
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            diag_path.write_text(json.dumps(diag, ensure_ascii=False), encoding="utf-8")
            rate = n_proc / (time.time() - t0)
            print(f"  +{n_proc} (tổng {len(out)}/{len(qs)}) · {rate:.2f} q/s · ETA {(len(qs)-len(out))/rate/60:.0f}p", flush=True)

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    diag_path.write_text(json.dumps(diag, ensure_ascii=False), encoding="utf-8")
    import statistics
    print(f"XONG {len(out)} câu ({time.time()-t0:.0f}s) | TB điều/câu {statistics.mean(len(r['relevant_articles']) for r in out):.3f} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
