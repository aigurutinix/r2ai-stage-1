"""v25 — BẢN DỰNG LẠI HOÀN CHỈNH TỪ ĐẦU (không phải sửa v20).

Full pipeline / câu, tích hợp các đòn non-finetune đã kiểm chứng:
  1. retrieve câu gốc (rộng, orig_topk)
  2. + decomposition NÔNG (subqueries.json) — như v20
  3. + penalty-vế: câu có vế chế tài → retrieve câu gốc, lọc WHITELIST NĐ xử phạt
  4. merge → collapse (họ luật) → collapse_versions (bỏ bản cũ khi có bản mới)
  5. adaptive select (max_k)
  6. judge keep-top-2: giữ điều nếu rank<=keep_top HOẶC judge=CÓ (bảo vệ recall top-rank)

Chạy QUA ĐÊM (judge LLM ~ vài giờ, resumable qua cache + ghi output mỗi 50 câu):
  QDRANT_COLLECTION=vbpl_aiteam QDRANT_URL=http://localhost:6333 EMBED_BACKEND=st \
  EMBED_ST_MODEL=AITeamVN/Vietnamese_Embedding_v2 HYBRID_SEARCH=true USE_HNSW=false \
  USE_RERANKER=true BM25_INDEX_PATH=data/bm25_vbpl_aiteam.pkl PYTHONUTF8=1 PYTHONPATH=. \
  python scripts/build_full_v25.py --out data/submission_v25_full.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tests.build_submission_v12 import (  # noqa: E402
    collapse, adaptive, merge_cands, hits_to_cands, judge, load_text_lookup,
)
from exp_v24 import collapse_versions, PENALTY_WHITELIST, _PENALTY  # noqa: E402

QFILE = Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json")
JUDGE_CACHE = ROOT / "data/judge_cache_v25.json"
V20 = ROOT / "data/submission_v20_clean.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/submission_v25_full.json")
    ap.add_argument("--orig-topk", type=int, default=14)
    ap.add_argument("--sub-topk", type=int, default=10)
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--keep-top", type=int, default=2)
    ap.add_argument("--pen-add", type=int, default=2)
    ap.add_argument("--subq", default="data/subqueries.json")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    qs = json.loads(QFILE.read_text(encoding="utf-8"))
    if args.limit:
        qs = qs[: args.limit]
    subq = json.loads((ROOT / args.subq).read_text(encoding="utf-8"))
    ans = {r["id"]: r.get("answer", "") for r in json.loads(V20.read_text(encoding="utf-8"))}

    from backend.rag import RAGPipeline
    rag = RAGPipeline()
    llm = texts = None
    cache: dict[str, list[bool]] = {}
    if not args.no_judge:
        from backend.llm import LLMClient
        llm = LLMClient()
        texts = load_text_lookup(rag.settings.qdrant_collection, client=rag.store.client)
        if JUDGE_CACHE.exists():
            cache = json.loads(JUDGE_CACHE.read_text(encoding="utf-8"))

    out_path = ROOT / args.out
    out: list[dict] = []
    done: set = set()
    if out_path.exists():
        try:
            out = json.loads(out_path.read_text(encoding="utf-8"))
            done = {r["id"] for r in out}
            print(f"RESUME: đã có {len(done)} câu trong {args.out} → bỏ qua, chạy tiếp.", flush=True)
        except Exception:  # noqa: BLE001
            out, done = [], set()
    t0 = time.time()
    for i, q in enumerate(qs, 1):
        qid, question = q["id"], q["question"]
        if qid in done:
            continue
        subs = subq.get(str(qid)) or []

        lists = [hits_to_cands(rag.retrieve(question, top_k=args.orig_topk))]
        if len(subs) >= 2:
            for s in subs:
                lists.append(hits_to_cands(rag.retrieve(s, top_k=args.sub_topk)))
        merged = collapse(merge_cands(lists))
        max_k = min(len(subs) + 1, args.max_k) if len(subs) >= 2 else args.max_k
        chosen = adaptive(merged, t_abs=0.40, ratio=0.80, min_k=1, max_k=max_k)

        # penalty-vế: thêm NĐ xử phạt (whitelist) câu chế tài còn thiếu
        if _PENALTY.search(question):
            have = {c["_key"] for c in chosen}
            added = 0
            for c in hits_to_cands(rag.retrieve(question, top_k=30)):
                if added >= args.pen_add:
                    break
                if c["art"].split("|")[0].strip() not in PENALTY_WHITELIST:
                    continue
                if c["_key"] in have:
                    continue
                have.add(c["_key"])
                chosen.append(c)
                added += 1

        # bỏ bản cũ khi có bản mới cùng họ
        keep_arts = set(collapse_versions([c["art"] for c in chosen]))
        chosen = [c for c in chosen if c["art"] in keep_arts]
        chosen.sort(key=lambda c: c["rr"], reverse=True)

        # judge keep-top: giữ rank < keep_top HOẶC judge=CÓ
        if not args.no_judge and len(chosen) >= 2:
            ckey = str(qid)
            if ckey in cache and len(cache[ckey]) == len(chosen):
                verdicts = cache[ckey]
            else:
                verdicts = judge(question, chosen, texts, llm)
                cache[ckey] = verdicts
                if i % 20 == 0:
                    JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            kept = [c for idx, (c, v) in enumerate(zip(chosen, verdicts)) if idx < args.keep_top or v]
        else:
            kept = chosen

        docs, seen = [], set()
        for c in kept:
            if c["doc"] not in seen:
                seen.add(c["doc"])
                docs.append(c["doc"])
        out.append({"id": qid, "question": question, "answer": ans.get(qid, ""),
                    "relevant_docs": docs, "relevant_articles": [c["art"] for c in kept]})

        # Dọn VRAM định kỳ — chống phân mảnh/OOM khi embed+reranker+Qwen cùng cư trú (12GB sát trần)
        if i % 25 == 0:
            import torch, gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        if i % 50 == 0:
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{i}/{len(qs)}] {i/(time.time()-t0):.2f} q/s · ETA {(len(qs)-i)/(i/(time.time()-t0))/60:.0f}p", flush=True)

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if cache:
        JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    import statistics
    print(f"XONG {len(out)} câu ({time.time()-t0:.0f}s) | TB điều/câu {statistics.mean(len(r['relevant_articles']) for r in out):.3f} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
