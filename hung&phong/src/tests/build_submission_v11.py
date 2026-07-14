"""v11: Wide retrieval (top-20, no reranker) + LLM-judge tích hợp.

Thay vì retrieve top-8 rồi adaptive-select, v11:
  1. Retrieve top-20 dense+BM25 (KHÔNG reranker → nhanh ~0.1s/query)
  2. Với câu phân rã: top-15 gốc + top-10 mỗi câu con → merge max-score
  3. Collapse bản luật cũ → lấy top-10 còn lại
  4. LLM-judge tất cả top-10 trong 1 call Qwen → giữ CÓ (max 3)
  5. Fallback top-1 nếu tất cả bị lọc

Mục tiêu: recall tăng (gold ở vị trí 9-15 vẫn được tìm thấy) + precision giữ cao.
ETA: ~1 tiếng (không có reranker → nhanh hơn v9 nhiều).

Chạy:
  QDRANT_COLLECTION=vbpl_aiteam EMBED_BACKEND=st EMBED_ST_MODEL=AITeamVN/Vietnamese_Embedding_v2
  HYBRID_SEARCH=true USE_HNSW=true USE_RERANKER=false BM25_INDEX_PATH=data/bm25_vbpl_v2.pkl
  python -m tests.build_submission_v11 --out data/submission_v11.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("USE_TF", "0")

ROOT = Path(__file__).resolve().parents[1]
JUDGE_CACHE = ROOT / "data" / "judge_cache_v11.json"

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger("sub_v11")
logger.setLevel(logging.INFO)

# ── Collapse bản luật cũ (inline, khỏi import chéo) ──────────────────────────
_YEAR = re.compile(r"/(19|20)(\d{2})/")
_TRAILING_SO = re.compile(r"\s+số:?\s*$")
_PRIMARY = ("luật", "bộ luật", "pháp lệnh")


def _year_of(sk: str) -> int:
    m = _YEAR.search(sk or "")
    return int(m.group(1) + m.group(2)) if m else 0


def _is_primary(name: str) -> bool:
    n = (name or "").lower().strip()
    return any(n.startswith(p) for p in _PRIMARY)


def _family(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"sửa đổi,? bổ sung.*", "", s)
    s = re.sub(r"số\s.*", "", s)
    s = re.sub(r"\d{4}", "", s)
    return re.sub(r"\s+", " ", s).strip(" .,")


def _law_name(p: dict) -> str:
    loai = (p.get("loai_van_ban") or "").strip()
    title = _TRAILING_SO.sub("", (p.get("title") or "").strip())
    if loai and title.lower().startswith(loai.lower()):
        return title
    return f"{loai} {title}".strip() if loai else title


def collapse(cands: list[dict]) -> list[dict]:
    newest: dict[str, int] = {}
    for c in cands:
        name = c.get("_name", "")
        if _is_primary(name):
            fam = _family(name)
            newest[fam] = max(newest.get(fam, 0), _year_of(c.get("_sk", "")))
    out = []
    for c in cands:
        name = c.get("_name", "")
        if _is_primary(name) and _year_of(c.get("_sk", "")) < newest.get(_family(name), 0):
            continue
        out.append(c)
    return out


# ── Candidate builder ─────────────────────────────────────────────────────────
def hit_to_cand(h: dict) -> dict | None:
    p = h.get("payload", {})
    sk, ds = p.get("so_ky_hieu"), p.get("dieu_so")
    if not sk or ds is None or int(ds) <= 0:
        return None
    name = _law_name(p)
    score = h.get("adj_score") or h.get("score") or 0.0
    return {
        "art": f"{sk}|{name}|Điều {ds}",
        "doc": f"{sk}|{name}",
        "_key": f"{sk}#{ds}",
        "_sk": sk,
        "_name": name,
        "score": float(score),
    }


def merge_hits(all_hits: list[list[dict]]) -> list[dict]:
    pool: dict[str, dict] = {}
    for hits in all_hits:
        for h in hits:
            c = hit_to_cand(h)
            if not c:
                continue
            k = c["_key"]
            if k not in pool or c["score"] > pool[k]["score"]:
                pool[k] = c
    return sorted(pool.values(), key=lambda c: c["score"], reverse=True)


# ── Text lookup từ Qdrant ─────────────────────────────────────────────────────
def load_text_lookup(collection: str, client=None) -> dict[str, str]:
    print(f"Scroll {collection} lấy text...", flush=True)
    if client is None:
        from backend.qdrant_store import QdrantStore
        client = QdrantStore().client
    total = client.count(collection, exact=True).count
    lookup: dict[str, str] = {}
    offset, done = None, 0
    while True:
        pts, offset = client.scroll(collection, limit=2000, offset=offset,
                                    with_payload=True, with_vectors=False)
        for p in pts:
            pl = p.payload or {}
            sk = str(pl.get("so_ky_hieu") or "")
            ds = pl.get("dieu_so")
            if not sk or ds is None:
                continue
            key = f"{sk}|{_law_name(pl)}|Điều {ds}"
            lookup[key] = str(pl.get("text") or "")
        done += len(pts)
        if done % 40000 < 2000:
            print(f"  {done:,}/{total:,}", flush=True)
        if offset is None:
            break
    print(f"Đã nạp {len(lookup):,} điều.", flush=True)
    return lookup


# ── LLM-judge (1 call / câu hỏi) ─────────────────────────────────────────────
_J_SYSTEM = (
    "Bạn là chuyên gia pháp luật Việt Nam. Với mỗi điều luật, hãy lý luận ngắn gọn "
    "rồi xác định CÓ (trực tiếp trả lời câu hỏi) hoặc KHÔNG (liên quan gián tiếp/không liên quan)."
)
_J_USER = """Câu hỏi: {question}

Với mỗi điều dưới đây:
• Tóm tắt điều quy định gì (1 câu)
• Đối chiếu với câu hỏi → CÓ trả lời trực tiếp không?

{block}

Trả lời ĐÚNG định dạng — {n} dòng, mỗi dòng: số thứ tự, dấu hai chấm, CÓ/KHÔNG, gạch ngang, lý do ngắn:
1: CÓ — [lý do]
2: KHÔNG — [lý do]
..."""

_VERDICT_RE = re.compile(r"^\s*(\d+)\s*[:\.]\s*(CÓ|KHÔNG|CO|KHONG)", re.IGNORECASE)


def _parse_verdicts(text: str, n: int) -> list[bool]:
    v: dict[int, bool] = {}
    for line in (text or "").splitlines():
        m = _VERDICT_RE.match(line)
        if m:
            v[int(m.group(1))] = m.group(2).upper() in ("CÓ", "CO")
    return [v.get(i + 1, True) for i in range(n)]


def judge(question: str, cands: list[dict], texts: dict[str, str], llm) -> list[bool]:
    if len(cands) <= 1:
        return [True] * len(cands)
    parts = []
    for i, c in enumerate(cands, 1):
        txt = (texts.get(c["art"]) or "")[:1200]
        parts.append(f"Điều {i} — {c['art']}\n{txt}")
    block = "\n\n".join(parts)
    user = _J_USER.format(question=question, block=block, n=len(cands))
    try:
        out = llm.complete(_J_SYSTEM, user, think=True)
        return _parse_verdicts(out, len(cands))
    except Exception as e:
        logger.warning("LLM lỗi: %s", e)
        return [True] * len(cands)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/submission_v11.json")
    ap.add_argument("--orig-topk", type=int, default=20)   # top-k câu gốc
    ap.add_argument("--sub-topk", type=int, default=12)    # top-k mỗi câu con
    ap.add_argument("--judge-topk", type=int, default=10)  # số ứng viên đưa vào judge
    ap.add_argument("--max-keep", type=int, default=3)     # số điều giữ sau judge
    ap.add_argument("--limit", type=int, default=0)        # 0 = tất cả
    args = ap.parse_args()

    qs = json.loads(Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json")
                    .read_text(encoding="utf-8"))
    if args.limit:
        qs = qs[:args.limit]
    subq = json.loads((ROOT / "data/subqueries.json").read_text(encoding="utf-8"))
    v8 = {r["id"]: r for r in json.loads(
        (ROOT / "data/submission_v8.json").read_text(encoding="utf-8"))}
    cache: dict[str, list[bool]] = (
        json.loads(JUDGE_CACHE.read_text(encoding="utf-8"))
        if JUDGE_CACHE.exists() else {}
    )

    from backend.rag import RAGPipeline
    from backend.llm import LLMClient

    rag = RAGPipeline()
    llm = LLMClient()
    # Dùng chung client của RAGPipeline — Qdrant embedded chỉ cho 1 instance/process
    texts = load_text_lookup(rag.settings.qdrant_collection, client=rag.store.client)

    logger.info("wide_orig=%d sub=%d judge_top=%d max_keep=%d | subq=%d",
                args.orig_topk, args.sub_topk, args.judge_topk, args.max_keep, len(subq))

    out_path = ROOT / args.out
    out: list[dict] = []
    n_filtered = n_fallback = n_judged = 0
    t0 = time.time()

    for i, q in enumerate(qs, 1):
        qid = q["id"]
        question = q["question"]
        subs = subq.get(str(qid)) or []

        # ── 1. Wide retrieval (no reranker) ───────────────────────────────────
        all_hits: list[list[dict]] = [rag.retrieve(question, top_k=args.orig_topk)]
        for s in (subs if len(subs) >= 2 else []):
            all_hits.append(rag.retrieve(s, top_k=args.sub_topk))

        # ── 2. Merge + collapse bản cũ ────────────────────────────────────────
        merged = merge_hits(all_hits)
        clean = collapse(merged)
        top_cands = clean[:args.judge_topk]

        if not top_cands:
            out.append(v8[qid])
            continue

        # ── 3. LLM-judge (resumable cache) ───────────────────────────────────
        ckey = str(qid)
        if ckey not in cache:
            verdicts = judge(question, top_cands, texts, llm)
            cache[ckey] = verdicts
            n_judged += 1
            if n_judged % 20 == 0:
                JUDGE_CACHE.write_text(
                    json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        else:
            verdicts = cache[ckey]
            if len(verdicts) != len(top_cands):
                verdicts = [True] * len(top_cands)

        kept = [c for c, v in zip(top_cands, verdicts) if v]
        if not kept:
            kept = top_cands[:1]
            n_fallback += 1
        elif len(kept) < len(top_cands):
            n_filtered += 1

        kept = kept[:args.max_keep]

        # ── 4. Build output ───────────────────────────────────────────────────
        arts = [c["art"] for c in kept]
        docs: list[str] = []
        seen: set = set()
        for c in kept:
            if c["doc"] not in seen:
                seen.add(c["doc"]); docs.append(c["doc"])

        answer = v8.get(qid, {}).get("answer", "")
        out.append({"id": qid, "question": question,
                    "answer": answer,
                    "relevant_docs": docs, "relevant_articles": arts})

        if i % 50 == 0:
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            logger.info("[%d/%d] filtered=%d fallback=%d %.2f q/s",
                        i, len(qs), n_filtered, n_fallback,
                        i / (time.time() - t0))

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    logger.info("XONG %d câu | filtered=%d fallback=%d (%.0fs) → %s",
                len(out), n_filtered, n_fallback, time.time() - t0, out_path)


if __name__ == "__main__":
    main()
