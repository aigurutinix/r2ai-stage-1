"""Full pipeline for the second 1000 questions, with v47 + v49 filters inline.

Default target is id 1001-2000 from C:/Users/PHONG/Downloads/R2AIStage1DATA.json.
The flow is:
  retrieve + subquery + adaptive + optional judge
  penalty whitelist
  concept-additive
  version collapse
  v47 domain/version filter
  v49 known-old blacklist
  optional zip packing

Example:
  PYTHONUTF8=1 PYTHONPATH=. python scripts/build_full_v50_v47v49.py \
    --lo 1001 --hi 2000 --out data/submission_v50_1001_2000.json --zip
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("QDRANT_COLLECTION", "vbpl_aiteam")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("EMBED_BACKEND", "st")
os.environ.setdefault("EMBED_ST_MODEL", "AITeamVN/Vietnamese_Embedding_v2")
os.environ.setdefault("HYBRID_SEARCH", "true")
os.environ.setdefault("USE_RERANKER", "true")
os.environ.setdefault("RERANKER_MODEL", "AITeamVN/Vietnamese_Reranker")
os.environ.setdefault("BM25_INDEX_PATH", "data/bm25_vbpl_aiteam.pkl")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_v39_domain import is_offdomain, rebuild_docs  # noqa: E402
from build_v47_filter import version_drop  # noqa: E402
from build_v49_blacklist import KNOWN_OLD, norm_sk  # noqa: E402
from exp_v24 import PENALTY_WHITELIST, _PENALTY, collapse_versions  # noqa: E402
from tests.build_submission_v12 import (  # noqa: E402
    adaptive,
    collapse,
    hits_to_cands,
    judge,
    load_text_lookup,
    merge_cands,
)

QFILE = Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json")
DEFAULT_ANS = ROOT / "data/submission_v33_v24map.json"
# Prompt judge đã đổi; dùng cache mới để không tái sử dụng verdict từ prompt cũ.
JUDGE_CACHE = ROOT / "data/judge_cache_v61_prompt.json"

CONCEPT_SYSTEM = """
Bạn tạo cụm truy hồi cho hệ thống tìm điều luật, KHÔNG trả lời câu hỏi.

Yêu cầu:
1. Liệt kê 3-6 cụm pháp lý ngắn, mỗi cụm là một chế định, hành vi, thủ tục, đối tượng, hậu quả pháp lý hoặc thuật ngữ lõi trong câu hỏi.
2. Giữ nguyên các từ khóa đặc thù trong câu hỏi: tên thủ tục, loại giấy phép, hành vi vi phạm, chủ thể, tài sản, lĩnh vực.
3. Có thể thêm thuật ngữ pháp lý tương đương phổ biến nếu chắc chắn, ví dụ "xử phạt vi phạm hành chính", "biện pháp khắc phục hậu quả", "thu hồi", "đăng ký", "cấp phép".
4. Không bịa số điều, số văn bản, tên cơ quan, mốc thời gian hoặc lĩnh vực mới không có trong câu hỏi.
5. Nếu câu hỏi có nhiều vế, tạo cụm cho từng vế quan trọng; không biến câu hỏi thành câu trả lời.

Đầu ra: một dòng duy nhất, các cụm cách nhau bằng dấu chấm phẩy, không giải thích.
""".strip()


def art_key(art: str) -> tuple[str, str]:
    p = art.split("|")
    return p[0].strip(), (p[-1].strip() if p else "")


def layer_type(name: str) -> str:
    import unicodedata

    n = "".join(c for c in unicodedata.normalize("NFD", name.lower()) if unicodedata.category(c) != "Mn")
    if n.startswith("bo luat") or n.startswith("luat") or n.startswith("phap lenh"):
        return "luat"
    if n.startswith("nghi dinh") or n.startswith("thong tu") or n.startswith("quyet dinh"):
        return "sub"
    return "other"


def pack_zip(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("results.json", json.dumps(data, ensure_ascii=False))
    with zipfile.ZipFile(dst) as z:
        names = z.namelist()
    assert names == ["results.json"], f"bad zip layout: {names}"
    print(f"PACK OK -> {dst} | {len(data)} rows", flush=True)


def load_answer_map(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return {int(r["id"]): r.get("answer", "") for r in rows if "id" in r}


def apply_v47_v49(question: str, arts: list[str], stats: dict[str, int]) -> list[str]:
    if not arts:
        return arts

    kept: list[str] = []
    dropped = 0
    for art in arts:
        if is_offdomain(art, question) and (len(arts) - dropped) > 1:
            dropped += 1
            stats["v47_domain"] += 1
            continue
        kept.append(art)
    if not kept:
        kept = arts[:1]

    before = len(kept)
    kept = version_drop(kept)
    stats["v47_version"] += before - len(kept)
    if not kept:
        kept = arts[:1]

    out: list[str] = []
    dropped_old = 0
    for art in kept:
        sk = norm_sk(art.split("|")[0])
        if sk in KNOWN_OLD and (len(kept) - dropped_old) > 1:
            dropped_old += 1
            stats["v49_blacklist"] += 1
            continue
        out.append(art)
    return out or kept[:1]


def add_penalty(question: str, chosen: list[dict], rag, top_k: int, max_add: int) -> int:
    if not _PENALTY.search(question):
        return 0
    have = {c["_key"] for c in chosen}
    added = 0
    for c in hits_to_cands(rag.retrieve(question, top_k=top_k)):
        if added >= max_add:
            break
        if c["art"].split("|")[0].strip() not in PENALTY_WHITELIST:
            continue
        if c["_key"] in have:
            continue
        have.add(c["_key"])
        chosen.append(c)
        added += 1
    return added


def add_concept(question: str, chosen: list[dict], rag, llm, top_k: int, max_add: int, min_rr: float) -> int:
    if max_add <= 0:
        return 0
    try:
        concept = (llm.complete(CONCEPT_SYSTEM, "Câu hỏi: " + question + "\nCụm truy hồi:", think=True) or "")
    except Exception as exc:  # noqa: BLE001
        print(f"  [concept skip] {type(exc).__name__}: {str(exc)[:80]}", flush=True)
        return 0
    concept = concept.strip().replace("\n", " ")[:250]
    if not concept:
        return 0

    have = {c["_key"] for c in chosen}
    cur_layers = {layer_type(c["art"].split("|")[1] if "|" in c["art"] else c["art"]) for c in chosen}
    want = None
    if "luat" in cur_layers and "sub" not in cur_layers:
        want = "sub"
    elif "sub" in cur_layers and "luat" not in cur_layers:
        want = "luat"

    cands = hits_to_cands(rag.retrieve(question + " " + concept, top_k=top_k))
    added = 0
    prefs = ([want] if want else []) + ["luat", "sub"]
    for pref in prefs:
        for c in cands:
            if added >= max_add:
                return added
            name = c["art"].split("|")[1] if "|" in c["art"] else c["art"]
            if layer_type(name) != pref:
                continue
            if c["_key"] in have:
                continue
            if float(c.get("rr") or 0.0) < min_rr:
                continue
            have.add(c["_key"])
            chosen.append(c)
            added += 1
    return added


def checkpoint(out_path: Path, rows: list[dict]) -> None:
    rows.sort(key=lambda r: int(r["id"]))
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qfile", default=str(QFILE))
    ap.add_argument("--lo", type=int, default=1001)
    ap.add_argument("--hi", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/submission_v50_1001_2000.json")
    ap.add_argument("--answer-source", default=str(DEFAULT_ANS))
    ap.add_argument("--judge-cache", default=str(JUDGE_CACHE))
    ap.add_argument("--subq", default="data/subqueries.json")
    ap.add_argument("--orig-topk", type=int, default=14)
    ap.add_argument("--sub-topk", type=int, default=10)
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--keep-top", type=int, default=2)
    ap.add_argument("--pen-topk", type=int, default=30)
    ap.add_argument("--pen-add", type=int, default=2)
    ap.add_argument("--concept-topk", type=int, default=12)
    ap.add_argument("--concept-add", type=int, default=2)
    ap.add_argument("--concept-rr", type=float, default=0.03)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--no-concept", action="store_true")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    qfile = Path(args.qfile)
    questions = json.loads(qfile.read_text(encoding="utf-8"))
    questions = [q for q in questions if args.lo <= int(q["id"]) <= args.hi]
    if args.limit:
        questions = questions[: args.limit]
    qids = {int(q["id"]) for q in questions}
    print(f"Target: {len(questions)} questions from {qfile} | id {args.lo}-{args.hi}", flush=True)
    print(
        "Config: "
        f"LLM_MODEL={os.environ.get('LLM_MODEL', '(from .env)')} | "
        f"QDRANT_COLLECTION={os.environ.get('QDRANT_COLLECTION')} | "
        f"EMBED_BACKEND={os.environ.get('EMBED_BACKEND')} | "
        f"EMBED_ST_MODEL={os.environ.get('EMBED_ST_MODEL')} | "
        f"HYBRID_SEARCH={os.environ.get('HYBRID_SEARCH')} | "
        f"USE_RERANKER={os.environ.get('USE_RERANKER')} | "
        f"RERANKER_MODEL={os.environ.get('RERANKER_MODEL')} | "
        f"BM25_INDEX_PATH={os.environ.get('BM25_INDEX_PATH')}",
        flush=True,
    )

    subq = json.loads((ROOT / args.subq).read_text(encoding="utf-8"))
    answers = load_answer_map(Path(args.answer_source))

    from backend.rag import RAGPipeline
    from backend.llm import LLMClient

    rag = RAGPipeline()
    llm = LLMClient()
    texts = None
    judge_cache = Path(args.judge_cache)
    cache: dict[str, list[bool]] = {}
    if not args.no_judge:
        texts = load_text_lookup(rag.settings.qdrant_collection, client=rag.store.client)
        if judge_cache.exists():
            cache = json.loads(judge_cache.read_text(encoding="utf-8"))

    out_path = ROOT / args.out
    rows: list[dict] = []
    done: set[int] = set()
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            rows = [r for r in existing if int(r["id"]) in qids]
            done = {int(r["id"]) for r in rows}
            print(f"RESUME: {len(done)} existing rows in {args.out}", flush=True)
        except Exception:  # noqa: BLE001
            rows, done = [], set()

    stats = {"penalty": 0, "concept": 0, "v47_domain": 0, "v47_version": 0, "v49_blacklist": 0}
    t0 = time.time()
    processed = len(done)

    for q in questions:
        qid = int(q["id"])
        if qid in done:
            continue
        q_t0 = time.time()
        question = q["question"]
        subs = subq.get(str(qid)) or []
        print(f"Q {processed + 1}/{len(questions)} id={qid} start | subq={len(subs)}", flush=True)

        lists = [hits_to_cands(rag.retrieve(question, top_k=args.orig_topk))]
        if len(subs) >= 2:
            for s in subs:
                lists.append(hits_to_cands(rag.retrieve(s, top_k=args.sub_topk)))
        print(f"  id={qid} retrieved lists={len(lists)} cands={sum(len(x) for x in lists)}", flush=True)

        merged = collapse(merge_cands(lists))
        max_k = min(len(subs) + 1, args.max_k) if len(subs) >= 2 else args.max_k
        chosen = adaptive(merged, t_abs=0.40, ratio=0.80, min_k=1, max_k=max_k)
        print(f"  id={qid} selected={len(chosen)} from merged={len(merged)}", flush=True)

        stats["penalty"] += add_penalty(question, chosen, rag, args.pen_topk, args.pen_add)

        keep_arts = set(collapse_versions([c["art"] for c in chosen]))
        chosen = [c for c in chosen if c["art"] in keep_arts]
        chosen.sort(key=lambda c: float(c.get("rr") or 0.0), reverse=True)

        if not args.no_judge and len(chosen) >= 2:
            print(f"  id={qid} judge start n={len(chosen)}", flush=True)
            ckey = str(qid)
            if ckey in cache and len(cache[ckey]) == len(chosen):
                verdicts = cache[ckey]
            else:
                verdicts = judge(question, chosen, texts or {}, llm)
                cache[ckey] = verdicts
            chosen = [c for idx, (c, v) in enumerate(zip(chosen, verdicts)) if idx < args.keep_top or v]
            print(f"  id={qid} judge done kept={len(chosen)}", flush=True)

        if not args.no_concept:
            print(f"  id={qid} concept start", flush=True)
            stats["concept"] += add_concept(
                question, chosen, rag, llm, args.concept_topk, args.concept_add, args.concept_rr
            )
            print(f"  id={qid} concept done n={len(chosen)}", flush=True)

        arts = collapse_versions([c["art"] for c in chosen])
        arts = apply_v47_v49(question, arts, stats)
        row = {
            "id": qid,
            "question": question,
            "answer": answers.get(qid, ""),
            "relevant_docs": rebuild_docs(arts),
            "relevant_articles": arts,
        }
        rows.append(row)
        done.add(qid)
        processed += 1
        print(f"Q {processed}/{len(questions)} id={qid} done | arts={len(arts)} | {time.time() - q_t0:.1f}s", flush=True)

        if processed % args.checkpoint_every == 0:
            checkpoint(out_path, rows)
            if cache:
                judge_cache.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            rate = processed / max(time.time() - t0, 1e-6)
            left = len(questions) - processed
            print(
                f"[{processed}/{len(questions)}] {rate:.3f} q/s | ETA {left / max(rate, 1e-6) / 60:.1f}m | "
                f"avg arts {statistics.mean(len(r['relevant_articles']) for r in rows):.2f} | stats {stats}",
                flush=True,
            )
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            gc.collect()

    checkpoint(out_path, rows)
    if cache:
        judge_cache.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    avg = statistics.mean(len(r["relevant_articles"]) for r in rows) if rows else 0.0
    print(f"DONE -> {args.out} | rows={len(rows)} | avg_articles={avg:.3f} | stats={stats}", flush=True)

    if args.zip:
        zip_path = out_path.with_suffix(".zip")
        pack_zip(out_path, zip_path)


if __name__ == "__main__":
    main()
