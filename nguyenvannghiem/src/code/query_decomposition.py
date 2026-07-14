#!/usr/bin/env python3
"""
Query decomposition for multi-hop legal retrieval.
Break complex questions into sub-questions, retrieve for each, merge results.

Usage:
    python3 query_decomposition.py decompose --workers 8
    python3 query_decomposition.py retrieve --source dense --top-k 100
    python3 query_decomposition.py rerank --workers 8 --max-candidates 40
"""

import json
import re
import pickle
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

BASE = Path("..") # adjust to your project root
R2AI_DATA = BASE / "R2AIStage1DATA.json"

LLM_URL = "http://localhost:8011/v1/chat/completions"
LLM_API_KEY = "token-abc123"
LLM_MODEL = "Qwen3-8B-AWQ"

DECOMPOSE_CACHE = BASE / "query_decompose_cache.json"
RETRIEVE_CACHE = BASE / "query_decompose_candidates.pkl"


def decompose_query(question, max_retries=2):
    prompt = f"""Phân tích câu hỏi pháp luật sau và tách thành các mệnh đề con độc lập.

Câu hỏi: {question}

Quy tắc:
- Mỗi mệnh đề con phải là một vấn đề pháp lý cụ thể, có thể tra cứu độc lập
- Giữ nguyên ngữ cảnh (chủ thể, đối tượng) trong mỗi mệnh đề
- Nếu câu hỏi đơn giản (chỉ 1 vấn đề), trả về chính câu hỏi đó
- Tối đa 3 mệnh đề con
- KHÔNG giải thích, chỉ liệt kê

Format:
1. [mệnh đề 1]
2. [mệnh đề 2]
3. [mệnh đề 3]"""

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                LLM_URL,
                headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8000,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                timeout=120,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            answer = (msg.get("content") or "").strip()

            # Parse numbered list
            subs = re.findall(r'\d+\.\s*(.+)', answer)
            if subs:
                return [s.strip().strip('[]') for s in subs[:3]]
            return [question]

        except Exception:
            if attempt < max_retries:
                time.sleep(1)
            else:
                return [question]


def run_decompose(workers=8):
    questions = json.loads(R2AI_DATA.read_text(encoding="utf-8"))
    print(f"Decomposing {len(questions)} questions...")

    if DECOMPOSE_CACHE.exists():
        cache = json.loads(DECOMPOSE_CACHE.read_text(encoding="utf-8"))
        print(f"  Loaded {len(cache)} cached")
    else:
        cache = {}

    done = 0

    def process(qi):
        q = questions[qi]
        key = str(q["id"])
        if key in cache:
            return qi, cache[key], False
        subs = decompose_query(q["question"])
        return qi, subs, True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, i): i for i in range(len(questions))}
        for future in as_completed(futures):
            qi, subs, is_new = future.result()
            q = questions[qi]
            cache[str(q["id"])] = subs
            done += 1
            if done % 50 == 0 or done == len(questions):
                print(f"  Decomposed: {done}/{len(questions)}", flush=True)
                DECOMPOSE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    DECOMPOSE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    # Stats
    counts = [len(cache[str(q["id"])]) for q in questions]
    print(f"\nDone! Avg sub-queries: {np.mean(counts):.1f}")
    print(f"  1 sub: {sum(1 for c in counts if c == 1)}")
    print(f"  2 subs: {sum(1 for c in counts if c == 2)}")
    print(f"  3 subs: {sum(1 for c in counts if c == 3)}")

    # Samples
    for q in questions[:3]:
        key = str(q["id"])
        print(f"\n  Q: {q['question'][:80]}...")
        for i, s in enumerate(cache[key]):
            print(f"    {i+1}. {s[:80]}")


def run_retrieve(source="dense", top_k=100):
    questions = json.loads(R2AI_DATA.read_text(encoding="utf-8"))

    if not DECOMPOSE_CACHE.exists():
        print("ERROR: Run 'decompose' first")
        return

    cache = json.loads(DECOMPOSE_CACHE.read_text(encoding="utf-8"))

    if source == "dense":
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer

        dense_dir = BASE / "retrieval_index_dense"
        print("Loading Dense index...")
        dense_index = faiss.read_index(str(dense_dir / "faiss.index"))
        with open(dense_dir / "metas.pkl", "rb") as f:
            metas = pickle.load(f)

        print("Loading embedding model...")
        model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", model_kwargs={"torch_dtype": torch.float16})
        model.max_seq_length = 512

        # Collect all sub-queries with their parent index
        all_subs = []
        parent_map = []
        for qi, q in enumerate(questions):
            key = str(q["id"])
            subs = cache.get(key, [q["question"]])
            # Always include original question first
            all_queries = [q["question"]] + [s for s in subs if s != q["question"]]
            for sq in all_queries:
                all_subs.append(sq)
                parent_map.append(qi)

        print(f"Total sub-queries: {len(all_subs)} for {len(questions)} questions")

        print("Encoding sub-queries...")
        query_embs = model.encode(
            all_subs, batch_size=64, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)

        print(f"Searching FAISS top-{top_k}...")
        scores, indices = dense_index.search(query_embs, top_k)

        # Merge results per parent question (dedup by law_id + art_no, keep best score)
        all_results = [[] for _ in range(len(questions))]
        for si in range(len(all_subs)):
            qi = parent_map[si]
            for rank in range(top_k):
                idx = int(indices[si, rank])
                score = float(scores[si, rank])
                if idx < 0 or score <= 0:
                    break
                all_results[qi].append({"dense_score": score, "dense_rank": rank, **metas[idx]})

        # Dedup per query, keep best score
        deduped_results = []
        for qi in range(len(questions)):
            seen = {}
            for h in all_results[qi]:
                key = (h.get("law_id", ""), h.get("article_number", ""))
                if key not in seen or h["dense_score"] > seen[key]["dense_score"]:
                    seen[key] = h
            # Sort by score desc
            sorted_hits = sorted(seen.values(), key=lambda x: x["dense_score"], reverse=True)
            deduped_results.append(sorted_hits[:top_k])

        with open(RETRIEVE_CACHE, "wb") as f:
            pickle.dump({"questions": questions, "candidates": deduped_results}, f)

        avg_hits = np.mean([len(r) for r in deduped_results])
        print(f"Saved. Avg candidates: {avg_hits:.1f}/query")


def run_rerank(workers=8, max_candidates=40):
    from retrieval_llm_rerank import (
        load_chunk_content_cache, llm_rerank_all, format_submission
    )

    if not RETRIEVE_CACHE.exists():
        print("ERROR: Run 'retrieve' first")
        return

    with open(RETRIEVE_CACHE, "rb") as f:
        data = pickle.load(f)

    questions = data["questions"]
    candidates = data["candidates"]
    queries = [q["question"] for q in questions]

    print(f"Loaded {len(queries)} queries, avg candidates: {np.mean([len(c) for c in candidates]):.1f}")

    content_cache = load_chunk_content_cache()

    print(f"LLM reranking ({workers} workers, max_candidates={max_candidates})...")
    t0 = time.time()
    llm_results = llm_rerank_all(
        queries, candidates, content_cache,
        n_workers=workers, max_candidates=max_candidates,
        questions=questions, source="decompose"
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Save cache
    with open(BASE / "llm_rerank_cache_decompose.pkl", "wb") as f:
        pickle.dump({"questions": questions, "llm_results": llm_results}, f)

    # Build submission
    submission = [format_submission(q["id"], q["question"], r or [])
                  for q, r in zip(questions, llm_results)]
    out = BASE / "submission_3_1_decompose.json"
    out.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_art = np.mean([len(s["relevant_articles"]) for s in submission])
    print(f"Saved to {out}, avg articles: {avg_art:.1f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_dec = sub.add_parser("decompose")
    p_dec.add_argument("--workers", type=int, default=8)

    p_ret = sub.add_parser("retrieve")
    p_ret.add_argument("--source", choices=["dense"], default="dense")
    p_ret.add_argument("--top-k", type=int, default=100)

    p_rerank = sub.add_parser("rerank")
    p_rerank.add_argument("--workers", type=int, default=8)
    p_rerank.add_argument("--max-candidates", type=int, default=40)

    args = parser.parse_args()

    if args.cmd == "decompose":
        run_decompose(args.workers)
    elif args.cmd == "retrieve":
        run_retrieve(args.source, args.top_k)
    elif args.cmd == "rerank":
        run_rerank(args.workers, args.max_candidates)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
