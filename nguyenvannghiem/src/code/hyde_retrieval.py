#!/usr/bin/env python3
"""
HyDE (Hypothetical Document Embeddings) for legal retrieval.
LLM generates a hypothetical legal article, then use its embedding to retrieve.

Usage:
    python3 hyde_retrieval.py generate --workers 8 --port 8011
    python3 hyde_retrieval.py retrieve --top-k 100
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
CACHE_FILE = BASE / "hyde_cache.json"


def generate_hyde(question, llm_url, max_retries=2):
    prompt = f"""Bạn là chuyên gia pháp luật Việt Nam. Hãy viết một đoạn văn bản pháp luật (điều luật) mà bạn nghĩ sẽ được viện dẫn để trả lời câu hỏi sau.

Câu hỏi: {question}

Yêu cầu:
- Viết như một điều luật thật trong văn bản pháp luật Việt Nam
- Bao gồm nội dung quy định cụ thể liên quan đến câu hỏi
- Viết 3-5 câu, ngắn gọn, đúng phong cách pháp lý
- KHÔNG nhắc số hiệu văn bản, số Điều cụ thể
- Chỉ viết nội dung điều luật, không giải thích"""

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                llm_url,
                headers={"Authorization": "Bearer token-abc123", "Content-Type": "application/json"},
                json={
                    "model": "Qwen3-8B-AWQ",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0,
                },
                timeout=60,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            return (msg.get("content") or "").strip()
        except Exception:
            if attempt < max_retries:
                time.sleep(1)
            else:
                return ""


def run_generate(workers=8, port=8011):
    llm_url = f"http://localhost:{port}/v1/chat/completions"
    questions = json.loads(R2AI_DATA.read_text(encoding="utf-8"))
    print(f"Generating HyDE for {len(questions)} questions (port {port})...")

    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        print(f"  Loaded {len(cache)} cached")
    else:
        cache = {}

    done = 0

    def process(qi):
        q = questions[qi]
        key = str(q["id"])
        if key in cache:
            return qi, cache[key], False
        hyde = generate_hyde(q["question"], llm_url)
        return qi, hyde, True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, i): i for i in range(len(questions))}
        for future in as_completed(futures):
            qi, hyde, is_new = future.result()
            q = questions[qi]
            cache[str(q["id"])] = hyde
            done += 1
            if done % 50 == 0 or done == len(questions):
                print(f"  Generated: {done}/{len(questions)}", flush=True)
                CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    lengths = [len(cache[str(q["id"])]) for q in questions if str(q["id"]) in cache]
    print(f"Done! Avg length: {np.mean(lengths):.0f} chars")
    empty = sum(1 for l in lengths if l == 0)
    print(f"Empty: {empty}")

    for q in questions[:3]:
        key = str(q["id"])
        print(f"\n  Q: {q['question'][:60]}...")
        print(f"  HyDE: {cache[key][:120]}...")


def run_retrieve(top_k=100):
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    questions = json.loads(R2AI_DATA.read_text(encoding="utf-8"))
    if not CACHE_FILE.exists():
        print("ERROR: Run 'generate' first")
        return
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    # Also load decompose cache for sub-queries
    decompose_cache = {}
    dc_path = BASE / "query_decompose_cache.json"
    if dc_path.exists():
        decompose_cache = json.loads(dc_path.read_text(encoding="utf-8"))

    dense_dir = BASE / "retrieval_index_dense"
    print("Loading Dense index...")
    dense_index = faiss.read_index(str(dense_dir / "faiss.index"))
    with open(dense_dir / "metas.pkl", "rb") as f:
        metas = pickle.load(f)

    print("Loading embedding model...")
    model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = 512

    # Build queries: original + decomposed sub-queries + HyDE
    all_queries = []
    parent_map = []
    query_types = []

    for qi, q in enumerate(questions):
        key = str(q["id"])

        # Original question
        all_queries.append(q["question"])
        parent_map.append(qi)
        query_types.append("original")

        # Decomposed sub-queries
        subs = decompose_cache.get(key, [])
        for s in subs:
            if s != q["question"]:
                all_queries.append(s)
                parent_map.append(qi)
                query_types.append("decompose")

        # HyDE
        hyde = cache.get(key, "")
        if hyde:
            all_queries.append(hyde)
            parent_map.append(qi)
            query_types.append("hyde")

    print(f"Total queries: {len(all_queries)} ({len(questions)} original + decompose + hyde)")

    print("Encoding...")
    query_embs = model.encode(
        all_queries, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    print(f"Searching FAISS top-{top_k}...")
    scores, indices = dense_index.search(query_embs, top_k)

    # Merge results per parent question
    all_results = [{} for _ in range(len(questions))]
    for si in range(len(all_queries)):
        qi = parent_map[si]
        for rank in range(top_k):
            idx = int(indices[si, rank])
            score = float(scores[si, rank])
            if idx < 0 or score <= 0:
                break
            m = metas[idx]
            key = (m.get("law_id", ""), str(m.get("article_number", "")))
            if key not in all_results[qi] or score > all_results[qi][key]["dense_score"]:
                all_results[qi][key] = {"dense_score": score, "dense_rank": rank, **m}

    # Sort and limit
    candidates = []
    for qi in range(len(questions)):
        sorted_hits = sorted(all_results[qi].values(), key=lambda x: x["dense_score"], reverse=True)[:top_k]
        candidates.append(sorted_hits)

    out_file = BASE / "hyde_candidates.pkl"
    with open(out_file, "wb") as f:
        pickle.dump({"questions": questions, "candidates": candidates}, f)

    avg_hits = np.mean([len(c) for c in candidates])
    print(f"Saved to {out_file}, avg candidates: {avg_hits:.1f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--workers", type=int, default=8)
    p_gen.add_argument("--port", type=int, default=8011)

    p_ret = sub.add_parser("retrieve")
    p_ret.add_argument("--top-k", type=int, default=100)

    args = parser.parse_args()

    if args.cmd == "generate":
        run_generate(args.workers, args.port)
    elif args.cmd == "retrieve":
        run_retrieve(args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
