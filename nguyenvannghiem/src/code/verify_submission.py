#!/usr/bin/env python3
"""
Verify each article in a submission individually via LLM.
Remove false positives to increase precision.

Usage:
    python3 verify_submission.py verify --input results.json --workers 8
    python3 verify_submission.py submit --input results.json
"""

import json
import re
import os
import pickle
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE = Path("..") # adjust to your project root
CHUNKS_DIR = BASE / "vbpl_dataset" / "chunks"

LLM_URL = "http://localhost:8011/v1/chat/completions"
LLM_API_KEY = "token-abc123"
LLM_MODEL = "Qwen3-8B-AWQ"

DOC_TYPES = [
    "hien_phap", "bo_luat", "luat", "phap_lenh",
    "nghi_dinh", "nghi_quyet", "nghi_quyet_lien_tich",
    "thong_tu", "thong_tu_lien_tich", "quyet_dinh",
]

_CONTENT_CACHE = {}


def load_chunk_content_cache():
    global _CONTENT_CACHE
    if _CONTENT_CACHE:
        return _CONTENT_CACHE
    print("Loading chunk content cache...")
    t0 = time.time()
    count = 0
    for dtype in DOC_TYPES:
        chunk_dir = CHUNKS_DIR / dtype
        if not chunk_dir.exists():
            continue
        for fpath in sorted(chunk_dir.glob("*.json")):
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            for chunk in data.get("chunks", []):
                cid = chunk.get("chunk_id", "")
                if not cid:
                    continue
                content = "\n".join(chunk.get("content", [])).strip()
                if content:
                    _CONTENT_CACHE[cid] = content
                    count += 1
    print(f"  Loaded {count} chunks in {time.time()-t0:.1f}s")
    return _CONTENT_CACHE


def parse_article_key(art_key):
    parts = art_key.split("|")
    if len(parts) >= 3:
        law_id = parts[0]
        doc_name = parts[1]
        dieu_part = parts[2]
        art_no_match = re.search(r'\d+', dieu_part)
        art_no = art_no_match.group() if art_no_match else ""
        return law_id, doc_name, art_no
    return "", "", ""


def build_article_content_index():
    """Build index: (law_id, article_number) -> content using dense metas + chunk cache"""
    print("Building article content index...")
    t0 = time.time()
    content_cache = load_chunk_content_cache()

    import pickle
    dense_meta_path = BASE / "retrieval_index_dense" / "metas.pkl"
    with open(dense_meta_path, "rb") as f:
        metas = pickle.load(f)

    index = {}
    for m in metas:
        lid = m.get("law_id", "")
        art_no = str(m.get("article_number", ""))
        cid = m.get("chunk_id", "")
        if not (lid and art_no and cid):
            continue
        key = (lid, art_no)
        if key in index:
            continue
        content = content_cache.get(cid, "")
        if content:
            index[key] = content
    print(f"  Indexed {len(index)} articles in {time.time()-t0:.1f}s")
    return index


def llm_verify_article(question, art_key, article_content, max_retries=2):
    law_id, doc_name, art_no = parse_article_key(art_key)

    header = f"{doc_name} - Điều {art_no}"
    if article_content:
        article_text = f"{header}\n{article_content[:1500]}"
    else:
        article_text = header

    prompt = f"""Bạn là chuyên gia pháp luật Việt Nam. Xác định điều luật sau có TRỰC TIẾP quy định về vấn đề trong câu hỏi không.

Câu hỏi: {question}

Điều luật:
{article_text}

Điều luật này có trực tiếp liên quan và được viện dẫn để trả lời câu hỏi không?
Trả lời CHỈ "YES" hoặc "NO"."""

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
            answer = (msg.get("content") or "").strip().upper()

            if "YES" in answer:
                return True
            elif "NO" in answer:
                return False
            else:
                return True  # default keep (recall-friendly)

        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
            else:
                return True  # keep on error (recall-friendly)


def run_verify(input_file, workers=8):
    with open(input_file) as f:
        submission = json.load(f)

    art_index = build_article_content_index()

    # Build task list: (query_idx, article_idx, question, art_key)
    tasks = []
    for qi, item in enumerate(submission):
        for ai, art_key in enumerate(item.get("relevant_articles", [])):
            tasks.append((qi, ai, item["question"], art_key))

    print(f"Total articles to verify: {len(tasks)}")

    cache_file = BASE / "verify_cache.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        print(f"  Loaded {len(cache)} cached results")
    else:
        cache = {}

    results = {}
    done = 0
    kept = 0
    removed = 0

    def process(task):
        qi, ai, question, art_key = task
        cache_key = f"{qi}|{art_key}"
        if cache_key in cache:
            return qi, ai, art_key, cache[cache_key]

        law_id, doc_name, art_no = parse_article_key(art_key)
        content = art_index.get((law_id, str(art_no)), "")
        verdict = llm_verify_article(question, art_key, content)
        return qi, ai, art_key, verdict

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, t): t for t in tasks}
        for future in as_completed(futures):
            qi, ai, art_key, verdict = future.result()
            cache_key = f"{qi}|{art_key}"
            cache[cache_key] = verdict
            results[(qi, ai)] = verdict
            done += 1
            if verdict:
                kept += 1
            else:
                removed += 1

            if done % 50 == 0 or done == len(tasks):
                print(f"  Verified: {done}/{len(tasks)} | kept={kept} removed={removed} ({removed/done*100:.1f}% removed)", flush=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(cache, f)

    # Save final cache
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)

    # Build filtered submission
    filtered = []
    for qi, item in enumerate(submission):
        new_articles = []
        new_docs = set()
        for ai, art_key in enumerate(item.get("relevant_articles", [])):
            if results.get((qi, ai), True):
                new_articles.append(art_key)
                parts = art_key.split("|")
                if len(parts) >= 2:
                    new_docs.add(f"{parts[0]}|{parts[1]}")

        # Rebuild answer from kept articles
        answer_parts = []
        for art_key in new_articles:
            parts = art_key.split("|")
            if len(parts) >= 3:
                answer_parts.append(f"Theo {parts[2]} {parts[1]}")

        filtered.append({
            "id": item["id"],
            "question": item["question"],
            "answer": ". ".join(answer_parts) + "." if answer_parts else "",
            "relevant_docs": list(new_docs),
            "relevant_articles": new_articles,
        })

    out_file = str(input_file).replace(".json", "_verified.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    orig_total = sum(len(item["relevant_articles"]) for item in submission)
    new_total = sum(len(item["relevant_articles"]) for item in filtered)
    print(f"\nDone! {orig_total} -> {new_total} articles ({orig_total - new_total} removed, {(orig_total-new_total)/orig_total*100:.1f}%)")
    print(f"Saved to {out_file}")

    # Stats
    new_counts = [len(item["relevant_articles"]) for item in filtered]
    print(f"Avg articles/query: {sum(new_counts)/len(new_counts):.1f}")
    empty = sum(1 for c in new_counts if c == 0)
    print(f"Empty queries: {empty}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--input", required=True)
    p_verify.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()

    if args.cmd == "verify":
        run_verify(args.input, args.workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
