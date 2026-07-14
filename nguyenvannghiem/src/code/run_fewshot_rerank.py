#!/usr/bin/env python3
"""
Few-shot LLM rerank: inject similar QA examples into prompt to calibrate LLM selection.
Uses fewshot_index (FAISS on data_final questions) to find top-3 similar examples.

Usage:
    python3 run_fewshot_rerank.py [--port 8011] [--workers 4] [--sample 50]
"""

import json
import re
import pickle
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

BASE = Path("..") # adjust to your project root
FEWSHOT_CACHE = BASE / "fewshot_examples_cache.pkl"


def load_content_cache():
    CHUNKS_DIR = BASE / "vbpl_dataset" / "chunks"
    DOC_TYPES = ["hien_phap", "bo_luat", "luat", "phap_lenh", "nghi_dinh",
                 "nghi_quyet", "nghi_quyet_lien_tich", "thong_tu", "thong_tu_lien_tich", "quyet_dinh"]
    cc = {}
    for dtype in DOC_TYPES:
        d = CHUNKS_DIR / dtype
        if not d.exists():
            continue
        for fpath in sorted(d.glob("*.json")):
            data = json.loads(fpath.read_text())
            for c in data.get("chunks", []):
                cid = c.get("chunk_id", "")
                content = "\n".join(c.get("content", [])).strip()
                if cid and content:
                    cc[cid] = content
    return cc


def get_text(c, content_cache, max_chars=3000):
    cid = c.get("chunk_id", "")
    content = content_cache.get(cid, "")
    if not content and cid.startswith("qa_"):
        vbid = c.get("van_ban_id", "")
        ano = c.get("article_number", "")
        if vbid and ano:
            content = content_cache.get(f"{vbid}#dieu_{ano}", "")
    doc_title = c.get("doc_title", "")
    art_no = c.get("article_number", "")
    art_title = c.get("article_title", "")
    header = f"{doc_title} - Điều {art_no}"
    if art_title:
        header += f". {art_title}"
    if content:
        return f"{header}\n{content[:max_chars]}"
    return header


def build_fewshot_cache(questions):
    """Pre-compute few-shot examples for all queries. Cache to disk."""
    print("Loading fewshot index...")
    index = faiss.read_index(str(BASE / "fewshot_index/questions.faiss"))
    meta = pickle.load(open(BASE / "fewshot_index/meta.pkl", "rb"))

    print("Loading embedding model...")
    model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", trust_remote_code=True)
    model.max_seq_length = 512

    all_qs = [q["question"] for q in questions]
    print(f"Embedding {len(all_qs)} test queries...")
    embeddings = model.encode(all_qs, normalize_embeddings=True, batch_size=128,
                              show_progress_bar=True).astype("float32")

    D, I = index.search(embeddings, 6)  # top-6 to allow dedup

    cache = {}
    for qi in range(len(questions)):
        examples = []
        seen_q = set()
        for score, idx in zip(D[qi], I[qi]):
            if score < 0.5:
                break
            m = meta[idx]
            q_text = m["question"][:100]
            if q_text in seen_q:
                continue
            seen_q.add(q_text)
            examples.append({
                "question": m["question"],
                "cites": m["cite_short"][:3],
                "score": float(score),
            })
            if len(examples) >= 3:
                break
        cache[qi] = examples

    with open(FEWSHOT_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved fewshot cache: {FEWSHOT_CACHE}")
    return cache


def build_fewshot_block(examples):
    """Format few-shot examples as text block for prompt."""
    if not examples:
        return ""
    lines = ["=== Ví dụ câu hỏi tương tự đã có đáp án ==="]
    for i, ex in enumerate(examples):
        cites_str = " | ".join(ex["cites"])
        lines.append(f"Câu hỏi tương tự {i+1}: {ex['question']}")
        lines.append(f"→ Điều luật trực tiếp áp dụng: {cites_str}")
    lines.append("(Dùng ví dụ trên để hiểu mức độ 'trực tiếp quy định', không phải để copy)")
    lines.append("=== Hết ví dụ ===")
    return "\n".join(lines)


def llm_select(question, candidates, content_cache, fewshot_examples, port, max_retries=2):
    if not candidates:
        return []

    n = len(candidates)
    candidate_lines = []
    for i, c in enumerate(candidates):
        text = get_text(c, content_cache)
        candidate_lines.append(f"[{i+1}] {text}")
    candidates_text = "\n\n".join(candidate_lines)

    fewshot_block = build_fewshot_block(fewshot_examples)

    prompt = f"""Bạn là chuyên gia pháp luật Việt Nam. Nhiệm vụ: xác định CHÍNH XÁC điều luật nào TRỰC TIẾP quy định về vấn đề trong câu hỏi.

{fewshot_block}

Câu hỏi cần xác định: {question}

Các điều luật ứng viên:
{candidates_text}

Quy tắc chọn:
- CHỈ chọn điều luật TRỰC TIẾP quy định hoặc điều chỉnh vấn đề trong câu hỏi
- KHÔNG chọn điều luật chỉ liên quan gián tiếp, chung chung, hoặc chỉ nhắc đến từ khóa tương tự
- Tham khảo ví dụ trên: điều luật được chọn phải CỤ THỂ như trong ví dụ, không phải tất cả điều liên quan
- Thường cần 2-5 điều luật

Trả lời CHỈ bằng danh sách số thứ tự, cách nhau bởi dấu phẩy. Ví dụ: 1,3,5
Chỉ trả lời số, không giải thích."""

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"http://localhost:{port}/v1/chat/completions",
                headers={"Authorization": "Bearer token-abc123", "Content-Type": "application/json"},
                json={
                    "model": "Qwen3-8B-AWQ",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=120,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            answer = (msg.get("content") or "").strip()

            numbers = re.findall(r'\d+', answer)
            selected = [int(x) - 1 for x in numbers if 0 < int(x) <= n]
            selected = list(dict.fromkeys(selected))  # dedup preserve order

            if not selected:
                return candidates[:3]
            return [candidates[i] for i in selected]

        except Exception:
            if attempt < max_retries:
                time.sleep(1)
            else:
                return candidates[:3]


def run(port=8011, workers=4, sample=None):
    questions = json.loads((BASE / "R2AIStage1DATA.json").read_text())
    if sample:
        questions = questions[:sample]

    with open(BASE / "rerank_intersection_scores.pkl", "rb") as f:
        score_cache = pickle.load(f)

    # Load or build fewshot cache
    if FEWSHOT_CACHE.exists():
        print(f"Loading fewshot cache from {FEWSHOT_CACHE}...")
        fewshot_cache = pickle.load(open(FEWSHOT_CACHE, "rb"))
    else:
        fewshot_cache = build_fewshot_cache(questions)

    print("Loading content cache...")
    content_cache = load_content_cache()
    print(f"  {len(content_cache)} chunks")

    candidates_all = [score_cache.get(qi, [])[:15] for qi in range(len(questions))]

    # Stats on few-shot coverage
    has_examples = sum(1 for qi in range(len(questions)) if fewshot_cache.get(qi))
    avg_score = np.mean([ex["score"] for qi in range(len(questions))
                         for ex in fewshot_cache.get(qi, [])] or [0])
    print(f"Few-shot coverage: {has_examples}/{len(questions)} queries have examples")
    print(f"Avg similarity score: {avg_score:.3f}")

    print(f"\nReranking {len(questions)} queries (port={port}, workers={workers})...")
    from retrieval_llm_rerank import make_doc_name, format_submission

    results = [None] * len(questions)
    done = 0
    t0 = time.time()

    def process(qi):
        q = questions[qi]
        cands = candidates_all[qi]
        examples = fewshot_cache.get(qi, [])
        selected = llm_select(q["question"], cands, content_cache, examples, port)
        return qi, selected

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, qi): qi for qi in range(len(questions))}
        for future in as_completed(futures):
            qi, selected = future.result()
            results[qi] = selected
            done += 1
            if done % 100 == 0 or done == len(questions):
                elapsed = time.time() - t0
                eta = elapsed / done * (len(questions) - done)
                print(f"  {done}/{len(questions)} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    print(f"Done in {time.time()-t0:.1f}s")

    submission = [format_submission(q["id"], q["question"], r or [])
                  for q, r in zip(questions, results)]

    suffix = f"_sample{sample}" if sample else ""
    out = BASE / f"submission_3_1_fewshot{suffix}.json"
    out.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")

    avg = np.mean([len(s["relevant_articles"]) for s in submission])
    empty = sum(1 for s in submission if not s["relevant_articles"])
    print(f"Saved: {out}")
    print(f"Avg articles: {avg:.1f}, empty: {empty}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    run(args.port, args.workers, args.sample)
