#!/usr/bin/env python3
"""
Generate synthetic questions for each legal article using LLM.
Creates diverse, user-like questions that don't mention article names.

Usage:
    python3 gen_synthetic_questions.py --n-workers 16 --batch-start 0 --batch-end 50000
    python3 gen_synthetic_questions.py --resume
"""

import json
import re
import os
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE = Path("..") # adjust to your project root
CHUNKS_DIR = BASE / "vbpl_dataset" / "chunks"
META_DIR = BASE / "vbpl_dataset" / "metadata"
OUTPUT_DIR = BASE / "synthetic_qa"

LLM_URL = "http://localhost:8011/v1/chat/completions"
LLM_API_KEY = "token-abc123"
LLM_MODEL = "Qwen3-8B-AWQ"

DOC_TYPES = [
    "hien_phap", "bo_luat", "luat", "phap_lenh",
    "nghi_dinh", "nghi_quyet", "nghi_quyet_lien_tich",
    "thong_tu", "thong_tu_lien_tich", "quyet_dinh",
]

DOC_NUM_RE = re.compile(r'(\d+/\d{4}/[A-ZĐa-zđ\d\-]+)')


def extract_law_id(title):
    m = DOC_NUM_RE.search(title)
    return m.group(1) if m else ""


def load_all_articles():
    meta_map = {}
    for dtype in DOC_TYPES:
        meta_file = META_DIR / f"{dtype}.json"
        if not meta_file.exists():
            continue
        docs = json.loads(meta_file.read_text(encoding="utf-8"))
        for doc in docs:
            vb_id = f"{dtype}/{doc['item_id']}"
            meta_map[vb_id] = {
                "title": doc.get("title", ""),
                "description": doc.get("description", ""),
            }

    articles = []
    for dtype in DOC_TYPES:
        chunk_dir = CHUNKS_DIR / dtype
        if not chunk_dir.exists():
            continue
        for fpath in sorted(chunk_dir.glob("*.json")):
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception:
                continue

            van_ban_id = f"{dtype}/{fpath.stem}"
            vb_meta = meta_map.get(van_ban_id, {})
            doc_title = vb_meta.get("title", "")
            doc_desc = vb_meta.get("description", "")
            law_id = extract_law_id(doc_title)

            for chunk in data.get("chunks", []):
                article_no = chunk.get("article_number", "")
                article_title = chunk.get("article_title", "")
                content = "\n".join(chunk.get("content", [])).strip()

                if not content and not article_title:
                    continue

                articles.append({
                    "chunk_id": chunk.get("chunk_id", ""),
                    "van_ban_id": van_ban_id,
                    "law_id": law_id,
                    "doc_title": doc_title,
                    "doc_description": doc_desc,
                    "article_number": article_no,
                    "article_title": article_title,
                    "content": content[:2000],
                })

    return articles


def gen_questions_for_article(article, max_retries=2):
    doc_title = article["doc_title"]
    art_no = article["article_number"]
    art_title = article["article_title"]
    content = article["content"]

    if not content or len(content) < 50:
        return []

    text = f"{doc_title}\nĐiều {art_no}. {art_title}\n{content[:1500]}"

    prompt = f"""Đọc điều luật sau và tạo 1-3 cặp câu hỏi - câu trả lời mà người dân thường sẽ hỏi, mà đáp án nằm trong điều luật này.

Điều luật:
{text}

Yêu cầu:
- Câu hỏi viết bằng ngôn ngữ đời thường, giống người dân thực sự hỏi
- KHÔNG nhắc đến tên điều luật, số hiệu văn bản, hoặc "Điều X" trong câu hỏi
- Câu trả lời ngắn gọn, chính xác, dựa trên nội dung điều luật
- Mỗi cặp Q&A phải khác nhau về góc độ/khía cạnh

Trả lời ĐÚNG FORMAT (mỗi cặp 2 dòng):
Q1: [câu hỏi 1]
A1: [câu trả lời 1]
Q2: [câu hỏi 2]
A2: [câu trả lời 2]
Q3: [câu hỏi 3]
A3: [câu trả lời 3]"""

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                LLM_URL,
                headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.7,
                },
                timeout=30,
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip()

            qa_pairs = []
            lines = answer.split("\n")
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                qm = re.match(r'^Q\d+[:\.]?\s*(.+)', line)
                if qm:
                    q = qm.group(1).strip()
                    a = ""
                    if i + 1 < len(lines):
                        am = re.match(r'^A\d+[:\.]?\s*(.+)', lines[i+1].strip())
                        if am:
                            a = am.group(1).strip()
                            i += 1
                    if len(q) > 15:
                        q = q if q.endswith("?") else q + "?"
                        qa_pairs.append({"question": q, "answer": a})
                i += 1

            return qa_pairs[:3]

        except Exception:
            if attempt < max_retries:
                time.sleep(1)
            else:
                return []


def process_batch(articles, n_workers=16, output_file=None):
    results = []
    done = 0
    total = len(articles)

    def process(idx):
        art = articles[idx]
        qa_pairs = gen_questions_for_article(art)
        return idx, qa_pairs

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process, i): i for i in range(total)}
        for future in as_completed(futures):
            idx, qa_pairs = future.result()
            art = articles[idx]
            for qa in qa_pairs:
                results.append({
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "chunk_id": art["chunk_id"],
                    "van_ban_id": art["van_ban_id"],
                    "law_id": art["law_id"],
                    "doc_title": art["doc_title"],
                    "doc_description": art["doc_description"],
                    "article_number": art["article_number"],
                    "article_title": art["article_title"],
                })
            done += 1
            if done % 500 == 0 or done == total:
                print(f"  {done}/{total} articles, {len(results)} questions generated", flush=True)

            if output_file and done % 5000 == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--batch-start", type=int, default=0)
    parser.add_argument("--batch-end", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all articles...")
    articles = load_all_articles()
    print(f"  {len(articles)} articles total")

    end = args.batch_end or len(articles)
    batch = articles[args.batch_start:end]
    print(f"  Processing [{args.batch_start}:{end}] = {len(batch)} articles")

    output_file = OUTPUT_DIR / f"synthetic_{args.batch_start}_{end}.json"

    if args.resume and output_file.exists():
        existing = json.loads(output_file.read_text(encoding="utf-8"))
        print(f"  Resuming from {len(existing)} existing questions")

    print(f"Generating questions ({args.n_workers} workers)...")
    t0 = time.time()
    results = process_batch(batch, n_workers=args.n_workers, output_file=output_file)
    elapsed = time.time() - t0

    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  {len(results)} questions from {len(batch)} articles")
    print(f"  Avg {len(results)/max(len(batch),1):.1f} questions/article")
    print(f"  Saved to {output_file}")


if __name__ == "__main__":
    main()
