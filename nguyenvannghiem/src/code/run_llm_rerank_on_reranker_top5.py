#!/usr/bin/env python3
"""
LLM rerank on reranker V2 ck-8000 top-5 candidates.
Feeds only top-5 reranker candidates to the LLM for final selection.

Usage:
    python3 run_llm_rerank_on_reranker_top5.py --port 8011 --model Qwen3-8B-AWQ --tag 8b
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
API_KEY = "token-abc123"


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


def get_text(c, content_cache, max_chars=10000):
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


def llm_select(question, candidates, content_cache, port, model, max_retries=2):
    if not candidates:
        return []

    n = len(candidates)
    candidate_lines = []
    for i, c in enumerate(candidates):
        text = get_text(c, content_cache)
        candidate_lines.append(f"[{i+1}] {text}")
    candidates_text = "\n\n".join(candidate_lines)

    prompt = f"""Bạn là chuyên gia pháp luật Việt Nam. Nhiệm vụ: xác định CHÍNH XÁC điều luật nào TRỰC TIẾP quy định về vấn đề trong câu hỏi.

Câu hỏi: {question}

Các điều luật ứng viên:
{candidates_text}

Quy tắc chọn:
- CHỈ chọn điều luật TRỰC TIẾP quy định hoặc điều chỉnh vấn đề trong câu hỏi
- KHÔNG chọn điều luật chỉ liên quan gián tiếp, chung chung, hoặc chỉ nhắc đến từ khóa tương tự
- Đọc kỹ nội dung điều luật, đối chiếu với câu hỏi trước khi quyết định
- Thường cần 2-5 điều luật

Trả lời CHỈ bằng danh sách số thứ tự, cách nhau bởi dấu phẩy. Ví dụ: 1,3,5
Chỉ trả lời số, không giải thích."""

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"http://localhost:{port}/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8000,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                timeout=180,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            answer = (msg.get("content") or "").strip()
            thinking = msg.get("reasoning_content") or msg.get("reasoning") or ""

            numbers = re.findall(r'\d+', answer)
            selected = [int(x) - 1 for x in numbers if 0 < int(x) <= n]
            selected = list(dict.fromkeys(selected))

            result_cands = [candidates[i] for i in selected] if selected else candidates[:3]
            return result_cands, thinking

        except Exception:
            if attempt < max_retries:
                time.sleep(1)
            else:
                return candidates[:3], ""


def run(port, model, tag, workers=4):
    questions = json.loads((BASE / "R2AIStage1DATA.json").read_text())

    with open(BASE / "rerank_intersection_scores.pkl", "rb") as f:
        score_cache = pickle.load(f)

    print("Loading content cache...")
    content_cache = load_content_cache()
    print(f"  {len(content_cache)} chunks")

    # Take reranker top-5 as candidates
    candidates_all = [score_cache.get(qi, [])[:5] for qi in range(len(questions))]
    avg_cands = np.mean([len(c) for c in candidates_all])
    print(f"  Avg candidates: {avg_cands:.1f}")

    def make_doc_name(doc_title, doc_description):
        if doc_description:
            return f"{doc_title} {doc_description}"
        return doc_title

    def format_submission(qid, question, selected_cands):
        answer_parts, relevant_docs, relevant_articles = [], [], []
        seen_docs, seen_articles = set(), set()
        for r in selected_cands:
            law_id = r.get("law_id", "")
            doc_name = make_doc_name(r.get("doc_title", ""), r.get("doc_description", ""))
            art_no = r.get("article_number", "")
            doc_key = f"{law_id}|{doc_name}"
            if doc_key not in seen_docs and law_id:
                seen_docs.add(doc_key)
                relevant_docs.append(doc_key)
            art_key = f"{law_id}|{doc_name}|Điều {art_no}"
            if art_key not in seen_articles and law_id and art_no:
                seen_articles.add(art_key)
                relevant_articles.append(art_key)
                art_title = r.get("article_title", "")
                if art_title:
                    answer_parts.append(f"Theo Điều {art_no} {doc_name}: {art_title}")
                else:
                    answer_parts.append(f"Căn cứ Điều {art_no} {doc_name}")
        return {
            "id": qid, "question": question,
            "answer": ". ".join(answer_parts) + "." if answer_parts else "",
            "relevant_docs": relevant_docs, "relevant_articles": relevant_articles,
        }

    print(f"\nLLM rerank on reranker top-5 (port={port}, model={model}, workers={workers})...")
    results = [None] * len(questions)
    done = 0
    t0 = time.time()

    all_thinking = [None] * len(questions)

    def process(qi):
        q = questions[qi]
        selected, thinking = llm_select(q["question"], candidates_all[qi], content_cache, port, model)
        return qi, selected, thinking

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, qi): qi for qi in range(len(questions))}
        for future in as_completed(futures):
            qi, selected, thinking = future.result()
            results[qi] = selected
            all_thinking[qi] = thinking
            done += 1
            if done % 100 == 0 or done == len(questions):
                elapsed = time.time() - t0
                eta = elapsed / done * (len(questions) - done)
                print(f"  {done}/{len(questions)} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    print(f"Done in {time.time()-t0:.1f}s")

    submission = [format_submission(q["id"], q["question"], r or [])
                  for q, r in zip(questions, results)]

    out = BASE / f"submission_3_1_llm{tag}_rerankerv2ck8k_top5.json"
    out.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")

    avg = np.mean([len(s["relevant_articles"]) for s in submission])
    empty = sum(1 for s in submission if not s["relevant_articles"])
    print(f"Saved: {out}")
    print(f"Avg articles: {avg:.1f}, empty: {empty}")

    # Save thinking traces
    thinking_path = BASE / f"thinking_traces_llm{tag}_rerankerv2ck8k_top5.json"
    thinking_data = []
    for qi, q in enumerate(questions):
        thinking_data.append({
            "id": q["id"],
            "question": q["question"],
            "thinking": all_thinking[qi] or "",
            "selected": [format_submission(q["id"], q["question"], results[qi] or [])["relevant_articles"]],
        })
    thinking_path.write_text(json.dumps(thinking_data, ensure_ascii=False, indent=2))
    print(f"Saved thinking: {thinking_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", type=str, default="Qwen3-8B-AWQ")
    parser.add_argument("--tag", type=str, required=True, help="output tag, e.g. 8b")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.port, args.model, args.tag, args.workers)
