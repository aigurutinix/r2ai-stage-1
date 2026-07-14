#!/usr/bin/env python3
"""
LLM-based reranking/classification for R2AI Task 3.1.

Pipeline: BM25S top-N → LLM selects relevant articles → submission

Usage:
    python3 retrieval_llm_rerank.py rerank-all --candidates 20
    python3 retrieval_llm_rerank.py submit
    python3 retrieval_llm_rerank.py evaluate --limit 50 --candidates 20
"""

import json
import re
import os
import pickle
import argparse
import unicodedata
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import bm25s
import faiss
import requests
from underthesea import word_tokenize as vi_tokenize

BASE = Path("..") # adjust to your project root
CHUNKS_DIR = BASE / "vbpl_dataset" / "chunks"
DATA_FINAL = BASE / "data_final"
INDEX_DIR = BASE / "retrieval_index_bm25s_v7"
R2AI_DATA = BASE / "R2AIStage1DATA.json"
CACHE_PATH = BASE / "llm_rerank_cache.pkl"
DENSE_INDEX_DIR = BASE / "retrieval_index_dense"

LLM_URL = "http://localhost:8011/v1/chat/completions"
LLM_API_KEY = "token-abc123"
LLM_MODEL = "Qwen3-8B-AWQ"

DOC_TYPES = [
    "hien_phap", "bo_luat", "luat", "phap_lenh",
    "nghi_dinh", "nghi_quyet", "nghi_quyet_lien_tich",
    "thong_tu", "thong_tu_lien_tich", "quyet_dinh",
]

DIEU_RE = re.compile(r'[Đđ]iều\s+(\d+)', re.IGNORECASE)
DOC_NUM_RE = re.compile(r'(\d+/\d{4}/[A-ZĐa-zđ\d\-]+)')

_CONTENT_CACHE = {}


def normalize_text(text, use_vi_tok=False):
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if use_vi_tok:
        try:
            text = vi_tokenize(text, format='text')
        except Exception:
            pass
    return text


def extract_dieu_numbers(text):
    return set(DIEU_RE.findall(text))


def extract_law_id(title):
    m = DOC_NUM_RE.search(title)
    return m.group(1) if m else ""


def make_doc_name(doc_title, doc_description):
    title = doc_title.strip()
    desc = doc_description.strip()
    if desc and title:
        if desc.lower().startswith(title.split()[0].lower()):
            return desc
        return f"{title} {desc}"
    return title or desc


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


def load_index():
    print("Loading BM25S index...")
    t0 = time.time()
    model = bm25s.BM25.load(str(INDEX_DIR / "bm25s_model"), load_corpus=False)
    with open(INDEX_DIR / "metas.pkl", "rb") as f:
        metas = pickle.load(f)
    print(f"  {len(metas)} chunks, loaded in {time.time()-t0:.1f}s")
    return model, metas


def retrieve_bm25s(queries, model, metas, top_k=20, batch_size=100):
    normalized = [normalize_text(q, use_vi_tok=True) for q in queries]
    all_results = []

    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        batch = normalized[start:end]

        query_tokens = bm25s.tokenize(
            batch, lower=True, stopwords=None,
            token_pattern=r'(?u)\b\w\w+\b',
            return_ids=True, show_progress=False,
        )
        results, scores = model.retrieve(query_tokens, k=top_k)

        for j in range(len(batch)):
            hits = []
            for rank in range(top_k):
                idx = int(results[j, rank])
                score = float(scores[j, rank])
                if score <= 0:
                    break
                hits.append({"bm25_score": score, "bm25_rank": rank, **metas[idx]})
            all_results.append(hits)

    return all_results


def build_candidate_text(meta, content_cache, max_chars=800):
    chunk_id = meta.get("chunk_id", "")
    doc_title = meta.get("doc_title", "").strip()
    art_title = meta.get("article_title", "").strip()
    art_no = meta.get("article_number", "")

    content = content_cache.get(chunk_id, "")
    if not content and chunk_id.startswith("qa_"):
        van_ban_id = meta.get("van_ban_id", "")
        if van_ban_id and art_no:
            content = content_cache.get(f"{van_ban_id}#dieu_{art_no}", "")

    header = f"{doc_title} - Điều {art_no}"
    if art_title:
        header += f". {art_title}"

    if content:
        return f"{header}\n{content[:max_chars]}"
    return header


def deduplicate_candidates(hits):
    seen = set()
    unique = []
    for h in hits:
        key = (h.get("law_id", ""), h.get("article_number", ""))
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique


def llm_select_articles(question, candidates, content_cache, max_retries=2, max_candidates=40):
    if not candidates:
        return []

    deduped = deduplicate_candidates(candidates)[:max_candidates]

    candidate_lines = []
    for i, c in enumerate(deduped):
        text = build_candidate_text(c, content_cache, max_chars=3000)
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
- Thường cần 2-6 điều luật

Trả lời CHỈ bằng danh sách số thứ tự, cách nhau bởi dấu phẩy. Ví dụ: 1,3,5
Chỉ trả lời số, không giải thích."""

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
                timeout=180,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            answer = (msg.get("content") or "").strip()

            numbers = re.findall(r'\d+', answer)
            selected_indices = [int(n) - 1 for n in numbers if 0 < int(n) <= len(deduped)]

            if not selected_indices:
                return deduped[:3]

            return [deduped[i] for i in selected_indices]

        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
            else:
                return deduped[:5]


def llm_rerank_all(queries, all_bm25_results, content_cache, n_workers=8, max_candidates=40, questions=None, source="dense"):
    results = [None] * len(queries)
    done = 0

    def process(idx):
        return idx, llm_select_articles(queries[idx], all_bm25_results[idx], content_cache, max_candidates=max_candidates)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process, i): i for i in range(len(queries))}
        for future in as_completed(futures):
            idx, selected = future.result()
            results[idx] = selected
            done += 1
            if done % 5 == 0 or done == len(queries):
                print(f"  LLM rerank: {done}/{len(queries)}", flush=True)
                if questions:
                    cache_file = BASE / f"llm_rerank_cache_{source}.pkl"
                    with open(cache_file, "wb") as f:
                        pickle.dump({"questions": questions, "llm_results": results}, f)

    return results


def format_submission(qid, question, results):
    answer_parts = []
    relevant_docs = []
    relevant_articles = []
    seen_docs = set()
    seen_articles = set()

    for r in results:
        law_id = r["law_id"]
        doc_name = make_doc_name(r["doc_title"], r["doc_description"])
        art_no = r["article_number"]

        doc_key = f"{law_id}|{doc_name}"
        if doc_key not in seen_docs and law_id:
            seen_docs.add(doc_key)
            relevant_docs.append(doc_key)

        art_key = f"{law_id}|{doc_name}|Điều {art_no}"
        if art_key not in seen_articles and law_id and art_no:
            seen_articles.add(art_key)
            relevant_articles.append(art_key)
            title_part = r["article_title"]
            if title_part:
                answer_parts.append(f"Theo Điều {art_no} {doc_name}: {title_part}")
            else:
                answer_parts.append(f"Căn cứ Điều {art_no} {doc_name}")

    return {
        "id": qid,
        "question": question,
        "answer": ". ".join(answer_parts) + "." if answer_parts else "",
        "relevant_docs": relevant_docs,
        "relevant_articles": relevant_articles,
    }


def load_ground_truth(limit=None):
    qa_items = []
    for f in sorted(DATA_FINAL.rglob("qa_*.json")):
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            cites = item.get("article_cite", [])
            if not cites:
                continue
            gt_dieu = set()
            for c in cites:
                if isinstance(c, str):
                    c = {"title": c}
                gt_dieu.update(extract_dieu_numbers(c.get("title", "")))
            if gt_dieu:
                qa_items.append({"question": item["question"], "gt_dieu": gt_dieu})
        if limit and len(qa_items) >= limit:
            break
    return qa_items[:limit] if limit else qa_items


def compute_metrics(predictions, ground_truths):
    precisions, recalls = [], []
    for pred, gt in zip(predictions, ground_truths):
        pred_dieu = extract_dieu_numbers(pred.get("answer", ""))
        gt_set = gt["gt_dieu"]
        if not pred_dieu and not gt_set:
            precisions.append(1.0); recalls.append(1.0); continue
        if not pred_dieu:
            precisions.append(0.0); recalls.append(0.0); continue
        if not gt_set:
            precisions.append(0.0); recalls.append(1.0); continue
        tp = len(pred_dieu & gt_set)
        precisions.append(tp / len(pred_dieu))
        recalls.append(tp / len(gt_set))
    P = np.mean(precisions)
    R = np.mean(recalls)
    F2 = (5 * P * R) / (4 * P + R) if (4 * P + R) > 0 else 0
    return {"precision": P, "recall": R, "f2": F2}


def run_evaluate(limit=50, candidates=20):
    bm25_model, metas = load_index()
    content_cache = load_chunk_content_cache()

    gt_data = load_ground_truth(limit)
    queries = [g["question"] for g in gt_data]
    print(f"  {len(gt_data)} questions")

    print(f"BM25S top-{candidates}...")
    bm25_results = retrieve_bm25s(queries, bm25_model, metas, top_k=candidates)

    # BM25S baselines
    print("\nBM25S baselines:")
    for k in [3, 5, 8]:
        preds = [format_submission(i, g["question"], hits[:k])
                 for i, (g, hits) in enumerate(zip(gt_data, bm25_results))]
        m = compute_metrics(preds, gt_data)
        print(f"  k={k}: P={m['precision']:.4f} R={m['recall']:.4f} F2={m['f2']:.4f}")

    # LLM rerank
    print(f"\nLLM reranking {len(queries)} queries...")
    t0 = time.time()
    llm_results = llm_rerank_all(queries, bm25_results, content_cache, n_workers=8)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(queries)/elapsed:.1f} q/s)")

    preds_llm = [format_submission(i, g["question"], r or [])
                 for i, (g, r) in enumerate(zip(gt_data, llm_results))]
    m = compute_metrics(preds_llm, gt_data)
    avg_k = np.mean([len(r) for r in llm_results if r])
    print(f"\n  LLM rerank: P={m['precision']:.4f} R={m['recall']:.4f} F2={m['f2']:.4f} avg_k={avg_k:.1f}")

    # Samples
    print("\nSamples:")
    for i in range(min(5, len(gt_data))):
        gt = sorted(gt_data[i]["gt_dieu"])
        pred_bm25 = sorted(extract_dieu_numbers(format_submission(i, "", bm25_results[i][:5]).get("answer", "")))
        pred_llm = sorted(extract_dieu_numbers(preds_llm[i].get("answer", "")))
        print(f"  Q{i+1}: GT={gt} BM25={pred_bm25} LLM={pred_llm}")


def load_dense_index():
    print("Loading Dense index...")
    t0 = time.time()
    dense_index = faiss.read_index(str(DENSE_INDEX_DIR / "faiss.index"))
    with open(DENSE_INDEX_DIR / "metas.pkl", "rb") as f:
        dense_metas = pickle.load(f)
    print(f"  {dense_index.ntotal} vectors, loaded in {time.time()-t0:.1f}s")
    return dense_index, dense_metas


def load_embed_model():
    import torch
    from sentence_transformers import SentenceTransformer
    print("Loading embedding model...")
    model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = 512
    print("  Loaded")
    return model


def retrieve_dense(queries, embed_model, dense_index, dense_metas, top_k=100):
    print(f"  Encoding {len(queries)} queries...")
    query_embs = embed_model.encode(
        queries, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  Searching FAISS top-{top_k}...")
    scores, indices = dense_index.search(query_embs, top_k)

    all_results = []
    for qi in range(len(queries)):
        hits = []
        for rank in range(top_k):
            idx = int(indices[qi, rank])
            score = float(scores[qi, rank])
            if idx < 0 or score <= 0:
                break
            hits.append({"dense_score": score, "dense_rank": rank, **dense_metas[idx]})
        all_results.append(hits)
    return all_results


def run_rerank_all(candidates=100, n_workers=8, source="bm25", max_candidates=40):
    content_cache = load_chunk_content_cache()

    questions = json.loads(R2AI_DATA.read_text(encoding="utf-8"))
    queries = [q["question"] for q in questions]
    print(f"Processing {len(questions)} questions, source={source}...")

    if source == "hybrid":
        bm25_model, bm25_metas = load_index()
        dense_index, dense_metas = load_dense_index()
        embed_model = load_embed_model()
        print(f"BM25S top-{candidates}...")
        bm25_results = retrieve_bm25s(queries, bm25_model, bm25_metas, top_k=candidates)
        print(f"Dense top-{candidates}...")
        dense_results = retrieve_dense(queries, embed_model, dense_index, dense_metas, top_k=candidates)
        print("Merging BM25+Dense candidates...")
        retrieval_results = []
        for qi in range(len(queries)):
            seen = set()
            merged = []
            for h in bm25_results[qi]:
                key = (h.get("law_id", ""), h.get("article_number", ""))
                if key not in seen:
                    seen.add(key)
                    merged.append(h)
            for h in dense_results[qi]:
                key = (h.get("law_id", ""), h.get("article_number", ""))
                if key not in seen:
                    seen.add(key)
                    merged.append(h)
            retrieval_results.append(merged)
        avg_merged = np.mean([len(r) for r in retrieval_results])
        print(f"  Avg merged candidates: {avg_merged:.1f}/query")
    elif source == "dense":
        dense_index, dense_metas = load_dense_index()
        embed_model = load_embed_model()
        print(f"Dense top-{candidates}...")
        retrieval_results = retrieve_dense(queries, embed_model, dense_index, dense_metas, top_k=candidates)
    else:
        bm25_model, metas = load_index()
        print(f"BM25S top-{candidates}...")
        retrieval_results = retrieve_bm25s(queries, bm25_model, metas, top_k=candidates)

    print(f"LLM reranking ({n_workers} workers, max_candidates={max_candidates})...")
    t0 = time.time()
    llm_results = llm_rerank_all(queries, retrieval_results, content_cache, n_workers=n_workers, max_candidates=max_candidates, questions=questions, source=source)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(queries)/elapsed:.1f} q/s)")

    cache_file = BASE / f"llm_rerank_cache_{source}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump({"questions": questions, "llm_results": llm_results}, f)
    print(f"  Cached to {cache_file}")

    avg_k = np.mean([len(r) for r in llm_results if r])
    print(f"  Avg selected: {avg_k:.1f} articles/query")


def run_submit(source="bm25"):
    cache_file = BASE / f"llm_rerank_cache_{source}.pkl"
    if not cache_file.exists():
        cache_file = CACHE_PATH
    if not cache_file.exists():
        print(f"ERROR: No cache found. Run 'rerank-all' first.")
        return

    print(f"Loading cache: {cache_file}")
    with open(cache_file, "rb") as f:
        cache = pickle.load(f)
    questions = cache["questions"]
    llm_results = cache["llm_results"]

    submission = [format_submission(q["id"], q["question"], r or [])
                  for q, r in zip(questions, llm_results)]

    suffix = f"_{source}" if source != "bm25" else ""
    out = BASE / f"submission_3_1_llm_rerank{suffix}.json"
    out.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_art = np.mean([len(s["relevant_articles"]) for s in submission])
    non_empty = sum(1 for s in submission if s["relevant_articles"])
    print(f"Saved to {out}")
    print(f"  Avg articles: {avg_art:.1f}, non-empty: {non_empty}/{len(submission)}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--limit", type=int, default=50)
    p_eval.add_argument("--candidates", type=int, default=20)

    p_cache = sub.add_parser("rerank-all")
    p_cache.add_argument("--candidates", type=int, default=100)
    p_cache.add_argument("--n-workers", type=int, default=16)
    p_cache.add_argument("--source", choices=["bm25", "dense", "hybrid"], default="bm25")
    p_cache.add_argument("--max-candidates", type=int, default=40, help="Max unique articles sent to LLM")

    p_sub = sub.add_parser("submit")
    p_sub.add_argument("--source", choices=["bm25", "dense", "hybrid"], default="bm25")

    args = parser.parse_args()

    if args.cmd == "evaluate":
        run_evaluate(args.limit, args.candidates)
    elif args.cmd == "rerank-all":
        run_rerank_all(args.candidates, args.n_workers, args.source, args.max_candidates)
    elif args.cmd == "submit":
        run_submit(args.source)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
