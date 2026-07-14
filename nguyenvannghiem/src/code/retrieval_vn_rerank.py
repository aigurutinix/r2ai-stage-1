#!/usr/bin/env python3
"""
BM25S + Vietnamese Reranker pipeline for R2AI Task 3.1.

2-stage: BM25S top-N candidates -> AITeamVN/Vietnamese_Reranker -> top-k results.

Usage:
    python3 retrieval_vn_rerank.py evaluate --limit 500 --candidates 50 --top-k 5
    python3 retrieval_vn_rerank.py evaluate --limit 10 --candidates 50 --top-k 5
    python3 retrieval_vn_rerank.py retrieve --candidates 50 --top-k 5
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

import numpy as np
import bm25s
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from underthesea import word_tokenize as vi_tokenize

BASE = Path("..") # adjust to your project root
CHUNKS_DIR = BASE / "vbpl_dataset" / "chunks"
DATA_FINAL = BASE / "data_final"
INDEX_DIR = BASE / "retrieval_index_bm25s_v7"
R2AI_DATA = BASE / "R2AIStage1DATA.json"

DOC_TYPES = [
    "hien_phap", "bo_luat", "luat", "phap_lenh",
    "nghi_dinh", "nghi_quyet", "nghi_quyet_lien_tich",
    "thong_tu", "thong_tu_lien_tich", "quyet_dinh",
]

DIEU_RE = re.compile(r'[Đđ]iều\s+(\d+)', re.IGNORECASE)
DOC_NUM_RE = re.compile(r'(\d+/\d{4}/[A-ZĐa-zđ\d\-]+)')

RERANKER_MODEL = "AITeamVN/Vietnamese_Reranker"

# Global content cache: chunk_id -> content string
_CONTENT_CACHE = {}


def normalize_text(text: str, use_vi_tok: bool = False) -> str:
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


def extract_dieu_numbers(text: str) -> set:
    return set(DIEU_RE.findall(text))


def extract_law_id(title: str) -> str:
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
    """Load full article content from chunk files into a global cache.
    Maps chunk_id -> content string for all VBPL chunks.
    """
    global _CONTENT_CACHE
    if _CONTENT_CACHE:
        return _CONTENT_CACHE

    print("Loading chunk content cache from VBPL files...")
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
                content_lines = chunk.get("content", [])
                content = "\n".join(content_lines).strip()
                if content:
                    _CONTENT_CACHE[cid] = content
                    count += 1

    print(f"  Loaded {count} chunk contents in {time.time()-t0:.1f}s")
    return _CONTENT_CACHE


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


def load_index():
    print("Loading BM25S index...")
    t0 = time.time()
    model = bm25s.BM25.load(str(INDEX_DIR / "bm25s_model"), load_corpus=False)
    with open(INDEX_DIR / "metas.pkl", "rb") as f:
        metas = pickle.load(f)
    print(f"  {len(metas)} chunks, loaded in {time.time()-t0:.1f}s")
    return model, metas


def retrieve_bm25s(queries, model, metas, top_k=50, batch_size=100):
    """Retrieve top-k candidates from BM25S for each query."""
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

        if end % 200 == 0 or end == len(queries):
            print(f"  BM25S: {end}/{len(queries)}", flush=True)

    return all_results


def build_passage_text(meta, content_cache, max_content_chars=1800):
    """Build passage text for reranking.

    For VBPL chunks: doc_title + doc_description + article_title + full article content
    For QA chunks: resolve to corresponding VBPL article content via van_ban_id + article_number
    Truncates content to fit within reranker's token limit.
    """
    parts = []
    doc_title = meta.get("doc_title", "").strip()
    doc_desc = meta.get("doc_description", "").strip()
    art_title = meta.get("article_title", "").strip()
    chunk_id = meta.get("chunk_id", "")

    if doc_title:
        parts.append(doc_title)
    if doc_desc:
        parts.append(doc_desc)
    if art_title:
        parts.append(art_title)

    # Try to get full content: direct lookup first, then resolve QA -> VBPL
    content = content_cache.get(chunk_id, "")
    if not content and chunk_id.startswith("qa_"):
        # Resolve QA chunk to corresponding VBPL article content
        van_ban_id = meta.get("van_ban_id", "")
        art_no = meta.get("article_number", "")
        if van_ban_id and art_no:
            vbpl_chunk_id = f"{van_ban_id}#dieu_{art_no}"
            content = content_cache.get(vbpl_chunk_id, "")

    if content:
        # Truncate to stay within model's 2048-token passage limit
        # ~4 chars per token for Vietnamese, leave room for titles
        parts.append(content[:max_content_chars])
    else:
        # Fallback: add article reference
        art_no = meta.get("article_number", "")
        if art_no and art_no != "0":
            dieu_ref = f"Điều {art_no}"
            if dieu_ref not in " ".join(parts):
                parts.append(dieu_ref)

    return " ".join(parts)


def load_reranker():
    """Load the Vietnamese reranker model."""
    print(f"Loading reranker: {RERANKER_MODEL}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL, torch_dtype=torch.float16
    )
    model = model.cuda().eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return tokenizer, model


def rerank_batch(queries, candidates_list, tokenizer, model, content_cache,
                 top_k=5, rerank_batch_size=32):
    """Rerank candidates for each query using the Vietnamese reranker.

    Args:
        queries: list of query strings
        candidates_list: list of list of candidate dicts (from BM25S)
        tokenizer: reranker tokenizer
        model: reranker model
        content_cache: chunk_id -> content mapping
        top_k: number of top results to return after reranking
        rerank_batch_size: batch size for reranker inference

    Returns:
        list of list of candidate dicts with reranker scores, sorted by score descending
    """
    all_reranked = []

    for qi, (query, candidates) in enumerate(zip(queries, candidates_list)):
        if not candidates:
            all_reranked.append([])
            continue

        # Build pairs for reranking
        passages = [build_passage_text(c, content_cache) for c in candidates]
        pairs = [[query, p] for p in passages]

        # Score in batches
        all_scores = []
        for bs in range(0, len(pairs), rerank_batch_size):
            be = min(bs + rerank_batch_size, len(pairs))
            batch_pairs = pairs[bs:be]

            with torch.no_grad():
                inputs = tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=2304,
                ).to('cuda')
                scores = model(**inputs, return_dict=True).logits.view(-1).float()
                all_scores.extend(scores.cpu().tolist())

        # Attach scores and sort
        for i, c in enumerate(candidates):
            c["rerank_score"] = all_scores[i]

        sorted_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        all_reranked.append(sorted_candidates[:top_k])

        if (qi + 1) % 50 == 0 or (qi + 1) == len(queries):
            print(f"  Reranked: {qi+1}/{len(queries)}", flush=True)

    return all_reranked


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
            gt_full = set()
            for c in cites:
                title = c.get("title", "")
                doc_title = c.get("doc_title", "")
                dieu_nums = extract_dieu_numbers(title)
                gt_dieu.update(dieu_nums)
                lid = extract_law_id(doc_title)
                for d in dieu_nums:
                    gt_full.add(f"{lid}|Điều {d}")

            if gt_dieu:
                qa_items.append({
                    "question": item["question"],
                    "gt_dieu": gt_dieu,
                    "gt_full": gt_full,
                })
        if limit and len(qa_items) >= limit:
            break
    return qa_items[:limit] if limit else qa_items


def compute_metrics(predictions, ground_truths, mode="dieu_only"):
    precisions, recalls = [], []

    for pred, gt in zip(predictions, ground_truths):
        pred_dieu = extract_dieu_numbers(pred.get("answer", ""))

        if mode == "dieu_only":
            gt_set = gt["gt_dieu"]
            pred_set = pred_dieu
        else:
            gt_set = gt["gt_full"]
            pred_set = set()
            for art in pred.get("relevant_articles", []):
                parts = art.split("|")
                if len(parts) >= 3:
                    lid = extract_law_id(parts[0]) or parts[0]
                    pred_set.add(f"{lid}|{parts[2]}")

        if not pred_set and not gt_set:
            precisions.append(1.0); recalls.append(1.0)
            continue
        if not pred_set:
            precisions.append(0.0); recalls.append(0.0)
            continue
        if not gt_set:
            precisions.append(0.0); recalls.append(1.0)
            continue

        tp = len(pred_set & gt_set)
        precisions.append(tp / len(pred_set))
        recalls.append(tp / len(gt_set))

    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    f2 = (5 * macro_p * macro_r) / (4 * macro_p + macro_r) if (4 * macro_p + macro_r) > 0 else 0.0

    return {"precision": macro_p, "recall": macro_r, "f2": f2,
            "num_queries": len(predictions),
            "avg_pred": np.mean([len(extract_dieu_numbers(p.get("answer", ""))) for p in predictions]),
            "avg_gt": np.mean([len(g["gt_dieu"]) for g in ground_truths])}


def run_evaluation(limit=500, candidates=50, top_k=5):
    bm25_model, metas = load_index()
    content_cache = load_chunk_content_cache()

    print(f"Loading ground truth (limit={limit})...")
    gt_data = load_ground_truth(limit)
    print(f"  {len(gt_data)} questions")

    queries = [g["question"] for g in gt_data]

    # Stage 1: BM25S retrieval
    print(f"\nStage 1: BM25S top-{candidates}...")
    t0 = time.time()
    bm25_results = retrieve_bm25s(queries, bm25_model, metas, top_k=candidates)
    bm25_time = time.time() - t0
    print(f"  BM25S done in {bm25_time:.1f}s")

    # Check content coverage for candidates (including QA->VBPL resolution)
    total_cands = sum(len(hits) for hits in bm25_results)
    with_content = 0
    for hits in bm25_results:
        for h in hits:
            cid = h["chunk_id"]
            if cid in content_cache:
                with_content += 1
            elif cid.startswith("qa_"):
                vbid = h.get("van_ban_id", "")
                art = h.get("article_number", "")
                if vbid and art and f"{vbid}#dieu_{art}" in content_cache:
                    with_content += 1
    print(f"  Content coverage: {with_content}/{total_cands} candidates ({100*with_content/max(total_cands,1):.1f}%)")

    # BM25S-only baselines
    print("\n" + "=" * 60)
    print("BM25S-only baselines (no reranking)")
    print("=" * 60)
    for bk in [5, 8, 10]:
        bm25_topk = [hits[:bk] for hits in bm25_results]
        preds_bm25 = [format_submission(i+1, g["question"], r)
                      for i, (g, r) in enumerate(zip(gt_data, bm25_topk))]
        for mode in ["dieu_only", "full"]:
            m = compute_metrics(preds_bm25, gt_data, mode)
            label = "Điều" if mode == "dieu_only" else "law+Điều"
            print(f"  BM25S top-{bk:2d} ({label}): P={m['precision']:.4f} R={m['recall']:.4f} F2={m['f2']:.4f}")

    # Stage 2: Reranking
    print(f"\nStage 2: Reranking top-{candidates} with Vietnamese_Reranker...")
    rerank_tokenizer, rerank_model = load_reranker()

    t0 = time.time()
    reranked_results = rerank_batch(
        queries, bm25_results, rerank_tokenizer, rerank_model, content_cache,
        top_k=max(top_k, 10),  # Get top-10 for sweep
        rerank_batch_size=16,
    )
    rerank_time = time.time() - t0
    print(f"  Reranking done in {rerank_time:.1f}s")

    # Reranked results (pure reranker ordering)
    print("\n" + "=" * 60)
    print(f"BM25S top-{candidates} -> Reranker results (pure reranker order)")
    print("=" * 60)
    for rk in [3, 5, 8, 10]:
        reranked_topk = [hits[:rk] for hits in reranked_results]
        preds_rerank = [format_submission(i+1, g["question"], r)
                        for i, (g, r) in enumerate(zip(gt_data, reranked_topk))]
        for mode in ["dieu_only", "full"]:
            m = compute_metrics(preds_rerank, gt_data, mode)
            label = "Điều" if mode == "dieu_only" else "law+Điều"
            print(f"  Reranked top-{rk:2d} ({label}): P={m['precision']:.4f} R={m['recall']:.4f} F2={m['f2']:.4f}")

    # Hybrid: weighted fusion of BM25S rank and reranker score
    print("\n" + "=" * 60)
    print("Hybrid fusion: alpha * normalized_bm25 + (1-alpha) * normalized_rerank")
    print("=" * 60)
    for alpha in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
        hybrid_results = []
        for hits in bm25_results:
            if not hits:
                hybrid_results.append([])
                continue
            # Normalize BM25S scores to [0, 1]
            bm25_scores = [h["bm25_score"] for h in hits]
            bm25_max = max(bm25_scores) if bm25_scores else 1
            bm25_min = min(bm25_scores) if bm25_scores else 0
            bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1

            # Normalize reranker scores to [0, 1]
            rerank_scores = [h.get("rerank_score", -10) for h in hits]
            rr_max = max(rerank_scores) if rerank_scores else 1
            rr_min = min(rerank_scores) if rerank_scores else 0
            rr_range = rr_max - rr_min if rr_max > rr_min else 1

            for h in hits:
                norm_bm25 = (h["bm25_score"] - bm25_min) / bm25_range
                norm_rr = (h.get("rerank_score", -10) - rr_min) / rr_range
                h["hybrid_score"] = alpha * norm_bm25 + (1 - alpha) * norm_rr

            sorted_hits = sorted(hits, key=lambda x: x["hybrid_score"], reverse=True)
            hybrid_results.append(sorted_hits)

        for rk in [5, 8]:
            hybrid_topk = [hits[:rk] for hits in hybrid_results]
            preds_hybrid = [format_submission(i+1, g["question"], r)
                            for i, (g, r) in enumerate(zip(gt_data, hybrid_topk))]
            for mode in ["dieu_only", "full"]:
                m = compute_metrics(preds_hybrid, gt_data, mode)
                label = "Điều" if mode == "dieu_only" else "law+Điều"
                print(f"  alpha={alpha:.1f} top-{rk} ({label}): P={m['precision']:.4f} R={m['recall']:.4f} F2={m['f2']:.4f}")

    # Reranker as filter: keep BM25S order but remove low-reranker-score items
    print("\n" + "-" * 60)
    print("Reranker as filter (BM25S order, filter by reranker score):")
    for fetch_k in [10, 15, 20]:
        bm25_topfetch = [hits[:fetch_k] for hits in bm25_results]
        for thresh in [-3.0, -2.0, -1.0, 0.0, 1.0]:
            filtered = []
            for hits in bm25_topfetch:
                kept = [h for h in hits if h.get("rerank_score", -10) >= thresh]
                if not kept and hits:
                    kept = [hits[0]]
                filtered.append(kept)
            preds_f = [format_submission(i+1, g["question"], r)
                       for i, (g, r) in enumerate(zip(gt_data, filtered))]
            m = compute_metrics(preds_f, gt_data, "dieu_only")
            avg_k = np.mean([len(r) for r in filtered])
            mf = compute_metrics(preds_f, gt_data, "full")
            print(f"  fetch={fetch_k:2d} thresh>={thresh:5.1f}: Điều P={m['precision']:.4f} R={m['recall']:.4f} F2={m['f2']:.4f} | law+Điều F2={mf['f2']:.4f} avg_k={avg_k:.1f}")

    # Show samples
    print("\n" + "-" * 60)
    print("Sample comparisons (BM25S vs Reranked):")
    for i in range(min(5, len(gt_data))):
        q = gt_data[i]["question"]
        gd = gt_data[i]["gt_dieu"]

        bm25_top5 = bm25_results[i][:5]
        rerank_top5 = reranked_results[i][:5]

        pred_bm25 = format_submission(i+1, q, bm25_top5)
        pred_rerank = format_submission(i+1, q, rerank_top5)

        pd_bm25 = extract_dieu_numbers(pred_bm25.get("answer", ""))
        pd_rerank = extract_dieu_numbers(pred_rerank.get("answer", ""))

        print(f"\n  Q{i+1}: {q[:80]}...")
        print(f"    GT Điều: {sorted(gd)}")
        print(f"    BM25S top-5 Điều: {sorted(pd_bm25)} (match: {sorted(pd_bm25 & gd)})")
        print(f"    Reranked top-5 Điều: {sorted(pd_rerank)} (match: {sorted(pd_rerank & gd)})")

        # Show reranker scores for top-5
        for j, h in enumerate(rerank_top5[:5]):
            art = h.get('article_number', '?')
            rs = h.get('rerank_score', 0)
            bs = h.get('bm25_score', 0)
            br = h.get('bm25_rank', '?')
            cid = h.get('chunk_id', '')
            has_content = "+" if cid in content_cache else "-"
            passage_len = len(build_passage_text(h, content_cache))
            print(f"      #{j+1} Điều {art} score={rs:.2f} (bm25_rank={br} bm25={bs:.1f}) content={has_content} len={passage_len}")

    print(f"\n  Total time: BM25S={bm25_time:.1f}s + Rerank={rerank_time:.1f}s = {bm25_time+rerank_time:.1f}s")


def apply_hybrid_fusion(bm25_results, alpha=0.7, top_k=5):
    """Combine BM25S and reranker scores with weighted fusion."""
    hybrid_results = []
    for hits in bm25_results:
        if not hits:
            hybrid_results.append([])
            continue
        # Normalize BM25S scores to [0, 1]
        bm25_scores = [h["bm25_score"] for h in hits]
        bm25_max = max(bm25_scores) if bm25_scores else 1
        bm25_min = min(bm25_scores) if bm25_scores else 0
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1

        # Normalize reranker scores to [0, 1]
        rerank_scores = [h.get("rerank_score", -10) for h in hits]
        rr_max = max(rerank_scores) if rerank_scores else 1
        rr_min = min(rerank_scores) if rerank_scores else 0
        rr_range = rr_max - rr_min if rr_max > rr_min else 1

        for h in hits:
            norm_bm25 = (h["bm25_score"] - bm25_min) / bm25_range
            norm_rr = (h.get("rerank_score", -10) - rr_min) / rr_range
            h["hybrid_score"] = alpha * norm_bm25 + (1 - alpha) * norm_rr

        sorted_hits = sorted(hits, key=lambda x: x["hybrid_score"], reverse=True)
        hybrid_results.append(sorted_hits[:top_k])
    return hybrid_results


CACHE_PATH = BASE / "rerank_cache.pkl"


def run_rerank_all(candidates=50):
    """Stage 1+2: BM25S + Rerank all 2000 queries, save scores to cache."""
    bm25_model, metas = load_index()
    content_cache = load_chunk_content_cache()

    questions = json.loads(R2AI_DATA.read_text(encoding="utf-8"))
    queries = [q["question"] for q in questions]
    print(f"Processing {len(questions)} questions...")

    print(f"\nStage 1: BM25S top-{candidates}...")
    t0 = time.time()
    bm25_results = retrieve_bm25s(queries, bm25_model, metas, top_k=candidates)
    print(f"  BM25S done in {time.time()-t0:.1f}s")

    print(f"\nStage 2: Reranking with Vietnamese_Reranker...")
    rerank_tokenizer, rerank_model = load_reranker()

    t0 = time.time()
    rerank_batch(
        queries, bm25_results, rerank_tokenizer, rerank_model, content_cache,
        top_k=candidates,
        rerank_batch_size=16,
    )
    print(f"  Reranking done in {time.time()-t0:.1f}s")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"questions": questions, "bm25_results": bm25_results}, f)
    print(f"  Cached to {CACHE_PATH}")


def run_submit(top_k=5, alpha=0.7, score_threshold=None):
    """Generate submission from cached rerank scores. Instant."""
    if not CACHE_PATH.exists():
        print(f"ERROR: No cache found at {CACHE_PATH}. Run 'rerank-all' first.")
        return

    print("Loading cached scores...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    questions = cache["questions"]
    bm25_results = cache["bm25_results"]

    print(f"Generating submission: alpha={alpha}, top_k={top_k}, threshold={score_threshold}")
    final_results = apply_hybrid_fusion(bm25_results, alpha=alpha, top_k=top_k)

    if score_threshold is not None:
        filtered = []
        for hits in final_results:
            kept = [h for h in hits if h.get("rerank_score", -10) >= score_threshold]
            if not kept and hits:
                kept = [hits[0]]
            filtered.append(kept)
        final_results = filtered
        avg_k = np.mean([len(r) for r in filtered])
        print(f"  After threshold >= {score_threshold}: avg {avg_k:.1f} results/query")

    submission = [format_submission(q["id"], q["question"], r)
                  for q, r in zip(questions, final_results)]

    out = BASE / "submission_3_1_vn_rerank.json"
    out.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_art = np.mean([len(s["relevant_articles"]) for s in submission])
    non_empty = sum(1 for s in submission if s["relevant_articles"])
    print(f"Saved to {out}")
    print(f"  Avg articles: {avg_art:.1f}, non-empty: {non_empty}/{len(submission)}")


def run_retrieval(candidates=50, top_k=5, alpha=0.7, score_threshold=None):
    """Full pipeline (for backward compat). Use rerank-all + submit instead."""
    run_rerank_all(candidates)
    run_submit(top_k, alpha, score_threshold)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--limit", type=int, default=500)
    p_eval.add_argument("--candidates", type=int, default=50, help="BM25S candidates per query")
    p_eval.add_argument("--top-k", type=int, default=5, help="Final top-k after reranking")

    p_ret = sub.add_parser("retrieve")
    p_ret.add_argument("--candidates", type=int, default=50)
    p_ret.add_argument("--top-k", type=int, default=5)
    p_ret.add_argument("--alpha", type=float, default=0.7, help="Hybrid fusion weight (BM25S)")
    p_ret.add_argument("--score-threshold", type=float, default=None)

    p_cache = sub.add_parser("rerank-all", help="BM25S + rerank all 2000 queries, save cache")
    p_cache.add_argument("--candidates", type=int, default=50)

    p_sub = sub.add_parser("submit", help="Generate submission from cached scores (instant)")
    p_sub.add_argument("--top-k", type=int, default=5)
    p_sub.add_argument("--alpha", type=float, default=0.7)
    p_sub.add_argument("--score-threshold", type=float, default=None)

    args = parser.parse_args()

    if args.cmd == "evaluate":
        run_evaluation(args.limit, args.candidates, args.top_k)
    elif args.cmd == "retrieve":
        run_retrieval(args.candidates, args.top_k, args.alpha, args.score_threshold)
    elif args.cmd == "rerank-all":
        run_rerank_all(args.candidates)
    elif args.cmd == "submit":
        run_submit(args.top_k, args.alpha, args.score_threshold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
