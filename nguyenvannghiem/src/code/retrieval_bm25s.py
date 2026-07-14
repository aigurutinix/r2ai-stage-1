#!/usr/bin/env python3
"""
BM25S-based legal article retrieval for R2AI Task 3.1.

Usage:
    python3 retrieval_bm25s.py build-index
    python3 retrieval_bm25s.py evaluate --limit 500 --top-k 10
    python3 retrieval_bm25s.py retrieve --top-k 20
    python3 retrieval_bm25s.py all --limit 500 --top-k 20
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
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import bm25s
from underthesea import word_tokenize as vi_tokenize

BASE = Path("..") # adjust to your project root
CHUNKS_DIR = BASE / "vbpl_dataset" / "chunks"
META_DIR = BASE / "vbpl_dataset" / "metadata"
DATA_FINAL = BASE / "data_final"
SYNTHETIC_DIR = BASE / "synthetic_qa"
INDEX_DIR = BASE / "retrieval_index_bm25s_v7"
R2AI_DATA = BASE / "R2AIStage1DATA.json"

DOC_TYPES = [
    "hien_phap", "bo_luat", "luat", "phap_lenh",
    "nghi_dinh", "nghi_quyet", "nghi_quyet_lien_tich",
    "thong_tu", "thong_tu_lien_tich", "quyet_dinh",
]

DIEU_RE = re.compile(r'[Đđ]iều\s+(\d+)', re.IGNORECASE)
DOC_NUM_RE = re.compile(r'(\d+/\d{4}/[A-ZĐa-zđ\d\-]+)')


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


USE_VI_TOK = True


def _vi_tok_batch(batch):
    results = []
    for t in batch:
        try:
            results.append(vi_tokenize(t, format='text'))
        except Exception:
            results.append(t)
    return results


def extract_dieu_numbers(text: str) -> set:
    return set(DIEU_RE.findall(text))


def extract_law_id(title: str) -> str:
    m = DOC_NUM_RE.search(title)
    return m.group(1) if m else ""


def load_metadata() -> dict:
    meta_map = {}
    for dtype in DOC_TYPES:
        meta_file = META_DIR / f"{dtype}.json"
        if not meta_file.exists():
            continue
        docs = json.loads(meta_file.read_text(encoding="utf-8"))
        for doc in docs:
            vb_id = f"{dtype}/{doc['item_id']}"
            meta_map[vb_id] = {
                "item_id": doc["item_id"],
                "doc_type": dtype,
                "title": doc.get("title", ""),
                "description": doc.get("description", ""),
                "hieu_luc": doc.get("hieu_luc", ""),
            }
    return meta_map


def load_all_chunks(meta_map: dict):
    texts = []
    metas = []
    doc_count = 0

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
            for chunk in data.get("chunks", []):
                article_no = chunk.get("article_number", "")
                article_title = chunk.get("article_title", "")
                content = chunk.get("content", [])
                noi_dung = "\n".join(content)

                if not noi_dung.strip() and not article_title.strip():
                    continue

                doc_title = vb_meta.get("title", "")
                doc_desc = vb_meta.get("description", "")
                law_id = extract_law_id(doc_title)

                full_text = f"{doc_title} {doc_desc} {article_title} {noi_dung}".strip()
                texts.append(normalize_text(full_text, use_vi_tok=False))

                metas.append({
                    "chunk_id": chunk.get("chunk_id", ""),
                    "article_number": article_no,
                    "article_title": article_title,
                    "van_ban_id": van_ban_id,
                    "law_id": law_id,
                    "doc_title": doc_title,
                    "doc_description": doc_desc,
                    "hieu_luc": vb_meta.get("hieu_luc", ""),
                })

                if len(texts) % 10000 == 0:
                    print(f"  {doc_count} docs, {len(texts)} chunks...", flush=True)

            doc_count += 1

    print(f"  VBPL: {doc_count} docs, {len(texts)} chunks")
    return texts, metas


def load_qa_chunks(meta_map: dict):
    """Load QA data as additional chunks — bridges colloquial language to legal articles."""
    cite_map = {}
    cite_map_path = DATA_FINAL / "cite_mapping.json"
    if cite_map_path.exists():
        cite_map = json.loads(cite_map_path.read_text(encoding="utf-8"))

    texts = []
    metas = []
    qa_count = 0
    skipped = 0
    seen_keys = set()

    qa_files = sorted(DATA_FINAL.rglob("qa_*.json"))
    for fpath in qa_files:
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]

        for item in items:
            question = item.get("question", "")
            answer = item.get("answer", "")
            cites = item.get("article_cite", [])
            if not question or not cites:
                skipped += 1
                continue

            for cite in cites:
                if isinstance(cite, str):
                    cite = {"title": cite, "content": ""}
                title = cite.get("title", "")
                dieu_nums = extract_dieu_numbers(title)
                if not dieu_nums:
                    continue

                item_id = cite.get("item_id", "")
                doc_title = cite.get("doc_title", "")
                hieu_luc = cite.get("hieu_luc", "")

                if not item_id and title in cite_map:
                    mapped = cite_map[title]
                    item_id = mapped.get("item_id", "")
                    doc_title = mapped.get("title", "")
                    hieu_luc = mapped.get("hieu_luc", "")
                elif not item_id:
                    for cm_key, cm_val in cite_map.items():
                        if title in cm_key or cm_key in title:
                            item_id = cm_val.get("item_id", "")
                            doc_title = cm_val.get("title", "")
                            hieu_luc = cm_val.get("hieu_luc", "")
                            break

                law_id = extract_law_id(doc_title) if doc_title else ""
                art_no = sorted(dieu_nums)[0]

                dedup_key = (question.strip().lower(), item_id, art_no)
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                cite_content = cite.get("content", "")

                qa_text = f"{question} {answer} {cite_content} {title}".strip()
                texts.append(normalize_text(qa_text, use_vi_tok=False))

                van_ban_id = ""
                for dtype in DOC_TYPES:
                    candidate = f"{dtype}/{item_id}"
                    if candidate in meta_map:
                        van_ban_id = candidate
                        break

                metas.append({
                    "chunk_id": f"qa_{qa_count}_{art_no}",
                    "article_number": art_no,
                    "article_title": title,
                    "van_ban_id": van_ban_id,
                    "law_id": law_id,
                    "doc_title": doc_title,
                    "doc_description": "",
                    "hieu_luc": hieu_luc,
                })
                qa_count += 1

        if len(texts) % 50000 < 500:
            print(f"  QA: {len(texts)} chunks...", flush=True)

    print(f"  QA: {len(texts)} chunks from {len(qa_files)} files (skipped {skipped})")
    return texts, metas


def load_synthetic_qa_chunks(meta_map: dict):
    """Load synthetic QA data as additional chunks for BM25S."""
    texts = []
    metas = []
    seen_questions = set()
    count = 0
    dupes = 0

    for fpath in sorted(SYNTHETIC_DIR.glob("synthetic_*.json")):
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            continue

        for item in data:
            question = item.get("question", "")
            answer = item.get("answer", "")
            doc_title = item.get("doc_title", "")
            doc_desc = item.get("doc_description", "")
            art_no = item.get("article_number", "")
            art_title = item.get("article_title", "")
            van_ban_id = item.get("van_ban_id", "")
            law_id = item.get("law_id", "") or extract_law_id(doc_title)
            chunk_id = item.get("chunk_id", "")

            if not question or not art_no:
                continue

            q_key = question.strip()
            if q_key in seen_questions:
                dupes += 1
                continue
            seen_questions.add(q_key)

            syn_text = f"{question} {answer} {art_title} {doc_title}".strip()
            texts.append(normalize_text(syn_text, use_vi_tok=False))

            metas.append({
                "chunk_id": chunk_id or f"syn_{count}_{art_no}",
                "article_number": art_no,
                "article_title": art_title,
                "van_ban_id": van_ban_id,
                "law_id": law_id,
                "doc_title": doc_title,
                "doc_description": doc_desc,
                "hieu_luc": "",
            })
            count += 1

    print(f"  Synthetic QA: {len(texts)} chunks ({dupes} dupes removed) from {len(list(SYNTHETIC_DIR.glob('synthetic_*.json')))} files")
    return texts, metas


def build_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading metadata...")
    meta_map = load_metadata()
    print(f"  {len(meta_map)} documents")

    print("Loading VBPL chunks...")
    texts, metas = load_all_chunks(meta_map)

    print("Loading QA chunks...")
    qa_texts, qa_metas = load_qa_chunks(meta_map)
    texts.extend(qa_texts)
    metas.extend(qa_metas)

    print("Loading Synthetic QA chunks...")
    syn_texts, syn_metas = load_synthetic_qa_chunks(meta_map)
    texts.extend(syn_texts)
    metas.extend(syn_metas)
    print(f"  Combined: {len(texts)} total chunks")

    if USE_VI_TOK:
        print("Vietnamese tokenization (parallel)...")
        t0 = time.time()
        n_workers = min(os.cpu_count() or 4, 16)
        batch_size = max(1, len(texts) // (n_workers * 4))
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        print(f"  {n_workers} workers, {len(batches)} batches of ~{batch_size}")

        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_vi_tok_batch, b): i for i, b in enumerate(batches)}
            result_map = {}
            for future in as_completed(futures):
                idx = futures[future]
                result_map[idx] = future.result()
                done += 1
                if done % 10 == 0 or done == len(batches):
                    print(f"  vi_tokenize: {done}/{len(batches)} batches", flush=True)

        texts = []
        for i in range(len(batches)):
            texts.extend(result_map[i])
        print(f"  vi_tokenize done in {time.time()-t0:.1f}s")

    print("Tokenizing for BM25S...")
    t0 = time.time()
    corpus_tokens = bm25s.tokenize(
        texts,
        lower=True,
        stopwords=None,
        token_pattern=r'(?u)\b\w\w+\b',
        return_ids=True,
        show_progress=True,
    )
    print(f"  Tokenized in {time.time()-t0:.1f}s, vocab size: {len(corpus_tokens.vocab)}")

    print("Building BM25S index (method=lucene)...")
    t0 = time.time()
    model = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
    model.index(corpus_tokens)
    print(f"  Indexed in {time.time()-t0:.1f}s")

    print("Saving index...")
    model.save(str(INDEX_DIR / "bm25s_model"), corpus=corpus_tokens)
    with open(INDEX_DIR / "metas.pkl", "wb") as f:
        pickle.dump(metas, f)

    print(f"Index saved to {INDEX_DIR}")
    print(f"  {len(texts)} chunks indexed")


def load_index():
    print("Loading BM25S index...")
    t0 = time.time()
    model = bm25s.BM25.load(str(INDEX_DIR / "bm25s_model"), load_corpus=False)
    with open(INDEX_DIR / "metas.pkl", "rb") as f:
        metas = pickle.load(f)
    print(f"  {len(metas)} chunks, loaded in {time.time()-t0:.1f}s")
    return model, metas


def retrieve_batch(queries, model, metas, top_k=20, batch_size=100):
    normalized = [normalize_text(q, USE_VI_TOK) for q in queries]
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
                hits.append({"score": score, **metas[idx]})
            all_results.append(hits)

        if end % 200 == 0 or end == len(queries):
            print(f"  {end}/{len(queries)}", flush=True)

    return all_results


def apply_score_threshold(all_results, threshold_ratio=0.3, min_k=1, max_k=10):
    filtered = []
    for hits in all_results:
        if not hits:
            filtered.append([])
            continue
        max_score = hits[0]["score"]
        if max_score <= 0:
            filtered.append([])
            continue
        cutoff = max_score * threshold_ratio
        kept = [h for h in hits if h["score"] >= cutoff]
        kept = kept[:max_k] if len(kept) > max_k else kept
        if len(kept) < min_k:
            kept = hits[:min_k]
        filtered.append(kept)
    return filtered


def make_doc_name(doc_title, doc_description):
    """Format: Loại văn bản + Mã văn bản + Trích yếu.
    doc_title is like 'Luật 04/2017/QH14', doc_description is the trích yếu.
    Result: 'Luật 04/2017/QH14 Hỗ trợ doanh nghiệp nhỏ và vừa'
    """
    title = doc_title.strip()
    desc = doc_description.strip()
    if desc and title:
        if desc.lower().startswith(title.split()[0].lower()):
            return desc
        return f"{title} {desc}"
    return title or desc


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


def run_evaluation(limit=500, top_k=10):
    model, metas = load_index()

    print(f"Loading ground truth (limit={limit})...")
    gt_data = load_ground_truth(limit)
    print(f"  {len(gt_data)} questions")

    print("Retrieving...")
    t0 = time.time()
    queries = [g["question"] for g in gt_data]
    all_results = retrieve_batch(queries, model, metas, top_k=top_k)

    predictions = [format_submission(i+1, g["question"], r)
                   for i, (g, r) in enumerate(zip(gt_data, all_results))]
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\n" + "=" * 60)
    print(f"EVALUATION — Task 3.1 Retrieval (BM25S, top_k={top_k})")
    print("=" * 60)

    for mode in ["dieu_only", "full"]:
        m = compute_metrics(predictions, gt_data, mode)
        label = "Điều-number only" if mode == "dieu_only" else "law_id+Điều (strict)"
        print(f"\n  {label}:")
        print(f"    Precision : {m['precision']:.4f}")
        print(f"    Recall    : {m['recall']:.4f}")
        print(f"    F2        : {m['f2']:.4f}")
        print(f"    Avg pred  : {m['avg_pred']:.1f}  |  Avg GT: {m['avg_gt']:.1f}")

    print("\n" + "-" * 60)
    print("Samples:")
    for i in range(min(5, len(predictions))):
        p, g = predictions[i], gt_data[i]
        pd = extract_dieu_numbers(p.get("answer", ""))
        gd = g["gt_dieu"]
        print(f"  Q{i+1}: {g['question'][:80]}...")
        print(f"    Pred: {sorted(pd)}  GT: {sorted(gd)}  Match: {sorted(pd&gd)}")

    print("\n" + "-" * 60)
    print("Top-k sweep (Điều-number only):")
    for k in [3, 5, 10, 15, 20, 30]:
        if k != top_k:
            res_k = retrieve_batch(queries, model, metas, top_k=k)
            preds_k = [format_submission(i+1, g["question"], r)
                       for i, (g, r) in enumerate(zip(gt_data, res_k))]
        else:
            preds_k = predictions
        mk = compute_metrics(preds_k, gt_data, "dieu_only")
        print(f"  k={k:2d}  P={mk['precision']:.4f}  R={mk['recall']:.4f}  F2={mk['f2']:.4f}")

    print("\n" + "-" * 60)
    print("Score threshold sweep (fetch top-20, Điều-number only):")
    res_20 = retrieve_batch(queries, model, metas, top_k=20)
    for ratio in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
        for max_k in [5, 8, 10]:
            filtered = apply_score_threshold(res_20, threshold_ratio=ratio, min_k=1, max_k=max_k)
            preds_t = [format_submission(i+1, g["question"], r)
                       for i, (g, r) in enumerate(zip(gt_data, filtered))]
            mt = compute_metrics(preds_t, gt_data, "dieu_only")
            avg_k = np.mean([len(r) for r in filtered])
            print(f"  ratio={ratio:.2f} max_k={max_k:2d}  P={mt['precision']:.4f}  R={mt['recall']:.4f}  F2={mt['f2']:.4f}  avg_k={avg_k:.1f}")


def run_retrieval(top_k=20, threshold_ratio=None, max_k=None):
    model, metas = load_index()

    questions = json.loads(R2AI_DATA.read_text(encoding="utf-8"))
    print(f"Retrieving {len(questions)} questions...")

    fetch_k = max(top_k, 20) if threshold_ratio else top_k
    t0 = time.time()
    queries = [q["question"] for q in questions]
    all_results = retrieve_batch(queries, model, metas, top_k=fetch_k)

    if threshold_ratio:
        mk = max_k or top_k
        print(f"  Applying score threshold: ratio={threshold_ratio}, max_k={mk}")
        all_results = apply_score_threshold(all_results, threshold_ratio, min_k=1, max_k=mk)
        avg_k = np.mean([len(r) for r in all_results])
        print(f"  After threshold: avg {avg_k:.1f} results/query")

    submission = [format_submission(q["id"], q["question"], r)
                  for q, r in zip(questions, all_results)]
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(questions)/elapsed:.1f} q/s)")

    out = BASE / "submission_3_1_bm25s.json"
    out.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {out}")

    avg_art = np.mean([len(s["relevant_articles"]) for s in submission])
    non_empty = sum(1 for s in submission if s["relevant_articles"])
    print(f"  Avg articles: {avg_art:.1f}, non-empty: {non_empty}/{len(submission)}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("build-index")

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--limit", type=int, default=500)
    p_eval.add_argument("--top-k", type=int, default=10)

    p_ret = sub.add_parser("retrieve")
    p_ret.add_argument("--top-k", type=int, default=20)
    p_ret.add_argument("--threshold", type=float, default=None, help="Score ratio threshold (e.g. 0.7)")
    p_ret.add_argument("--max-k", type=int, default=None, help="Max results after threshold")

    p_all = sub.add_parser("all")
    p_all.add_argument("--limit", type=int, default=500)
    p_all.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()

    if args.cmd == "build-index":
        build_index()
    elif args.cmd == "evaluate":
        run_evaluation(args.limit, args.top_k)
    elif args.cmd == "retrieve":
        run_retrieval(args.top_k, threshold_ratio=args.threshold, max_k=args.max_k)
    elif args.cmd == "all":
        if not (INDEX_DIR / "bm25s_model").exists():
            build_index()
        run_evaluation(args.limit, args.top_k)
        run_retrieval(args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
