#!/usr/bin/env python3
"""
Dense retrieval for R2AI Task 3.1 using AITeamVN/Vietnamese_Embedding_v2.

Usage:
    python3 retrieval_dense.py build-index
    python3 retrieval_dense.py evaluate --limit 500 --top-k 5
    python3 retrieval_dense.py retrieve --top-k 5
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
import faiss

BASE = Path("..") # adjust to your project root
CHUNKS_DIR = BASE / "vbpl_dataset" / "chunks"
META_DIR = BASE / "vbpl_dataset" / "metadata"
DATA_FINAL = BASE / "data_final"
INDEX_DIR = BASE / "retrieval_index_dense"
SYNTHETIC_DIR = BASE / "synthetic_qa"
R2AI_DATA = BASE / "R2AIStage1DATA.json"

DOC_TYPES = [
    "hien_phap", "bo_luat", "luat", "phap_lenh",
    "nghi_dinh", "nghi_quyet", "nghi_quyet_lien_tich",
    "thong_tu", "thong_tu_lien_tich", "quyet_dinh",
]

DIEU_RE = re.compile(r'[Đđ]iều\s+(\d+)', re.IGNORECASE)
DOC_NUM_RE = re.compile(r'(\d+/\d{4}/[A-ZĐa-zđ\d\-]+)')

MODEL_NAME = "AITeamVN/Vietnamese_Embedding_v2"
EMBEDDING_DIM = 1024


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
                texts.append(full_text)

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

                if len(texts) % 50000 == 0:
                    print(f"  {doc_count} docs, {len(texts)} chunks...", flush=True)

            doc_count += 1

    print(f"  VBPL: {doc_count} docs, {len(texts)} chunks")
    return texts, metas


def load_qa_chunks(meta_map: dict):
    """Load QA data as additional chunks."""
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
                texts.append(qa_text)

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

    print(f"  QA: {len(texts)} chunks from {len(qa_files)} files (skipped {skipped})")
    return texts, metas


def encode_texts(model, texts, batch_size=256, desc="Encoding"):
    """Encode texts in batches using SentenceTransformer on GPU."""
    print(f"  {desc}: {len(texts)} texts, batch_size={batch_size}")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize for cosine sim via dot product
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    print(f"  Encoded in {elapsed:.1f}s ({len(texts)/elapsed:.0f} texts/s)")
    return embeddings.astype(np.float32)


def load_synthetic_qa_chunks(meta_map: dict):
    """Load deduped synthetic QA as additional chunks for dense index."""
    texts = []
    metas = []
    seen = set()
    dupes = 0

    for fpath in sorted(SYNTHETIC_DIR.glob("synthetic_*.json")):
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in data:
            question = item.get("question", "")
            answer = item.get("answer", "")
            art_no = item.get("article_number", "")
            if not question or not art_no:
                continue
            q_key = question.strip()
            if q_key in seen:
                dupes += 1
                continue
            seen.add(q_key)

            doc_title = item.get("doc_title", "")
            art_title = item.get("article_title", "")
            van_ban_id = item.get("van_ban_id", "")
            law_id = item.get("law_id", "")
            chunk_id = item.get("chunk_id", "")

            syn_text = f"{question} {answer} {art_title} {doc_title}".strip()
            texts.append(syn_text)
            metas.append({
                "chunk_id": chunk_id or f"syn_{len(texts)}_{art_no}",
                "article_number": art_no,
                "article_title": art_title,
                "van_ban_id": van_ban_id,
                "law_id": law_id,
                "doc_title": doc_title,
                "doc_description": item.get("doc_description", ""),
                "hieu_luc": "",
            })

    print(f"  Synthetic QA: {len(texts)} chunks ({dupes} dupes removed)")
    return texts, metas


def build_index(batch_size=64):
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

    # Synthetic QA bỏ — đã test dense v2, thêm noise cho dense
    print(f"  Combined: {len(texts)} total chunks")

    # Check if embeddings already exist on disk
    emb_path = INDEX_DIR / "embeddings.npy"
    meta_path = INDEX_DIR / "metas.pkl"
    texts_count_path = INDEX_DIR / "texts_count.txt"

    need_encode = True
    if emb_path.exists() and texts_count_path.exists():
        saved_count = int(texts_count_path.read_text().strip())
        if saved_count == len(texts):
            print(f"  Found cached embeddings for {saved_count} texts, loading...")
            embeddings = np.load(str(emb_path))
            need_encode = False
        else:
            print(f"  Cached embeddings count mismatch ({saved_count} vs {len(texts)}), re-encoding...")

    if need_encode:
        print("Loading embedding model...")
        import torch
        from sentence_transformers import SentenceTransformer
        # Clear GPU cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = SentenceTransformer(MODEL_NAME, model_kwargs={"torch_dtype": torch.float16})
        model.max_seq_length = 4096  # Limit seq length for memory efficiency
        print(f"  Model loaded: {MODEL_NAME} (fp16), dim={model.get_embedding_dimension()}, max_seq={model.max_seq_length}")

        # Encode in chunks and save progressively
        chunk_size = 50000
        all_embeddings = []

        # Check for partial progress
        partial_path = INDEX_DIR / "embeddings_partial.npy"
        partial_count_path = INDEX_DIR / "partial_count.txt"
        start_chunk = 0
        if partial_path.exists() and partial_count_path.exists():
            start_chunk = int(partial_count_path.read_text().strip())
            if start_chunk > 0 and start_chunk <= len(texts):
                print(f"  Resuming from chunk {start_chunk} (loading partial embeddings)...")
                partial_emb = np.load(str(partial_path))
                all_embeddings.append(partial_emb)

        for i in range(start_chunk, len(texts), chunk_size):
            end = min(i + chunk_size, len(texts))
            total_batches = (len(texts) - 1) // chunk_size + 1
            cur_batch = i // chunk_size + 1
            print(f"\n  Encoding chunk {cur_batch}/{total_batches} ({i}-{end})...")
            batch_emb = encode_texts(model, texts[i:end], batch_size=batch_size, desc=f"Chunk {cur_batch}")
            all_embeddings.append(batch_emb)

            # Save partial progress
            partial = np.vstack(all_embeddings)
            np.save(str(partial_path), partial)
            partial_count_path.write_text(str(end))
            print(f"  Saved partial embeddings: {partial.shape}")

        embeddings = np.vstack(all_embeddings)
        np.save(str(emb_path), embeddings)
        texts_count_path.write_text(str(len(texts)))
        # Clean up partial files
        if partial_path.exists():
            partial_path.unlink()
        if partial_count_path.exists():
            partial_count_path.unlink()
        print(f"  Saved embeddings: {embeddings.shape}")

    # Build FAISS index (IndexFlatIP for dot product / cosine similarity with normalized vectors)
    print("Building FAISS index...")
    t0 = time.time()
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    print(f"  FAISS index built in {time.time()-t0:.1f}s, {index.ntotal} vectors")

    # Save index and metas
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(meta_path, "wb") as f:
        pickle.dump(metas, f)

    print(f"Index saved to {INDEX_DIR}")
    print(f"  {len(texts)} chunks indexed, dim={EMBEDDING_DIM}")


def load_index():
    print("Loading FAISS index...")
    t0 = time.time()
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "metas.pkl", "rb") as f:
        metas = pickle.load(f)
    print(f"  {index.ntotal} vectors, {len(metas)} metas, loaded in {time.time()-t0:.1f}s")
    return index, metas


def load_model():
    print("Loading embedding model...")
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = 4096
    print(f"  Model loaded: {MODEL_NAME} (fp16)")
    return model


def retrieve_batch(queries, model, index, metas, top_k=10, batch_size=64):
    """Encode queries and search FAISS index."""
    print(f"  Encoding {len(queries)} queries...")
    t0 = time.time()
    query_embeddings = model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  Queries encoded in {time.time()-t0:.1f}s")

    print(f"  Searching FAISS index (top_k={top_k})...")
    t0 = time.time()
    scores, indices = index.search(query_embeddings, top_k)
    print(f"  Search done in {time.time()-t0:.1f}s")

    all_results = []
    for i in range(len(queries)):
        hits = []
        for j in range(top_k):
            idx = int(indices[i, j])
            score = float(scores[i, j])
            if idx < 0 or idx >= len(metas):
                continue
            hits.append({"score": score, **metas[idx]})
        all_results.append(hits)

    return all_results


def make_doc_name(doc_title, doc_description):
    title = doc_title.strip()
    desc = doc_description.strip()
    if desc and title:
        if desc.lower().startswith(title.split()[0].lower()):
            return desc
        return f"{title} {desc}"
    return title or desc


def format_submission(qid, question, results, max_articles=100):
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
        if len(relevant_articles) >= max_articles:
            break

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
                if isinstance(c, str):
                    c = {"title": c, "content": ""}
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


def run_evaluation(limit=500, top_k=5):
    index, metas = load_index()
    model = load_model()

    print(f"Loading ground truth (limit={limit})...")
    gt_data = load_ground_truth(limit)
    print(f"  {len(gt_data)} questions")

    print("Retrieving...")
    t0 = time.time()
    queries = [g["question"] for g in gt_data]
    all_results = retrieve_batch(queries, model, index, metas, top_k=top_k)

    predictions = [format_submission(i+1, g["question"], r)
                   for i, (g, r) in enumerate(zip(gt_data, all_results))]
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\n" + "=" * 60)
    print(f"EVALUATION — Task 3.1 Retrieval (Dense, top_k={top_k})")
    print("=" * 60)

    for mode in ["dieu_only", "full"]:
        m = compute_metrics(predictions, gt_data, mode)
        label = "Articles (Dieu-number only)" if mode == "dieu_only" else "law_id+Dieu (strict)"
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

    # Top-k sweep
    print("\n" + "-" * 60)
    print("Top-k sweep (Dieu-number only):")
    for k in [3, 5, 8, 10, 15, 20]:
        if k != top_k:
            res_k = retrieve_batch(queries, model, index, metas, top_k=k)
            preds_k = [format_submission(i+1, g["question"], r)
                       for i, (g, r) in enumerate(zip(gt_data, res_k))]
        else:
            preds_k = predictions
        mk = compute_metrics(preds_k, gt_data, "dieu_only")
        print(f"  k={k:2d}  P={mk['precision']:.4f}  R={mk['recall']:.4f}  F2={mk['f2']:.4f}")


def run_retrieval(top_k=5, fetch_k=None):
    if fetch_k is None:
        fetch_k = top_k * 2
    index, metas = load_index()
    model = load_model()

    questions = json.loads(R2AI_DATA.read_text(encoding="utf-8"))
    print(f"Retrieving {len(questions)} questions (fetch={fetch_k}, cap={top_k})...")

    t0 = time.time()
    queries = [q["question"] for q in questions]
    all_results = retrieve_batch(queries, model, index, metas, top_k=fetch_k)

    submission = [format_submission(q["id"], q["question"], r)
                  for q, r in zip(questions, all_results)]
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(questions)/elapsed:.1f} q/s)")

    out = BASE / "submission_3_1_dense.json"
    out.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {out}")

    avg_art = np.mean([len(s["relevant_articles"]) for s in submission])
    non_empty = sum(1 for s in submission if s["relevant_articles"])
    print(f"  Avg articles: {avg_art:.1f}, non-empty: {non_empty}/{len(submission)}")


def main():
    parser = argparse.ArgumentParser(description="Dense retrieval for R2AI Task 3.1")
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build-index")
    p_build.add_argument("--batch-size", type=int, default=64)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--limit", type=int, default=500)
    p_eval.add_argument("--top-k", type=int, default=5)

    p_ret = sub.add_parser("retrieve")
    p_ret.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.cmd == "build-index":
        build_index(batch_size=args.batch_size)
    elif args.cmd == "evaluate":
        run_evaluation(args.limit, args.top_k)
    elif args.cmd == "retrieve":
        run_retrieval(args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
