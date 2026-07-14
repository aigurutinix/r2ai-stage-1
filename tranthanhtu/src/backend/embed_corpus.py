"""Embed corpus_articles.jsonl → corpus_emb.npy (float32), INCREMENTAL.

Idempotent: re-run embeds only the NEW articles appended since last run.
- If corpus_emb.npy + corpus_emb_ids.json exist and their ids are a clean prefix
  of the current jsonl ids → embed only the tail and append.
- Otherwise (first run, or jsonl rebuilt/reordered) → embed everything.

Run:  python embed_corpus.py
"""
import json
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

import local_models_config as cfg

BATCH = 64


def _load_corpus():
    rows = []
    with open(cfg.CORPUS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and json.loads(line).get("doc_number"):
                rows.append(json.loads(line))
    return rows


def embed():
    if not os.path.exists(cfg.CORPUS_JSONL):
        sys.exit(f"Corpus not found: {cfg.CORPUS_JSONL}\nRun build_corpus.py first.")

    rows = _load_corpus()
    ids = [r["id"] for r in rows]
    docs = [f"{r['title']}\n{r['text']}" for r in rows]
    print(f"corpus: {len(rows)} articles")

    # Determine what's already embedded (incremental)
    old_vecs, start = None, 0
    if os.path.exists(cfg.CORPUS_EMB) and os.path.exists(cfg.CORPUS_EMB_IDS):
        saved_ids = json.load(open(cfg.CORPUS_EMB_IDS, encoding="utf-8"))
        old = np.load(cfg.CORPUS_EMB)
        if len(saved_ids) <= len(ids) and ids[:len(saved_ids)] == saved_ids and len(old) == len(saved_ids):
            old_vecs, start = old, len(saved_ids)
            print(f"đã embed {start} điều → chỉ embed {len(ids) - start} điều MỚI")
        else:
            print("ids lệch / corpus dựng lại → embed lại TOÀN BỘ")

    if start >= len(ids):
        print("Không có điều mới. corpus_emb.npy đã cập nhật.")
        return

    device = cfg.get_device()
    print(f"Loading '{cfg.EMBEDDING_MODEL}' on {device} ...")
    model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=device)
    model.max_seq_length = cfg.EMBED_MAX_SEQ_LEN

    new_vecs = model.encode(docs[start:], batch_size=BATCH, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=True).astype("float32")
    vecs = new_vecs if old_vecs is None else np.vstack([old_vecs, new_vecs])

    np.save(cfg.CORPUS_EMB, vecs)
    json.dump(ids, open(cfg.CORPUS_EMB_IDS, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nDone. corpus_emb.npy = {vecs.shape} (embed {len(ids) - start} điều mới).")


if __name__ == "__main__":
    embed()
