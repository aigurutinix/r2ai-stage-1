#!/usr/bin/env python3
"""
Build FAISS dense index on data_final questions for few-shot lookup.
Each vector = embedded question, metadata = question + article_cite.

Usage:
    python3 build_fewshot_index.py
Output:
    fewshot_index/  (FAISS index + metadata)
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE = Path("..") # adjust to your project root
OUT_DIR = BASE / "fewshot_index"
MODEL_PATH = "../data/retrieval_index_dense" # adjust to your project root  # same embedding model


def load_all_qa():
    """Load all QA entries from data_final with usable article_cite."""
    data_dir = BASE / "data_final"
    qa_entries = []

    for source_dir in sorted(data_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        if source_dir.name in ("__pycache__",):
            continue
        for topic_dir in sorted(source_dir.iterdir()):
            if not topic_dir.is_dir():
                continue
            for fpath in sorted(topic_dir.glob("*.json")):
                try:
                    data = json.loads(fpath.read_text(encoding="utf-8"))
                    if not isinstance(data, list):
                        continue
                    for item in data:
                        q = item.get("question", "").strip()
                        cites = item.get("article_cite", [])
                        if not q or not cites:
                            continue

                        # Normalize cite titles
                        cite_titles = []
                        for c in cites:
                            if isinstance(c, dict):
                                t = c.get("title", "").strip()
                            elif isinstance(c, str):
                                t = c.strip()
                            else:
                                continue
                            if t:
                                cite_titles.append(t)

                        if cite_titles:
                            qa_entries.append({
                                "question": q,
                                "cite_titles": cite_titles[:3],  # max 3 cites
                            })
                except Exception:
                    continue

    return qa_entries


def parse_cite_short(title):
    """Extract 'Điều X ...' from cite title for display."""
    m = re.search(r'(Điều\s+\d+[^\n,;]{0,60})', title)
    if m:
        return m.group(1).strip()
    return title[:80]


def main():
    print("Loading QA data...")
    qa_entries = load_all_qa()
    print(f"  {len(qa_entries)} usable QA entries")

    OUT_DIR.mkdir(exist_ok=True)

    print("Loading embedding model...")
    model = SentenceTransformer(
        "AITeamVN/Vietnamese_Embedding_v2",
        trust_remote_code=True,
    )
    model.max_seq_length = 512

    questions = [e["question"] for e in qa_entries]

    print(f"Embedding {len(questions)} questions...")
    batch_size = 256
    all_embeds = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeds.append(emb)
        if (i // batch_size) % 20 == 0:
            print(f"  {i}/{len(questions)}", flush=True)

    embeddings = np.vstack(all_embeds).astype("float32")
    print(f"  Embeddings shape: {embeddings.shape}")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  Index size: {index.ntotal}")

    faiss.write_index(index, str(OUT_DIR / "questions.faiss"))
    print(f"  Saved: {OUT_DIR / 'questions.faiss'}")

    # Save metadata (question + cite_titles)
    meta = []
    for e in qa_entries:
        meta.append({
            "question": e["question"],
            "cite_titles": e["cite_titles"],
            "cite_short": [parse_cite_short(t) for t in e["cite_titles"]],
        })
    with open(OUT_DIR / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"  Saved: {OUT_DIR / 'meta.pkl'}")

    # Quick sanity check
    test_q = "Điều kiện thành lập công ty TNHH một thành viên?"
    q_emb = model.encode([test_q], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, 3)
    print(f"\nSanity check: '{test_q}'")
    for rank, (score, idx) in enumerate(zip(D[0], I[0])):
        m = meta[idx]
        print(f"  [{rank+1}] score={score:.3f} Q: {m['question'][:70]}")
        print(f"       Cite: {', '.join(m['cite_short'][:2])}")

    print("\nDone.")


if __name__ == "__main__":
    main()
