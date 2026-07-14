#!/usr/bin/env python3
"""
Rerank intersection pool (HyDE ∩ BM25S) using Vietnamese_Reranker via vLLM.
Save scores for reuse.

Usage:
    python3 rerank_intersection.py --port 8012
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


def get_text(c, content_cache, max_chars=6000):
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


def run(port=8012):
    questions = json.loads((BASE / "R2AIStage1DATA.json").read_text())

    with open(BASE / "hyde_candidates.pkl", "rb") as f:
        hyde_data = pickle.load(f)
    hyde_candidates = hyde_data["candidates"]

    with open(BASE / "submission_3_1_decompose_bm25_top150.json") as f:
        bm25_sub = json.load(f)

    def extract_pairs(arts):
        pairs = set()
        for art_key in arts:
            parts = art_key.split("|")
            if len(parts) >= 3:
                lid = parts[0]
                m = re.search(r'\d+', parts[2])
                ano = m.group() if m else ""
                pairs.add((lid, ano))
        return pairs

    print("Loading content cache...")
    content_cache = load_content_cache()
    print(f"  {len(content_cache)} chunks")

    # Build intersection candidates
    print("Building intersection candidates...")
    all_candidates = []
    for qi in range(len(questions)):
        bm25_set = extract_pairs(bm25_sub[qi]["relevant_articles"])
        inter = [h for h in hyde_candidates[qi]
                 if (h.get("law_id", ""), str(h.get("article_number", ""))) in bm25_set]
        all_candidates.append(inter)

    avg_c = np.mean([len(c) for c in all_candidates])
    print(f"  Intersection: avg {avg_c:.1f}/query")

    # Rerank
    print(f"Reranking via Vietnamese_Reranker (port {port})...")
    score_cache = {}
    t0 = time.time()
    done = 0

    def score_single(query_text, passage, port):
        """Score a single pair, retrying with shorter text on 400."""
        for max_c in [6000, 3000, 1500]:
            try:
                resp = requests.post(
                    f"http://localhost:{port}/v1/score",
                    headers={"Authorization": "Bearer token-abc123", "Content-Type": "application/json"},
                    json={"model": "reranker", "text_1": query_text, "text_2": [passage[:max_c]]},
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json()["data"][0]["score"]
            except Exception:
                if max_c == 1500:
                    return 0.0

    def score_query(qi):
        q = questions[qi]
        cands = all_candidates[qi]
        if not cands:
            return qi, []
        texts = [get_text(c, content_cache) for c in cands]
        # Try batch first
        try:
            resp = requests.post(
                f"http://localhost:{port}/v1/score",
                headers={"Authorization": "Bearer token-abc123", "Content-Type": "application/json"},
                json={"model": "reranker", "text_1": q["question"], "text_2": texts},
                timeout=120,
            )
            resp.raise_for_status()
            scores = [d["score"] for d in sorted(resp.json()["data"], key=lambda x: x["index"])]
        except Exception:
            # Fallback: score individually with truncation retry
            scores = [score_single(q["question"], t, port) for t in texts]

        scored = []
        for i, c in enumerate(cands):
            scored.append({
                "reranker_score": float(scores[i]),
                "dense_score": c.get("dense_score", 0),
                "chunk_id": c.get("chunk_id", ""),
                "law_id": c.get("law_id", ""),
                "article_number": str(c.get("article_number", "")),
                "article_title": c.get("article_title", ""),
                "doc_title": c.get("doc_title", ""),
                "doc_description": c.get("doc_description", ""),
                "van_ban_id": c.get("van_ban_id", ""),
                "hieu_luc": c.get("hieu_luc", ""),
            })
        scored.sort(key=lambda x: x["reranker_score"], reverse=True)
        return qi, scored

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(score_query, qi): qi for qi in range(len(questions))}
        for future in as_completed(futures):
            qi, scored = future.result()
            score_cache[qi] = scored
            done += 1
            if done % 100 == 0 or done == len(questions):
                print(f"  {done}/{len(questions)} ({time.time()-t0:.1f}s)", flush=True)

    # Save scores
    cache_file = BASE / "rerank_intersection_scores.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(score_cache, f)
    print(f"\nSaved scores to {cache_file}")

    # Analysis: rank of best submission articles
    best_sub_path = BASE / "submission_3_1_decompose_2to6_c80.json"
    if best_sub_path.exists():
        with open(best_sub_path) as f:
            best_sub = json.load(f)

        reranker_ranks = []
        dense_ranks = []
        found = 0
        notfound = 0
        for qi in range(len(best_sub)):
            if qi not in score_cache:
                continue
            selected = set()
            for art_key in best_sub[qi]["relevant_articles"]:
                parts = art_key.split("|")
                if len(parts) >= 3:
                    lid = parts[0]
                    m = re.search(r'\d+', parts[2])
                    ano = m.group() if m else ""
                    selected.add((lid, ano))

            # Reranker rank
            for rank, s in enumerate(score_cache[qi]):
                key = (s["law_id"], s["article_number"])
                if key in selected:
                    reranker_ranks.append(rank)
                    selected.discard(key)

            # Dense rank (sort by dense_score)
            dense_sorted = sorted(score_cache[qi], key=lambda x: x["dense_score"], reverse=True)
            selected2 = set()
            for art_key in best_sub[qi]["relevant_articles"]:
                parts = art_key.split("|")
                if len(parts) >= 3:
                    lid = parts[0]
                    m = re.search(r'\d+', parts[2])
                    ano = m.group() if m else ""
                    selected2.add((lid, ano))
            for rank, s in enumerate(dense_sorted):
                key = (s["law_id"], s["article_number"])
                if key in selected2:
                    dense_ranks.append(rank)
                    found += 1
                    selected2.discard(key)
            notfound += len(selected2)

        rr = np.array(reranker_ranks)
        dr = np.array(dense_ranks)
        print(f"\n=== GT rank in intersection pool (29 avg candidates) ===")
        print(f"Found: {found}, Not found: {notfound}")
        print(f"{'':>10} {'Reranker':>10} {'Dense':>10}")
        for cutoff in [3, 5, 8, 10, 15, 20]:
            r_pct = (rr < cutoff).sum() / len(rr) * 100
            d_pct = (dr < cutoff).sum() / len(dr) * 100
            print(f"  Top-{cutoff:<4} {r_pct:>9.1f}% {d_pct:>9.1f}%")
        print(f"  Mean   {rr.mean():>9.1f}  {dr.mean():>9.1f}")
        print(f"  Median {np.median(rr):>9.0f}  {np.median(dr):>9.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()
    run(args.port)
