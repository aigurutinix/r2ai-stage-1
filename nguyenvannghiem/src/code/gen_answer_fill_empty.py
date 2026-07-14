#!/usr/bin/env python3
"""Fill empty answers in existing answer files."""

import json
import pickle
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from gen_answer import (
    BASE, load_content_cache, load_metadata_cache, get_article_info,
    gen_answer, FEWSHOT_MIN_SCORE, FEWSHOT_TOP_K,
)


def run_fill(port, model, answer_file, workers=4, reflection=False):
    questions = json.loads((BASE / "R2AIStage1DATA.json").read_text())
    submission = json.loads((BASE / answer_file).read_text())
    with open(BASE / "rerank_intersection_scores.pkl", "rb") as f:
        pool = pickle.load(f)
    fewshot_all = pickle.load(open(BASE / "fewshot_top5_reranked.pkl", "rb"))

    print("Loading caches...")
    content_cache = load_content_cache()
    meta_cache = load_metadata_cache()

    empty_ids = [i for i, s in enumerate(submission) if not s.get("answer", "")]
    print(f"Found {len(empty_ids)} empty answers to fill (reflection={reflection})")

    if not empty_ids:
        print("Nothing to fill!")
        return

    done = 0
    t0 = time.time()

    def process(qi):
        q = questions[qi]
        sub = submission[qi]
        cands = pool.get(qi, [])

        articles_info = []
        for art_str in sub.get("relevant_articles", []):
            info = get_article_info(art_str, cands, content_cache, meta_cache)
            if info:
                articles_info.append(info)

        if not articles_info:
            return qi, ""

        fewshot_examples = []
        for ex in fewshot_all[qi][:FEWSHOT_TOP_K * 2]:
            if ex["score_dense"] >= FEWSHOT_MIN_SCORE and (ex.get("answer") or ex.get("full_answer")):
                fewshot_examples.append(ex)
                if len(fewshot_examples) >= FEWSHOT_TOP_K:
                    break

        answer = gen_answer(q["question"], articles_info, fewshot_examples, port, model, reflection=reflection)
        return qi, answer

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, qi): qi for qi in empty_ids}
        for future in as_completed(futures):
            qi, answer = future.result()
            submission[qi]["answer"] = answer
            done += 1
            if done % 50 == 0 or done == len(empty_ids):
                elapsed = time.time() - t0
                print(f"  {done}/{len(empty_ids)} ({elapsed:.0f}s)", flush=True)

    filled = sum(1 for qi in empty_ids if submission[qi].get("answer", ""))
    print(f"Done in {time.time()-t0:.1f}s. Filled {filled}/{len(empty_ids)}")

    out = BASE / answer_file
    out.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--model", type=str, default="Qwen3-8B-AWQ")
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reflection", action="store_true")
    args = parser.parse_args()
    run_fill(args.port, args.model, args.file, args.workers, reflection=args.reflection)
