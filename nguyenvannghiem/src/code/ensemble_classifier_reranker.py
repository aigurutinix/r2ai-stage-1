#!/usr/bin/env python3
"""
Ensemble classifier P(Có) with reranker scores.
Sweep weights and thresholds to find optimal combination.

Usage:
    python3 ensemble_classifier_reranker.py
"""

import json
import pickle
import re
from pathlib import Path
from itertools import product

import numpy as np

BASE = Path("..") # adjust to your project root


def parse_art(art_str):
    parts = art_str.split("|")
    if len(parts) >= 3:
        m = re.search(r'\d+', parts[2])
        return (parts[0], m.group() if m else "")
    return None


def make_doc_name(doc_title, doc_description):
    if doc_description:
        return f"{doc_title} {doc_description}"
    return doc_title


def format_submission(qid, question, selected_cands):
    answer_parts = []
    relevant_docs = []
    relevant_articles = []
    seen_docs = set()
    seen_articles = set()
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
        "id": qid,
        "question": question,
        "answer": ". ".join(answer_parts) + "." if answer_parts else "",
        "relevant_docs": relevant_docs,
        "relevant_articles": relevant_articles,
    }


def load_ground_truth():
    """Load pseudo-GT from best LLM 8B thinking submission (F2=0.6056 on benchmark)."""
    gt_file = BASE / "submission_3_1_llm8b_rerankerv2ck8k_top5.json"
    sub = json.loads(gt_file.read_text())
    gt = {}
    for s in sub:
        qid = s["id"]
        arts = set()
        for a in s.get("relevant_articles", []):
            k = parse_art(a)
            if k:
                arts.add(k)
        if arts:
            gt[qid] = arts
    return gt


def compute_f2(submission, gt):
    precisions = []
    recalls = []
    for s in submission:
        qid = s["id"]
        if qid not in gt:
            continue
        gt_arts = gt[qid]
        pred_arts = set()
        for a in s["relevant_articles"]:
            k = parse_art(a)
            if k:
                pred_arts.add(k)
        if not pred_arts and not gt_arts:
            continue
        tp = len(pred_arts & gt_arts)
        p = tp / len(pred_arts) if pred_arts else 0
        r = tp / len(gt_arts) if gt_arts else 0
        precisions.append(p)
        recalls.append(r)

    avg_p = np.mean(precisions) if precisions else 0
    avg_r = np.mean(recalls) if recalls else 0
    f2 = (5 * avg_p * avg_r) / (4 * avg_p + avg_r) if (4 * avg_p + avg_r) > 0 else 0
    return f2, avg_p, avg_r


def main():
    questions = json.loads((BASE / "R2AIStage1DATA.json").read_text())
    with open(BASE / "rerank_intersection_scores.pkl", "rb") as f:
        pool = pickle.load(f)
    with open(BASE / "classifier_probs_top5.pkl", "rb") as f:
        cls_probs = pickle.load(f)

    gt = load_ground_truth()
    print(f"GT: {len(gt)} queries with ground truth")

    TOP_K = 5

    # Normalize reranker scores per query (min-max within each query's top-5)
    reranker_norm = {}
    for qi in range(len(questions)):
        cands = pool.get(qi, [])[:TOP_K]
        scores = [c.get("reranker_score", 0) for c in cands]
        smin, smax = min(scores) if scores else 0, max(scores) if scores else 1
        rng = smax - smin if smax > smin else 1
        for ci in range(len(cands)):
            reranker_norm[(qi, ci)] = (scores[ci] - smin) / rng

    # Show score distributions
    all_cls = [cls_probs.get((qi, ci), 0) for qi in range(len(questions)) for ci in range(len(pool.get(qi, [])[:TOP_K]))]
    all_rnk = [reranker_norm.get((qi, ci), 0) for qi in range(len(questions)) for ci in range(len(pool.get(qi, [])[:TOP_K]))]
    print(f"\nClassifier P(Có): mean={np.mean(all_cls):.3f}, std={np.std(all_cls):.3f}")
    print(f"Reranker (norm):  mean={np.mean(all_rnk):.3f}, std={np.std(all_rnk):.3f}")

    # Sweep: ensemble_score = alpha * P(Có) + (1-alpha) * reranker_norm
    # Then select top-k by ensemble_score, with threshold
    print(f"\n{'alpha':>6} {'method':>12} {'threshold':>10} {'avg':>5} {'empty':>6} {'F2':>7} {'P':>7} {'R':>7}")
    print("-" * 70)

    best_f2 = 0
    best_config = None

    # Method 1: weighted sum + threshold
    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        for threshold in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
            results = []
            for qi, q in enumerate(questions):
                cands = pool.get(qi, [])[:TOP_K]
                selected = []
                for ci, c in enumerate(cands):
                    p_cls = cls_probs.get((qi, ci), 0)
                    p_rnk = reranker_norm.get((qi, ci), 0)
                    score = alpha * p_cls + (1 - alpha) * p_rnk
                    if score >= threshold:
                        selected.append(c)
                results.append(format_submission(q["id"], q["question"], selected))

            f2, p, r = compute_f2(results, gt)
            avg = np.mean([len(s["relevant_articles"]) for s in results])
            empty = sum(1 for s in results if not s["relevant_articles"])

            if f2 > best_f2:
                best_f2 = f2
                best_config = ("weighted", alpha, threshold, avg, empty, f2, p, r)

            if f2 > 0.3:
                print(f"{alpha:>6.1f} {'weighted':>12} {threshold:>10.2f} {avg:>5.2f} {empty:>6d} {f2:>7.4f} {p:>7.4f} {r:>7.4f}")

    # Method 2: multiply P(Có) * reranker_score (raw, not normalized) + threshold
    for threshold in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
        results = []
        for qi, q in enumerate(questions):
            cands = pool.get(qi, [])[:TOP_K]
            selected = []
            for ci, c in enumerate(cands):
                p_cls = cls_probs.get((qi, ci), 0)
                p_rnk = c.get("reranker_score", 0)
                score = p_cls * p_rnk
                if score >= threshold:
                    selected.append(c)
            results.append(format_submission(q["id"], q["question"], selected))

        f2, p, r = compute_f2(results, gt)
        avg = np.mean([len(s["relevant_articles"]) for s in results])
        empty = sum(1 for s in results if not s["relevant_articles"])

        if f2 > best_f2:
            best_f2 = f2
            best_config = ("multiply", 0, threshold, avg, empty, f2, p, r)

        if f2 > 0.3:
            print(f"{'--':>6} {'multiply':>12} {threshold:>10.3f} {avg:>5.2f} {empty:>6d} {f2:>7.4f} {p:>7.4f} {r:>7.4f}")

    # Method 3: reranker top-k, filtered by classifier P(Có) >= threshold
    for top_n in [3, 4, 5]:
        for threshold in [0.05, 0.1, 0.15, 0.2, 0.3]:
            results = []
            for qi, q in enumerate(questions):
                cands = pool.get(qi, [])[:top_n]
                selected = []
                for ci, c in enumerate(cands):
                    p_cls = cls_probs.get((qi, ci), 0)
                    if p_cls >= threshold:
                        selected.append(c)
                results.append(format_submission(q["id"], q["question"], selected))

            f2, p, r = compute_f2(results, gt)
            avg = np.mean([len(s["relevant_articles"]) for s in results])
            empty = sum(1 for s in results if not s["relevant_articles"])

            if f2 > best_f2:
                best_f2 = f2
                best_config = (f"filter_top{top_n}", 0, threshold, avg, empty, f2, p, r)

            if f2 > 0.3:
                print(f"{'--':>6} {f'filter_top{top_n}':>12} {threshold:>10.2f} {avg:>5.2f} {empty:>6d} {f2:>7.4f} {p:>7.4f} {r:>7.4f}")

    # Baselines
    print(f"\n--- Baselines ---")
    for top_n in [3, 4, 5]:
        results = []
        for qi, q in enumerate(questions):
            cands = pool.get(qi, [])[:top_n]
            results.append(format_submission(q["id"], q["question"], cands))
        f2, p, r = compute_f2(results, gt)
        avg = np.mean([len(s["relevant_articles"]) for s in results])
        print(f"  Reranker top-{top_n}: avg={avg:.2f}, F2={f2:.4f}, P={p:.4f}, R={r:.4f}")

    print(f"\n=== Best config ===")
    if best_config:
        method, alpha, threshold, avg, empty, f2, p, r = best_config
        print(f"  {method} alpha={alpha} threshold={threshold} avg={avg:.2f} empty={empty} F2={f2:.4f} P={p:.4f} R={r:.4f}")

        # Save best submission
        results = []
        for qi, q in enumerate(questions):
            cands = pool.get(qi, [])[:TOP_K]
            selected = []
            for ci, c in enumerate(cands):
                p_cls = cls_probs.get((qi, ci), 0)
                p_rnk = reranker_norm.get((qi, ci), 0)
                if method == "weighted":
                    score = alpha * p_cls + (1 - alpha) * p_rnk
                    if score >= threshold:
                        selected.append(c)
                elif method == "multiply":
                    score = p_cls * c.get("reranker_score", 0)
                    if score >= threshold:
                        selected.append(c)
                else:
                    top_n = int(method.split("top")[1])
                    if ci < top_n and p_cls >= threshold:
                        selected.append(c)
            results.append(format_submission(q["id"], q["question"], selected))

        out = BASE / "submission_3_1_ensemble_best.json"
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
