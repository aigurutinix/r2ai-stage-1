#!/usr/bin/env python3
"""
Run LoRA classifier v2 via unsloth with batched inference.
Uses tokenizer padding for batch generation.

Usage:
    python3 run_classifier_unsloth.py --checkpoint legal_classifier_lora_v2/checkpoint-10000 --top-k 5 --batch-size 32
"""

import json
import pickle
import argparse
import re
import math
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

BASE = Path("..") # adjust to your project root
MAX_PASSAGE_CHARS = 4000

PROMPT_TEMPLATE = """Bạn là chuyên gia pháp luật Việt Nam. Hãy xác định điều luật sau có TRỰC TIẾP quy định về vấn đề trong câu hỏi không.

Câu hỏi: {question}

Điều luật:
{article}

Điều luật này có trực tiếp quy định về vấn đề trong câu hỏi không? Trả lời Có hoặc Không."""


def load_content_cache():
    CHUNKS_DIR = BASE / "vbpl_dataset" / "chunks"
    DOC_TYPES = ["hien_phap", "bo_luat", "luat", "phap_lenh", "nghi_dinh",
                 "nghi_quyet", "nghi_quyet_lien_tich", "thong_tu",
                 "thong_tu_lien_tich", "quyet_dinh"]
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
                doc_title = c.get("doc_title", "")
                art_no = c.get("article_number", "")
                art_title = c.get("article_title", "")
                header = f"{doc_title} - Điều {art_no}"
                if art_title:
                    header += f". {art_title}"
                if cid and content:
                    cc[cid] = f"{header}\n{content}"[:MAX_PASSAGE_CHARS]
    return cc


def get_passage(c, content_cache):
    cid = c.get("chunk_id", "")
    text = content_cache.get(cid, "")
    if not text and cid.startswith("qa_"):
        vbid = c.get("van_ban_id", "")
        ano = c.get("article_number", "")
        if vbid and ano:
            text = content_cache.get(f"{vbid}#dieu_{ano}", "")
    if not text:
        doc_title = c.get("doc_title", "")
        art_no = c.get("article_number", "")
        art_title = c.get("article_title", "")
        text = f"{doc_title} - Điều {art_no}"
        if art_title:
            text += f". {art_title}"
    return text[:MAX_PASSAGE_CHARS]


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


def parse_art(art_str):
    parts = art_str.split("|")
    if len(parts) >= 3:
        m = re.search(r'\d+', parts[2])
        return (parts[0], m.group() if m else "")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="legal_classifier_lora_v2/checkpoint-10000")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    questions = json.loads((BASE / "R2AIStage1DATA.json").read_text())
    with open(BASE / "rerank_intersection_scores.pkl", "rb") as f:
        pool = pickle.load(f)

    print("Loading content cache...")
    content_cache = load_content_cache()
    print(f"  {len(content_cache)} chunks")

    ck_path = args.checkpoint if args.checkpoint.startswith("/") else str(BASE / args.checkpoint)
    print(f"Loading model: {ck_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ck_path,
        max_seq_length=2048, load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    yes_id = tokenizer.encode(" Có", add_special_tokens=False)[-1]
    no_id = tokenizer.encode(" Không", add_special_tokens=False)[-1]
    print(f"  Yes={yes_id}, No={no_id}")

    # Build all prompts
    all_tasks = []
    for qi, q in enumerate(questions):
        cands = pool.get(qi, [])[:args.top_k]
        for ci, c in enumerate(cands):
            passage = get_passage(c, content_cache)
            prompt = PROMPT_TEMPLATE.format(question=q["question"], article=passage)
            all_tasks.append((qi, ci, prompt))

    print(f"Total pairs: {len(all_tasks)}")
    print(f"Batch size: {args.batch_size}")

    # Batched inference using forward logits (compare Có vs Không logits directly)
    all_probs = {}
    t0 = time.time()
    n_batches = math.ceil(len(all_tasks) / args.batch_size)

    for bi in tqdm(range(n_batches), desc="Batches"):
        batch = all_tasks[bi * args.batch_size : (bi + 1) * args.batch_size]
        prompts = [t[2] for t in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        for i, (qi, ci, _) in enumerate(batch):
            # Left padding: real tokens are at the end, last token is always at seq_len - 1
            last_pos = inputs["attention_mask"].shape[1] - 1
            logit_yes = outputs.logits[i, last_pos, yes_id].item()
            logit_no = outputs.logits[i, last_pos, no_id].item()
            max_l = max(logit_yes, logit_no)
            p_yes = math.exp(logit_yes - max_l) / (math.exp(logit_yes - max_l) + math.exp(logit_no - max_l))
            all_probs[(qi, ci)] = p_yes

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({len(all_tasks)/elapsed:.0f} pairs/sec)")

    # Save probs
    probs_path = BASE / f"classifier_v2_probs_top{args.top_k}.pkl"
    with open(probs_path, "wb") as f:
        pickle.dump(all_probs, f)
    print(f"Saved probs: {probs_path}")

    # Distribution
    all_p = list(all_probs.values())
    print(f"\nP(Có) distribution ({len(all_p)} pairs):")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        n = sum(1 for p in all_p if p >= t)
        print(f"  >= {t:.1f}: {n:>5d} ({n/len(all_p)*100:.1f}%)")



if __name__ == "__main__":
    main()
