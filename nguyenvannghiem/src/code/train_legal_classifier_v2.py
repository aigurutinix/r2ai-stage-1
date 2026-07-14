#!/usr/bin/env python3
"""
Train LoRA classifier v2: only reranker top-5 candidates as training data.
Labels from union 27B submission (higher recall).
Distribution matches inference exactly.

Usage:
    python3 train_legal_classifier_v2.py
"""

import json
import re
import pickle
import random
import logging
import os
from pathlib import Path

import torch
import numpy as np

random.seed(42)

BASE = Path("..") # adjust to your project root
MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
OUTPUT_DIR = "legal_classifier_lora_v2"
RUN_NAME = "legal-classifier-v2"
MAX_LENGTH = 2048
MAX_PASSAGE_CHARS = 4000
NUM_EPOCHS = 20
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 1
LEARNING_RATE = 2e-4
EVAL_STEPS = 1000
SAVE_STEPS = 1000
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0

LABEL_FILE = "data_final_hard_negatives.jsonl"

PROMPT_TEMPLATE = """Bạn là chuyên gia pháp luật Việt Nam. Hãy xác định điều luật sau có TRỰC TIẾP quy định về vấn đề trong câu hỏi không.

Câu hỏi: {question}

Điều luật:
{article}

Điều luật này có trực tiếp quy định về vấn đề trong câu hỏi không? Trả lời Có hoặc Không."""


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"logs/{RUN_NAME}.log"),
        ],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


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


def parse_art(art_str):
    parts = art_str.split("|")
    if len(parts) >= 3:
        m = re.search(r'\d+', parts[2])
        return (parts[0], m.group() if m else "")
    return None


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


def format_sample(row):
    prompt = PROMPT_TEMPLATE.format(
        question=row["question"],
        article=row["passage"],
    )
    answer = "Có" if row["label"] == 1 else "Không"
    return {"text": prompt + " " + answer}


def main():
    setup_logging()

    from unsloth import FastLanguageModel

    logging.info("Loading content cache...")
    content_cache = load_content_cache()
    logging.info(f"  {len(content_cache)} chunks")

    # Load data from hard negatives jsonl
    logging.info(f"Loading training data from {LABEL_FILE}...")
    all_rows = []
    with open(BASE / LABEL_FILE, "r") as f:
        for line in f:
            r = json.loads(line)
            all_rows.append({
                "question": r["query"],
                "passage": r["passage"][:MAX_PASSAGE_CHARS],
                "label": int(r["label"]),
            })

    n_pos = sum(1 for r in all_rows if r["label"] == 1)
    n_neg = len(all_rows) - n_pos
    logging.info(f"  Total: {len(all_rows)} ({n_pos} pos, {n_neg} neg, ratio 1:{n_neg/n_pos:.1f})")

    # Train on all, eval on 500 random samples
    random.shuffle(all_rows)
    train_rows = all_rows
    eval_rows = random.sample(all_rows, min(500, len(all_rows)))

    logging.info(f"  Train: {len(train_rows)}, Eval: {len(eval_rows)} samples")

    train_data = [format_sample(r) for r in train_rows]
    eval_data = [format_sample(r) for r in eval_rows]

    from datasets import Dataset
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    logging.info(f"Loading model with unsloth: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    from trl import SFTTrainer, SFTConfig

    yes_token_id = tokenizer.encode(" Có", add_special_tokens=False)[-1]
    no_token_id = tokenizer.encode(" Không", add_special_tokens=False)[-1]
    logging.info(f"  Yes token: {yes_token_id} ({tokenizer.decode([yes_token_id])})")
    logging.info(f"  No token: {no_token_id} ({tokenizer.decode([no_token_id])})")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = []
        targets = []
        for i in range(len(labels)):
            label_ids = labels[i]
            non_pad = np.where(label_ids != -100)[0]
            if len(non_pad) == 0:
                continue
            last_pos = non_pad[-1]
            if last_pos > 0:
                logit = logits[i][last_pos - 1]
                pred = 1 if logit[0] > logit[1] else 0
                true_token = label_ids[last_pos]
                target = 1 if true_token == yes_token_id else 0
                predictions.append(pred)
                targets.append(target)

        if not predictions:
            return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        acc = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average="binary", zero_division=0)
        prec = precision_score(targets, predictions, average="binary", zero_division=0)
        rec = recall_score(targets, predictions, average="binary", zero_division=0)
        return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=5,
            logging_steps=10,
            logging_first_step=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to="wandb",
            run_name=RUN_NAME,
            seed=42,
            max_seq_length=MAX_LENGTH,
            packing=False,
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=lambda logits, labels: logits[:, :, [yes_token_id, no_token_id]],
    )

    logging.info("Starting training...")
    trainer.train(resume_from_checkpoint=True)

    final_dir = f"{OUTPUT_DIR}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logging.info(f"Saved LoRA adapter to {final_dir}")


if __name__ == "__main__":
    main()
