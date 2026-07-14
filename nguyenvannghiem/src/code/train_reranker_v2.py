#!/usr/bin/env python3
"""
Fine-tune Vietnamese_Reranker on legal hard negatives.

Training data: data_final_hard_negatives.jsonl  (schema: {query, passage, label})
  - Built by mine_hard_negatives_dense.py: 10K data_final questions, dense top-20,
    positives = cited articles, hard negatives = ranks 5-12 (7 negatives / positive).
  - Grouped 1 positive + 7 hard negatives (GROUP_SIZE=8) for GroupShuffleSampler.

Eval: CrossEncoderRerankingEvaluator (NDCG@10 / MRR@10 / MAP) on a held-out
      500-query split taken from the same hard-negatives file.

Loss: BinaryCrossEntropyLoss (pos_weight balances the 1:7 ratio).
Output: reranker_finetuned_v2/checkpoint-8000 (sweet spot ~2 epochs, NDCG@10 ≈ 0.896).

Usage:
    python3 train_reranker_v2.py
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset
from torch.utils.data import Sampler
from transformers import EarlyStoppingCallback

from sentence_transformers import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

# ============================================================
# Config
# ============================================================
MODEL_NAME = "AITeamVN/Vietnamese_Reranker"
MAX_LENGTH = 2048
OUTPUT_DIR = "reranker_finetuned_v2"
RUN_NAME = "vietnamese-reranker-legal-v2"
NUM_EPOCHS = 2
GROUP_SIZE = 8  # 1 pos + 7 neg
BATCH_SIZE = GROUP_SIZE
LEARNING_RATE = 2e-5
EVAL_STEPS = 2000
MAX_PASSAGE_CHARS = 6000
EVAL_QUERIES = 500  # held-out queries for the reranking evaluator

BASE = Path("..") # adjust to your project root
HARD_NEG_FILE = BASE / "data_final_hard_negatives.jsonl"


class GroupShuffleSampler(Sampler):
    def __init__(self, dataset_size, group_size=8, seed=42):
        self.n_groups = dataset_size // group_size
        self.group_size = group_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        group_order = list(range(self.n_groups))
        rng.shuffle(group_order)
        for g in group_order:
            start = g * self.group_size
            for i in range(start, start + self.group_size):
                yield i

    def __len__(self):
        return self.n_groups * self.group_size

    def set_epoch(self, epoch):
        self.epoch = epoch


class GroupedCrossEncoderTrainer(CrossEncoderTrainer):
    def _get_train_sampler(self):
        return GroupShuffleSampler(
            len(self.train_dataset),
            group_size=GROUP_SIZE,
            seed=self.args.seed,
        )


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
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def load_hard_negatives():
    """Load data_final_hard_negatives.jsonl and group by query.

    The file is laid out in contiguous blocks per query: one label=1 row
    (the cited article) followed by its hard negatives (label=0). We group
    on the query string to reconstruct each (positive, [negatives]) sample.
    """
    by_query = defaultdict(lambda: {"pos": [], "neg": []})
    with open(HARD_NEG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = row["query"]
            passage = row["passage"][:MAX_PASSAGE_CHARS]
            if int(row["label"]) == 1:
                by_query[q]["pos"].append(passage)
            else:
                by_query[q]["neg"].append(passage)
    # keep only queries that have at least one positive and one negative
    samples = []
    for q, d in by_query.items():
        if d["pos"] and d["neg"]:
            samples.append({"query": q, "positive": d["pos"], "negative": d["neg"]})
    return samples


def build_train_rows(samples):
    """Each positive paired with 7 hard negatives → group of 8 rows."""
    train_rows = []
    stats = {"pos": 0, "neg": 0, "pad": 0}
    for s in samples:
        question = s["query"]
        negatives = s["negative"]
        for pos in s["positive"]:
            stats["pos"] += 1
            if len(negatives) >= 7:
                chosen = random.sample(negatives, 7)
                stats["neg"] += 7
            else:
                chosen = negatives[:]
                stats["neg"] += len(chosen)
                while len(chosen) < 7:
                    chosen.append(random.choice(negatives) if negatives else pos)
                    stats["pad"] += 1
                chosen = chosen[:7]
            train_rows.append({"sentence1": question, "sentence2": pos, "label": 1.0})
            for neg in chosen:
                train_rows.append({"sentence1": question, "sentence2": neg, "label": 0.0})
    n_groups = len(train_rows) // GROUP_SIZE
    train_rows = train_rows[:n_groups * GROUP_SIZE]
    logging.info(f"  Positives: {stats['pos']}, Negatives: {stats['neg']}, Padded: {stats['pad']}")
    logging.info(f"  Train: {len(train_rows)} rows ({n_groups} groups)")
    return Dataset.from_list(train_rows)


def main() -> None:
    setup_logging()
    random.seed(42)

    logging.info(f"Loading base model: {MODEL_NAME}")
    model = CrossEncoder(
        MODEL_NAME,
        num_labels=1,
        max_length=MAX_LENGTH,
        model_card_data=CrossEncoderModelCardData(
            language="vi",
            model_name="Vietnamese Reranker v2 fine-tuned on legal hard negatives",
        ),
    )

    logging.info(f"Loading hard negatives: {HARD_NEG_FILE}")
    samples = load_hard_negatives()
    logging.info(f"  {len(samples)} queries with pos+neg")

    # Hold out 500 queries for evaluation; train on the rest.
    random.shuffle(samples)
    n_eval = min(EVAL_QUERIES, len(samples) // 5)
    eval_samples = samples[:n_eval]
    train_samples = samples[n_eval:]
    logging.info(f"  Eval split: {len(eval_samples)} queries | Train split: {len(train_samples)} queries")

    train_dataset = build_train_rows(train_samples)

    n_pos = sum(1 for label in train_dataset["label"] if label > 0.5)
    n_neg = len(train_dataset) - n_pos
    pos_weight_value = n_neg / max(n_pos, 1)
    logging.info(f"  pos_weight: {pos_weight_value:.2f}")

    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(pos_weight_value))

    evaluator = CrossEncoderRerankingEvaluator(
        samples=eval_samples,
        at_k=10,
        name="legal-rerank",
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    )

    metric_key = "eval_legal-rerank_ndcg@10"

    args = CrossEncoderTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=3,
        logging_steps=10,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model=metric_key,
        greater_is_better=True,
        report_to="wandb",
        run_name=RUN_NAME,
        seed=42,
        dataloader_num_workers=4,
        dataloader_drop_last=True,
    )

    trainer = GroupedCrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # dummy, evaluator handles real eval
        loss=loss,
        evaluator=evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    logging.info("Starting training (baseline NDCG@10=0.8056)...")
    trainer.train()

    final_dir = f"{OUTPUT_DIR}/final"
    model.save_pretrained(final_dir)
    logging.info(f"Saved final model to {final_dir}")

    logging.info("Final evaluation:")
    final = evaluator(model)
    for k, v in sorted(final.items()):
        logging.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
