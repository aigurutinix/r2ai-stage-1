#!/bin/bash
# Fine-tune AITeamVN/Vietnamese_Reranker on legal hard negatives
# (data_final_hard_negatives.jsonl) -> reranker_finetuned_v2/checkpoint-8000.
set -e
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
# WANDB logging is optional; disable if no key:
# export WANDB_MODE=disabled

echo "=== Fine-tune Vietnamese_Reranker v2 (hard negatives) ==="
echo "$(date): Starting..."

PYTHONUNBUFFERED=1 python3 train_reranker_v2.py

echo "$(date): Done!  (best checkpoint: reranker_finetuned_v2/checkpoint-8000)"
