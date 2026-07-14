# Thư mục `models/`

Mô tả chi tiết toàn bộ model: xem [`../docs/model_description.md`](../docs/model_description.md).

## Model dùng trong pipeline (tất cả < 14B)

| Model | Nguồn / Vị trí | Params | Vai trò |
|-------|----------------|--------|---------|
| `AITeamVN/Vietnamese_Embedding_v2` | [HuggingFace](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2) | 568M | Dense embedding (dim 1024) |
| `AITeamVN/Vietnamese_Reranker` | [HuggingFace](https://huggingface.co/AITeamVN/Vietnamese_Reranker) | 568M | Reranker base (để fine-tune) |
| `reranker_finetuned_v2_ck8000/` | thư mục này | 568M | **Reranker chính (top-5 filter)** |
| `Qwen3-8B-AWQ` | [HuggingFace](https://huggingface.co/Qwen/Qwen3-8B-AWQ) | 8B | LLM decompose / HyDE / rerank / answer |
| `qwen3-1.7b-base/` | thư mục này | 1.7B | Base model cho LoRA classifier |
| `legal_classifier_lora_v2_ck23000/` | thư mục này | LoRA adapter | Classifier filter/add post-processing |

## Checkpoint reranker fine-tuned — `reranker_finetuned_v2_ck8000/`

Chứa các file inference của checkpoint (đã loại file train-only để giảm dung lượng):

- **Giữ (inference, ~2.3 GB)**: `model.safetensors` (2.27 GB), `config.json`,
  `sentencepiece.bpe.model`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`.
- **Bỏ (train-only)**: `optimizer.pt` (4.5 GB), `training_args.bin`, `scheduler.pt`,
  `rng_state.pth`, `trainer_state.json`.

Thông tin checkpoint:
- Base: `AITeamVN/Vietnamese_Reranker`. Kiến trúc XLM-RoBERTa-large (hidden 1024, 24 layers),
  Sigmoid activation.
- Checkpoint: step 8000 (~epoch 1.87). NDCG@10 ≈ 0.896 (base 0.8056).
- Train data: `data_final_hard_negatives.jsonl` (xem `../docs/model_description.md` mục 3).

Phục vụ inference:
```bash
vllm serve models/reranker_finetuned_v2_ck8000 --runner pooling --port 8012 --api-key token-abc123
```

> Không nộp `.env` (chứa `WANDB_API_KEY` — secret, chỉ dùng cho train logging).
