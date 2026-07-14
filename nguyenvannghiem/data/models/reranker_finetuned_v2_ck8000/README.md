---
language:
- vi
tags:
- sentence-transformers
- cross-encoder
- reranker
- legal
- r2ai
base_model: AITeamVN/Vietnamese_Reranker
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# Vietnamese Legal Reranker V2 (checkpoint 8000)

Cross-encoder reranker fine-tuned cho bài toán truy xuất điều luật Việt Nam, phục vụ cuộc thi **R2AI Stage 1 - Task 3.1** (Vietnamese Legal Article Retrieval).

## Base Model

[AITeamVN/Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker) — cross-encoder dựa trên XLM-RoBERTa, hỗ trợ tiếng Việt.

## Training

- **Data**: ~96K cặp (question, article) từ `data_final_hard_negatives.jsonl` — hard negatives được mine từ tập data_final (10K questions sampled by topic, dense FAISS top-20, ranks 5-12 làm negative). Gồm 12,057 positive và 84,399 negative (tỉ lệ 1:7)
- **Loss**: BinaryCrossEntropyLoss với pos_weight=7.0
- **Batch**: GroupShuffleSampler đảm bảo mỗi batch gồm 1 positive + 7 negatives cho cùng query
- **Max sequence length**: 2048 tokens
- **Checkpoint**: step 8000
- **Eval NDCG@10**: 0.9204

## Usage

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("NghiemAbe/r2ai-reranker-v2-ck8000")

pairs = [
    ["Thời hạn bảo hành công trình xây dựng là bao lâu?", "Điều 34. Bảo hành công trình xây dựng..."],
    ["Thời hạn bảo hành công trình xây dựng là bao lâu?", "Điều 50. Quản lý chất lượng vật liệu..."],
]
scores = model.predict(pairs)
```

## Context

Model này là một thành phần trong pipeline retrieval cho cuộc thi R2AI:
1. Query → Decompose → HyDE
2. Dense retrieval + BM25S → Intersection pool
3. **Reranker V2 (model này)** → Top-5 candidates
4. LLM 8B thinking → Final selection + Answer generation
