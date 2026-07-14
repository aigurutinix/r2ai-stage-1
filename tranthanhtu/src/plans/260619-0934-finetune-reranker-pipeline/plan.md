# Plan: Fine-tune reranker — phá trần 0.5766 → ~0.65+

**Ngày:** 2026-06-19 | **Metric:** ARTICLES_F2_MACRO | **Best hiện tại:** 0.5766 (off-the-shelf reranker, RERANK_MAX 512)

## Vì sao
Reranker đang OFF-THE-SHELF → trần ~0.55-0.62. Mọi đội VN-legal 0.65-0.70 đều fine-tune (arXiv 2412.00657: 64→79-84 MRR). DOCS_F2 > ARTICLES_F2 → reranker là mắt xích yếu. Fine-tune = đòn bẩy thật (rule cho phép).

## Pipeline (4 script, `scratch/finetune/`, chạy Kaggle 2×T4)

| # | script | đọc | ghi | ~thời gian |
|---|---|---|---|---|
| 1 | `gen_synthetic_pairs.py` | corpus + corpus_emb + Qwen2.5-7B | `synth_pairs.jsonl` | 2-4h |
| 2 | `mine_hard_negatives.py` | synth_pairs + corpus_emb + BGE-M3 | `train_reranker.jsonl` + `eval_pairs.jsonl` | ~0.5h |
| 3 | `train_reranker.py` | train_reranker.jsonl | `/kaggle/working/ft_reranker` | 1.5-3h |
| 4 | `eval_reranker.py` | eval_pairs + base + FT | báo cáo MRR/recall base vs FT | ~0.2h |

**Công thức (arXiv 2412.00657):** Qwen sinh câu hỏi aspect-guided từ mỗi điều → lọc bằng BGE-M3 top-40 self-retrieval → mine hard-neg ở rank [10,60) (tránh false-neg) → fine-tune cross-encoder. Passage text = `f"{title}\n{text}"` y hệt inference (không train/serve skew).

## Thứ tự chạy (Kaggle)

```bash
# Setup (1 cell): Internet ON, GPU T4×2; Add Input: corpus_articles.jsonl (93K) + corpus_emb.npy + stage1_questions.json
!git clone -q https://github.com/lehuubao2909/ai_legal_assistant.git || (cd ai_legal_assistant && git pull -q)
!pip install -q -U "FlagEmbedding>=1.3" deepspeed
%cd ai_legal_assistant

# 1. Synthetic (2-4h, Qwen) — Save Version sau bước này để giữ synth_pairs
!python scratch/finetune/gen_synthetic_pairs.py --max-articles 20000

# 2. Hard negatives (~0.5h, reuse corpus_emb.npy)
!python scratch/finetune/mine_hard_negatives.py

# 3. Train (1.5-3h, DDP 2×T4)
!torchrun --nproc_per_node 2 -m FlagEmbedding.finetune.reranker.encoder_only.base ... # train_reranker.py in lệnh đầy đủ; hoặc chạy wrapper:
!python scratch/finetune/train_reranker.py

# 4. GATE — eval offline TRƯỚC khi nộp
!python scratch/finetune/eval_reranker.py
```

## Hyperparams (T4-verified)
per_device_bs=2, train_group_size=8 (1 pos+7 neg), grad_accum=8, gradient_checkpointing ON, fp16, query_max_len=64 / passage_max_len=448 (=512 khớp RERANK_MAX), lr=2e-5 (nhẹ vì base đã adapt VN), 2 epoch, DDP torchrun 2 GPU. VRAM ~12-14GB/T4.

## Tích hợp (đổi 1 dòng)
Notebook retrieval cell 10: `RERANK_ID = "/kaggle/working/ft_reranker"` (hoặc Add-Input dataset). Giữ `RERANK_MAX=512`. Xóa `retrieved.json` cũ → chạy lại Phase A → sweep → nộp → so 0.5766.

## ⛔ GATE validate offline
**KHÔNG nộp leaderboard nếu `eval_reranker.py` chưa cho thấy FT > base rõ rệt** (MRR@10/recall@k). Tránh phí lượt nộp + bay mù.

## Rủi ro + xử lý
- **OOM train:** train_group_size 8→6, grad_accum→16; gradient_checkpointing PHẢI bật.
- **False-neg mining:** đã skip rank [0,10) + ±2 adjacency + same-(doc,article); nếu precision sập → thêm fuzzy-dedup clean_name+article.
- **Format:** OMIT pos_scores/neg_scores (KD off). pos/neg = đúng `title\ntext`.
- **Entrypoint drift:** pin `FlagEmbedding>=1.3`, assert import trước train; train script tự ghi ds_stage0.json nếu thiếu.
- **Synth quality:** log keep-rate của BGE-M3 top-40 filter; <40% → chỉnh prompt/cap text.

## Trạng thái script (workflow build + verify đối kháng 2026-06-19)
4 script đã author + review đối kháng (9 agents). Bug đã sửa: resume-dup (gen), eval_pairs interface (mine↔eval hội tụ schema `{query,pos,neg}`), DDP fallback (train). Syntax OK cả 4. **Chưa chạy thực tế Kaggle** → bug runtime (nếu có) sửa ở lần chạy đầu.

## Unresolved
- Quota Kaggle GPU đủ cho gen(2-4h)+train(1.5-3h) trong tuần?
- MAX_ARTICLES=20000 (mặc định) có đủ tín hiệu? Tăng nếu eval gain yếu.
- Cần eval gold R2AI thật (ngoài synthetic + gold_dev) để chốt — hiện chỉ proxy.
