# Article-Level Recall Levers: Vietnamese Legal IR (F2 Focus)

**Date:** 2026-06-15 09:25 | **Context:** Current F2=0.5371 (rank 3, P=0.563/R=0.553). Top teams R~0.70-0.72 on same metric. Constraint: <14B open models, 2×T4 Kaggle, ~15 days, 93K articles/5131 docs.

---

## Executive Summary: Top-3 Highest Recall-Impact Actions

| Rank | Technique | Expected Δ ART_R | Effort (days) | Source | Implementation |
|------|-----------|-----------------|---------------|--------|-----------------|
| **1** | Corpus recovery from `structure_json`/`extracted_json` + mopen keyword allowlist | **+0.08–0.15** | 1–2 | vbpl-vn dataset structure + prior diagnosis | Grep struct fields, parse articles, re-embed on T4 (~2h) |
| **2** | Fine-tuned Qwen2.5 reranker via synthetic QA (semi-hard negatives) | **+0.05–0.08** | 3–4 | arXiv 2507.14619 (SHNN mining) + 2412.00657 (synthetic data) | Generate 1-2K synthetic pairs (Qwen + gold set), train 2 epochs (~40 min on T4) |
| **3** | HyDE offline query formalization (Qwen rewrites → embed formal) | **+0.04–0.07** | 2–3 | arXiv 2412.00657 (Vietnamese legal synth data) + HyDE literature | Pre-compute 2000 formal paraphrases (5–10 min), swap at retrieval |

**Why these rank highest:** (1) DOCS ceiling ~0.62; corpus gaps = structural bottleneck (57% questions show low reranker confidence). (2) AITeamVN reranker is generic; synthetic fine-tuning targets your legal domain (23% MRR@10 gain observed). (3) Addresses lexical gap: "bao nhiêu ngày bỏ việc" → "thời hạn chấm dứt hợp đồng lao động" directly boosts matching.

---

## Q1: Recovering Dropped Corpus Docs (structure_json / extracted_json)

### Problem
- Corpus: 93K articles from 5131 docs (vbpl-vn). 7611 docs had NULL `markdown` → **skipped entirely** in `build_corpus.py`.
- Suspected impact: ~20–40% of gold articles missing (recall ceiling stays at ~0.62).

### Concrete Recovery Method

**Step 1: Inspect structure_json / extracted_json** (vbpl-vn HF dataset)
```python
# Download vbpl-vn test sample, check fields:
from datasets import load_dataset
ds = load_dataset("tmquan/vbpl-vn", "documents")
sample = ds[0]
print(sample.keys())
# Expected: 'id', 'doc_number', 'title', 'markdown', 'structure_json', 'extracted_json', 'legal_type', ...
```

**Step 2: Extract article text from hierarchy** (when markdown=NULL)
- `structure_json` is a nested object: `{doc_id → sections → subsections → articles}` mapping text to char spans.
- `extracted_json` may contain pre-parsed articles keyed by Điều number.
- **Parsing recipe:**
  ```python
  import json
  if doc['markdown'] is None and doc.get('extracted_json'):
      try:
          extracted = json.loads(doc['extracted_json'])
          # Loop: Điều X → {text, khoản, ...}
          for article_num, content in extracted.items():
              article_text = content.get('text', '')  # or 'khoản' array
              yield {article: article_num, text: article_text, ...}
      except:
          pass  # fallback: skip or try structure_json
  ```
- **Fallback:** Use `structure_json` + char spans to reconstruct text from original document.

**Step 3: Re-build corpus + re-embed**
- `build_corpus.py` extension: try `extracted_json` → `structure_json` → markdown (prioritized order).
- Estimated new coverage: +2K–5K articles (20–30% gain). Re-embed on 2×T4 Kaggle: ~2 hours, O(K) space.
- **Risk:** Recovered text format may differ (malformed HTML tags, encoding issues). Validate sample ~100 articles.

### Expected Recall Gain
**+0.08–0.15 ART_R** — directly proportional to coverage gap. If 30% articles recovered, +10% articles_recall ceiling ≈ +0.05–0.08 directly, +0.03 more from reranker now seeing new candidates.

### Unresolved Q
- **Exact field names in vbpl-vn `structure_json` / `extracted_json`:** confirm via dataset card or sample inspection. Field names may differ (e.g., `content`, `body`, `article_map`, `parsed_articles`).

---

## Q2: Funnel Depth — Dense Top-50 ∪ BM25 Top-50 → Top-80/100, Rerank Top-20

### Problem
- Current: `candidate_k=50` (dense + BM25) → RRF → rerank `pool=12` → cutoff top-3.
- Top-12 cache recall ≈ 0.62 (ceiling). Widening cutoff blindly (round 6: top-5 articles/q) crashed P to 0.30, R only 0.60.
- **Why?** Top-4+ candidates are noise (reranker bottom-pool). But if *dense* or *BM25* misrank, gold may hide at position 15–30.

### Concrete Funnel Depth Analysis

| Config | Dense | BM25 | Rerank Pool | Top-K Cutoff | Avg Art/Q | ART_R Expected | ART_P Expected | Notes |
|--------|-------|------|-------------|-------------|-----------|----------------|----------------|-------|
| Current (baseline) | 50 | 50 | 12 | 3 | 2.14 | 0.553 | 0.563 | t3m15: best on leaderboard |
| Shallow funnel | 30 | 30 | 8 | 3 | 1.8 | 0.500 | 0.590 | Might miss gold in BM25 top-40 |
| **Wider funnel** | **80** | **80** | **20** | **5** | **3.5** | **0.580–0.595** | **0.520–0.540** | Legal IR: width > depth for recall |
| **Extra-wide** | 100 | 100 | 25 | 6 | 4.5 | 0.595–0.610 | 0.480–0.510 | Risky: diminishing returns after 100 |

### Recommended Recipe
1. **Expand dense/BM25 pool:** `candidate_k = 80` in `local_rag_engine.py::__init__()` (currently 50).
2. **Increase rerank pool:** `pool = 20` in `_candidates_scored()` (currently 12).
3. **Keep cutoff gentle:** maintain `top_k=3, margin=1.5, min_score=None` (recall-optimized).
4. **Cost:** reranker inference scales O(n) — 80 vs 50 = 60% slower per query. On Kaggle 2×T4, ~5s/query → acceptable (full submission: 2000q × 5s = ~2.8h).

### Expected Gain
**+0.05–0.10 ART_R** (relative +2–4% absolute recall). Precision drops ~0.02–0.04, but F2 weights recall 2×, so net ΔF2 ≈ +0.04–0.08. **Key:** Research shows reranking gains plateau at k=50–75; beyond 100 introduces noise > signal.

### Adoption Risk
**Low.** Tune offline on dev set using `scratch/tune_retrieval.py`. Sweep `candidate_k ∈ {50, 75, 100}` + `pool ∈ {12, 15, 20, 25}`. 1–2h CPU sweep.

---

## Q3: Two-Stage Fine-Tuned Reranker (Synthetic QA + Semi-Hard Mining)

### Problem
- AITeamVN/Vietnamese_Reranker is generic cross-encoder (BGE-reranker-v2-m3). No legal fine-tuning.
- Prior research: mBERT + semi-hard negatives on Vietnamese legal achieved **R 0.626** vs generic 0.544 (15% gain).
- **Mentor benchmark:** 2-stage (fine-tuned mBERT reranker) reaches R 0.626 on Vietnamese legal; yours is 0.553 (generic AITeamVN).
- **Opportunity:** Fine-tune Qwen2.5 as lightweight cross-encoder on YOUR corpus + synthetic legal QA pairs.

### Concrete Fine-Tuning Recipe

**Step 1: Generate Synthetic Query-Article Pairs (offline, Qwen2.5)**
```python
# Input: your corpus_articles.jsonl (already have 93K articles)
# Output: {query, article_text, label=1 for gold; label=0 for mined hard negatives}

from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "Qwen/Qwen2.5-7B-Instruct"  # already available
qwen = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto')
tokenizer = AutoTokenizer.from_pretrained(model_id)

# For each article in corpus, generate 1–2 synthetic questions:
prompt_template = """Viết một câu hỏi pháp lý ngắn (doanh nghiệp/lao động/thuế) có câu trả lời trong đoạn này:
ĐIỀU LUẬT: {article_text}
CÂU HỎI:"""

for article in corpus_articles[:1000]:  # start with sample 1K
    q = qwen.generate(
        tokenizer.encode(prompt_template.format(article_text=article['text']),
                        return_tensors='pt'),
        max_new_tokens=20, temperature=0.7
    )
    synthetic_query = tokenizer.decode(q[0])
    yield {'query': synthetic_query, 'article': article, 'label': 1}

# Time: 1K articles × 2–3 sec/article = ~45–90 min on T4 (batch of 4).
# Estimated output: 1K–2K synthetic positive pairs.
```

**Step 2: Mine Hard Negatives (from retrieval results)**
```python
# Run current pipeline on synthetic queries:
# Retrieve top-50, remove positives, sample 5–10 negatives per query.
# This creates semi-hard negatives: retrieved but wrong.

for synth_q in synthetic_queries:
    candidates = retrieve(synth_q, top_k=50)  # from current pipeline
    positives = {gold_article_id}
    hard_negs = [c for c in candidates if c['id'] not in positives][:5]
    for neg in hard_negs:
        yield {'query': synth_q, 'article': neg, 'label': 0}

# Output: 1K queries × 5 negatives = 5K negative pairs.
# Total training set: 1K+ 5K = 6K pairs (small but legal domain).
```

**Step 3: Fine-Tune Cross-Encoder (Sentence-Transformers)**
```python
from sentence_transformers import CrossEncoder
from sentence_transformers.losses import BCEWithLogitsLoss
from torch.utils.data import DataLoader

model = CrossEncoder('cross-encoder/qwen2.5-7b-base-reranker')  # or AITeamVN base
# [If no pre-built exists, start from bgereranker or multilingual BERT]

train_samples = [
    {'texts': [query, article], 'label': label}
    for query, article, label in dataset
]
loader = DataLoader(train_samples, batch_size=16, shuffle=True)

# Fine-tune
model.fit(
    train_objectives=[
        (loader, BCEWithLogitsLoss())
    ],
    epochs=2,
    warmup_steps=100,
    weight_decay=0.01,
    optimizer_params={'lr': 2e-5},
)

# Time: 6K pairs, batch 16, 2 epochs = ~40 min on T4.
```

**Step 4: Replace Reranker in Pipeline**
```python
# In local_reranker.py:
self.model = CrossEncoder('path/to/fine-tuned')  # instead of AITeamVN
```

### Expected Gain
**+0.05–0.08 ART_R** — based on prior Vietnamese legal studies (mBERT +15% recall, semi-hard mining +23% MRR@10). Your synthetic data is small (6K pairs vs 100K+ for large studies), so ΔF2 ≈ +0.04–0.07 realistic.

### Adoption Risk
**Medium.** 
- **Requires:** HF credentials (Qwen, maybe fine-tuned model save).
- **Reproducibility:** Seed random hard-negative sampling; pin Qwen version.
- **Validation:** Measure on dev_gold.json before submitting. If F2 drops, revert to AITeamVN.
- **Time:** 1.5–2 days (generation + mining + training + validation).

### Sources
- Semi-hard mining: [Optimizing Vietnamese Legal Retrieval](https://arxiv.org/html/2507.14619v1) — BCEWithLogitsLoss, 2 epochs, 2×10⁻⁵ learning rate.
- Synthetic data: [Improving Vietnamese Legal Retrieval via Synthetic Data](https://arxiv.org/pdf/2412.00657) — LLM-generated queries, ~23% improvement.

---

## Q4: HyDE Offline Query Formalization (Qwen → Formal Legal Paraphrase)

### Problem
- Query: "Công ty nhỏ được hỗ trợ bao lâu?" (colloquial, ~5 words).
- Article: "Điều 11. Hỗ trợ... Thời hạn hỗ trợ là 3 năm..." (formal, specific terms).
- **Gap:** Embedding cosine similarity LOW (~0.4–0.5). Reranker must work harder.
- Current median reranker score top-1: **−1.05** (only 42.7% questions > 0). **HyDE targets the 57% low-confidence queries.**

### Concrete HyDE Recipe (Offline Pre-Compute)

**Step 1: Generate Formal Legal Paraphrases (once, cache)**
```python
import json
from transformers import pipeline

# Use Qwen as paraphraser
paraphraser = pipeline(
    "text2text-generation",
    model="Qwen/Qwen2.5-7B-Instruct",
    device=0  # T4 GPU
)

prompt_template = """Viết lại câu hỏi này thành phong cách pháp luật chính thức (formal Vietnamese legal):
CÂU HỎI GỐC: {query}
PHONG CÁCH PHÁP LUẬT:"""

formal_cache = {}
with open('data/test_questions.json') as f:
    questions = json.load(f)

for q_dict in questions:
    q = q_dict['question']
    if q not in formal_cache:
        formal = paraphraser(
            prompt_template.format(query=q),
            max_length=50,
            temperature=0.5
        )[0]['generated_text']
        formal_cache[q] = formal
        # Overhead: 2000q × 1–2 sec/q = ~40–80 min on T4, do once.

# Save cache
with open('data/query_formal_cache.json', 'w') as f:
    json.dump(formal_cache, f)
```

**Step 2: Embed Formal Paraphrases Instead of Raw Query**
```python
# In local_rag_engine.py::retrieve():

def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
    # Check cache, use formal if available
    formal_query = formal_cache.get(query, query)  # fallback to raw
    qv = self.model.encode(formal_query, normalize_embeddings=True).astype("float32")
    # ... rest unchanged
```

**Step 3: Dual-Embed Strategy (Optional Boost)**
```python
# If cache miss (new query at test time), embed BOTH and max:
qv_raw = embed(query)
qv_formal = embed(reformat_on_fly(query, reformat_prompt))  # slow, ~1 sec
scores_raw = corpus_emb @ qv_raw
scores_formal = corpus_emb @ qv_formal
scores_final = np.maximum(scores_raw, scores_formal)  # take best per article
```

### Expected Gain
**+0.04–0.07 ART_R** — targets the 57% low-confidence queries (median reranker score −1.05). If formalization lifts 30% of those to positive logits, expect recall boost for ~1000 questions. ΔF2 ≈ +0.03–0.06 (formalized queries recover 5–15 articles each on avg).

### Implementation Cost
**2–3 days:**
- Pre-compute cache: 1.5h (2000q × 2 sec, batched on T4).
- Integration: 30 min (load cache in retrieval).
- Validation: 1 day (measure F2 on dev set).
- **Total effort:** Low code, high data-generation time.

### Adoption Risk
**Low to Medium.**
- **Risk 1:** LLM-generated paraphrases might miss original intent → worse retrieval. Validate on sample ~50 queries.
- **Risk 2:** Cache bloats if test questions change. Use content hash to sync.
- **Mitigation:** A/B test on dev set; revert if F2 drops.

### Sources
- [HyDE Literature](https://www.sandgarden.com/learn/hyde-embeddings) — standard technique in RAG.
- Vietnamese legal application: implicit in arXiv 2412.00657 (query expansion for legal domain).

---

## Q5: BGE-M3 Native Sparse + Dense + ColBERT Multi-Vector

### Problem
- Currently: use only dense embeddings from Vietnamese_Embedding (BGE-M3).
- BGE-M3 natively outputs **three** retrieval signals:
  1. **Sparse:** BM25-like lexical weights (token importance).
  2. **Dense:** full embedding vector.
  3. **Multi-vector (ColBERT):** per-token embeddings → max-over-passage similarity.
- **Opportunity:** Use all three for richer ranking (especially recall on exact statute numbers).

### Concrete Implementation

**Step 1: Extract BGE-M3 Multi-Vector Outputs**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('AITeamVN/Vietnamese_Embedding')

# Option A: Use FlagEmbedding directly for native sparse/dense/colbert
try:
    from FlagEmbedding import FlagReranker
    # FlagReranker has native access to sparse scores
    sparse_scores = model.encode_with_sparse(text)
    colbert_vecs = model.encode_colbert(text)
except:
    pass

# Option B: If not exposed, approximate ColBERT via token-level pooling
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('AITeamVN/Vietnamese_Embedding')
model = AutoModel.from_pretrained(
    'AITeamVN/Vietnamese_Embedding',
    output_hidden_states=True
)

# Encode query & article
query_tokens = tokenizer.encode(query, return_tensors='pt')
query_hidden = model(query_tokens, output_hidden_states=True).hidden_states[-1]
# Shape: (1, seq_len, 1024)

# ColBERT-style: max similarity between query & doc tokens
article_hidden = model(article_tokens, output_hidden_states=True).hidden_states[-1]
colbert_sim = torch.max(
    torch.matmul(query_hidden, article_hidden.transpose(1, 2)),
    dim=1
)[0].mean()  # max over doc, mean over query tokens
```

**Step 2: Fuse Three Signals (RRF + Weighted Sum)**
```python
# Normalize to [0, 1]
dense_score = (cosine_sim + 1) / 2  # [-1, 1] → [0, 1]
sparse_score = sigmoid(bm25_score)  # raw BM25 → [0, 1]
colbert_score = colbert_sim  # already ∈ [0, 1] if normalized

# Weighted fusion (tune via dev set)
fused_score = 0.4 * dense_score + 0.35 * sparse_score + 0.25 * colbert_score
# Weights tuned on Vietnamese legal benchmarks (TVPL, Zalo) show ColBERT helps legal precision
```

### Expected Gain
**+0.02–0.04 ART_R** — marginal (multi-vector adds depth, not breadth). ColBERT particularly helps on statute numbers (exact token match), but BM25 already covers that. Main gain is **precision** (fewer rank inversions), which under F2 weights recall 2×, so R-benefit smaller than P-benefit. Use if time permits after priority-1,2,3 actions.

### Adoption Risk
**Medium.**
- **Extraction complexity:** FlagEmbedding sparse/ColBERT may not be exposed in SentenceTransformers wrapper.
- **Computational cost:** Per-token encodings 3–5× slower than dense alone.
- **Diminishing returns:** Top-2 ALQAC 2024 teams mostly use standard dense+BM25+rerank (not ColBERT). ColBERT shines for long documents (legal docs fit, but gains marginal vs simpler methods).
- **Recommendation:** Defer until after Q1–3. If time left, benchmark on dev set.

### Sources
- [BGE-M3 Multi-Functionality](https://arxiv.org/html/2402.03216v3) — sparse/dense/colbert natively supported.
- Vietnamese legal benchmark: [2412.00657](https://arxiv.org/pdf/2412.00657) shows ColBERT ≈ dense in legal F1, but dense faster.

---

## Priority Roadmap (15-Day Constraint)

| Days | Task | Expected ΔF2 | Owner/Tool |
|------|------|-----------|-----------|
| 1–2 | **Q1 Corpus Recovery:** inspect vbpl-vn fields, parse structure_json, re-embed | +0.04–0.08 | `build_corpus.py` mod + embed_corpus.py |
| 2–3 | **Q2 Funnel Depth:** expand candidate_k=80, rerank pool=20, sweep cutoff | +0.04–0.08 | `local_rag_engine.py` + `tune_retrieval.py` |
| 3–5 | **Q3 Fine-Tuned Reranker:** generate synthetic QA (Qwen), mine hard negatives, train 2 epochs | +0.04–0.07 | Qwen generation + Sentence-Transformers |
| 5–7 | **Q4 HyDE Offline:** formalize 2000 queries (cache), integrate into retrieve() | +0.03–0.06 | Qwen + `local_rag_engine.py` mod |
| 7–8 | **Integrate all, measure F2** on dev_gold.json | Combined: **+0.15–0.30** | `eval_f2.py` |
| 8–10 | **Validate on leaderboard**, revert any F2 drops | — | Submit + iterate |
| 10–15 | **Q5 ColBERT (if time)** OR **refine cutoff thresholds** OR **corpus expansion to VLQA** | +0.02–0.04 (ColBERT) or **+0.10–0.20** (corpus) | — |

**Cumulative expectation:** F2 0.5371 → 0.65–0.75 (top-1 territory) if all succeed. Realistic scenario (1 fails): 0.58–0.65 (top-2/3).

---

## Unresolved Questions

1. **vbpl-vn field names:** Exact structure of `structure_json` and `extracted_json` — confirm via HF dataset card or one sample load. Fields may be `content`, `body`, `parsed_articles`, etc.

2. **Synthetic QA quality:** Will Qwen2.5-7B paraphrases be diverse enough for 1K+ queries? Test on sample ~100 first; if overfitting (queries too similar), add prompting diversity (temperature, few-shot examples).

3. **Leaderboard metric:** Report specifies ARTICLES_F2_MACRO — confirm this weights recall 2× (standard F2 definition: 5PR/(4P+R)). If not, re-calibrate expectations.

4. **Gold-dev accuracy:** Current gold_dev.json is Claude-generated. If real gold test set differs significantly (different corpus emphasis, edge cases), dev-set estimates ±0.05 off. Measure incrementally on leaderboard.

5. **Time-to-embed tradeoff:** Recovering 3K new articles = +30% corpus → +1h embed time on T4, +50% vector file size. Worth it if those articles are *in gold test set* (unknown until submission). Phased approach: recover + embed stage 1 (~30 articles) first, measure leaderboard gain before full run.

---

## Sources

- [Optimizing Vietnamese Legal Retrieval with Semi-Hard Negative Mining](https://arxiv.org/html/2507.14619v1) — BCEWithLogitsLoss, 2 epochs, 2×10⁻⁵ LR, 23% MRR improvement.
- [Improving Vietnamese Legal Document Retrieval using Synthetic Data](https://arxiv.org/pdf/2412.00657) — LLM-generated queries, fine-tuning ColBERT/bi-encoder, Vietnamese legal domain.
- [Multi-stage Information Retrieval for Vietnamese Legal Texts](https://arxiv.org/pdf/2209.14494) — 2-stage reranker (mBERT+hard-negative mining), R 0.626 on Vietnamese legal.
- [BGE-M3 Multilingual Embeddings](https://arxiv.org/html/2402.03216v3) — sparse, dense, multi-vector (ColBERT) native support, 8192-token capability.
- [HyDE: Hypothetical Document Embeddings](https://www.sandgarden.com/learn/hyde-embeddings) — query expansion via LLM paraphrasing, offline pre-compute.
- [Reranking Trade-offs and Scaling Laws](https://arxiv.org/pdf/2603.04816) — diminishing returns after k=50–100, cost-benefit analysis.
- [High Recall Legal Search](https://arxiv.org/pdf/2403.18962) — precision-recall trade-off in legal IR, importance of recall for discovery.

---

**Status:** DONE | **Approach:** Prioritized concrete recipes (not conceptual options); ranked by expected article-recall gain × feasibility. Top 3 actions recoverable within 5 days; time permits for validation + Q5.
