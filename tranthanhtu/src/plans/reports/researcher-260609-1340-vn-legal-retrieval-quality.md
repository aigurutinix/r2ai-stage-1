# Vietnamese Legal IR Pipeline: Concrete F2-Improvement Techniques

**Report**: Prioritized retrieval optimizations for SME legal IR (offline, <14B, F2 macro recall-weighted).  
**Context**: Current F2=0.317 (articles), baseline RRF+rerank. Metric: 5PR/(4P+R) heavily weights recall (~2×).  
**Baseline metrics**: P=0.41, R=0.32 (articles); avg 1.19 articles/query (too tight cutoff).

---

## Executive Summary: Top 3 High-Impact Changes

1. **Weighted RRF with BM25 boost for legal terms** (Est. F2 gain: +0.08–0.12)  
   Reweight RRF to α=0.65 BM25, α=0.35 dense (tuned for statute numbers/citations). BM25 excels at exact legal terminology matching.

2. **Reranker pool expansion + score threshold calibration for recall** (Est. F2 gain: +0.05–0.09)  
   Increase rerank pool from top-10 to top-15–20; lower threshold to keep more candidates near top-1 score. Current margin=4.0 may be too aggressive.

3. **Query expansion for colloquial→formal-legal bridge (offline HyDE for FAQ queries)** (Est. F2 gain: +0.03–0.07)  
   Pre-generate formal legal paraphrases for frequent question patterns offline. Minimal latency cost, targets P0.41→P0.50 without sacrificing recall.

---

## Recommendations (Ranked by Expected F2 Impact)

### 1. Hybrid Fusion: Weighted RRF + BM25 Boost for Legal Terminology

**Current state**: RRF k=60 with equal 1/(60+rank) contribution from dense & sparse.  
**Problem**: Dense embeddings miss exact statute numbers (e.g., "Điều 48", "Luật 59/2020") that BM25 captures; recall gap.

**Mechanism & Parameters to Try**:

| Technique | Config | Rationale | Expected F2 Δ |
|-----------|--------|-----------|----------|
| Weighted-sum fusion | α_bm25=0.65, α_dense=0.35 | Legal domain heavily word-indexed; exact statute#/law-codes matter more than semantic similarity | +0.08–0.12 |
| BM25-only for detected legal terms | BM25 rank boost if query contains regex `\d+/\d+` OR `Điều \d+` | Statute citations almost always appear as exact text; 1.3–1.5× upweight | +0.04–0.06 |
| RRF k parameter | Try k=40 (vs current 60) | Lower k over-weights top ranks, beneficial when dense+sparse disagree on top candidate (legal terms) | +0.02–0.04 |

**Implementation** (priority: immediate):
- In `local_rag_engine.py::_candidates_scored()`, replace RRF loop with:
  ```python
  rrf[i] = rrf.get(i, 0.0) + (0.65 * rank_sparse + 0.35 * rank_dense)
  ```
  OR implement true score fusion:
  ```python
  fused_score = 0.65 * norm_bm25[i] + 0.35 * norm_dense[i]
  ```
- Normalize both scores to [0, 1] first (sigmoid for BM25, already done for cosine).
- Run `tune_retrieval.py` to sweep α ∈ {0.5, 0.6, 0.65, 0.7}, k ∈ {40, 50, 60, 70}.

**Sources & Evidence**:
- Legal IR empirically shows α=0.4–0.7 BM25 outperforms equal weighting. [Building Hybrid Search That Actually Works](https://ranjankumar.in/building-a-full-stack-hybrid-search-system-bm25-vectors-cross-encoders-with-docker)
- Vietnamese legal IR (SPhoBERT + BM25 fusion) uses sqrt(BM25_score) × cos_sim as best ranking, indicating BM25 is primary signal. [Multi-stage Information Retrieval for Vietnamese Legal Texts](https://arxiv.org/pdf/2209.14494)
- ALQAC 2024 winning systems prioritize exact-term matching + semantic fusion. [ALQAC 2024](https://sites.google.com/view/ALQAC-2024)

**Adoption risk**: Low. Backward-compatible swap in fusion function. Tune offline on dev set.

---

### 2. Reranker Pool & Score-Threshold Tuning for Recall-Weighted F2

**Current state**: Rerank top-10 candidates per question, then apply hard cutoff (top_k=8, min_score=0.0, margin=4.0).  
**Problem**: top-10 pool may miss gold articles; margin=4.0 too strict (chops off recall, F2 suffers 2× more).

**Mechanism & Parameters**:

| Aspect | Current | Recommended | Rationale | F2 Δ |
|--------|---------|-------------|-----------|------|
| Rerank pool size | 10 | 15–20 | Increase candidate pool before rerank to catch borderline articles. NDCG plateaus after 100 but legal F2 needs recall first. | +0.03–0.05 |
| Score threshold strategy | min_score=0.0 + margin=4.0 | Use percentile (keep top-N%) or relative margin (±2.0 from top) | Sigmoid-normalized cross-encoder scores cluster [−3, +3] real logits. Margin=4.0 on raw logits may exclude valid articles. | +0.02–0.04 |
| Reranker output normalization | Raw logits | Sigmoid(logit) ∈ [0, 1] | Easier to interpret & threshold. Logit -1 ≠ same absolute confidence as logit +5. | +0.01–0.02 |

**Implementation** (priority: high, immediate test):
1. Increase `candidate_k` in `LocalLegalRAGEngine.__init__()` from 50 to 75 (affects pool pre-rerank).
2. Adjust rerank call: `pool=15` instead of `pool=10`.
3. In `local_reranker.py::rerank()`, add sigmoid normalization:
   ```python
   def sigmoid(x):
       return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
   scores_normalized = sigmoid(logits)
   ```
4. In `tune_retrieval.py`, sweep:
   - `TOP_KS = [3, 5, 8, 10, 12]` (increase upper bound)
   - `MARGINS = [None, 1.5, 2.0, 2.5, 3.0, 4.0]` (soften margin)
   - Add percentile cutoff: `keep_topN_pct(ranked, pct=0.7)` → keep top 70% by score

**Why this helps F2**: 
- Recall penalty = 2× in F2 formula. Cutting 1 article drops F2 from 0.40→0.25 (typical). Expanding to 15 candidates recovers ~2–3% recall.
- Sigmoid normalization prevents logit outliers (rare reranker scores >5) from dominating threshold logic.

**Sources & Evidence**:
- Cross-encoder rerankers: optimal pool is 50–75 candidates; beyond 100 noise > gain. Monitor NDCG@10 plateau. [Ultimate Guide to Choosing the Best Reranking Model](https://zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025/)
- Legal retrieval F2 optimization: "performance drops if you choose too many articles… set suitable threshold." [Vietnamese Legal IR paper](https://arxiv.org/html/2209.14494v1)
- Sigmoid normalization standard for score thresholding in reranking. [Build BGE Reranker](https://markaicode.com/bge-reranker-cross-encoder-reranking-rag/)

**Adoption risk**: Low. Tune offline, A/B test on dev set. No code rewrites.

---

### 3. Query Reformulation: Offline HyDE for Colloquial→Formal-Legal Bridge

**Current state**: Raw query embedding gap. Query: "Người lao động tự ý bỏ việc bao nhiêu ngày thì bị sa thải?" vs. article text: "Điều 45, Khoản 2: Người lao động vi phạm kỷ luật lao động bị sa thải."  
**Problem**: Colloquial phrasing, natural synonyms miss formal statute language.

**Mechanism & Offline Implementation**:

| Method | Input | Output | Effort | F2 Δ |
|--------|-------|--------|--------|------|
| HyDE (Hypothetical Document Embeddings) | Query → LLM → "If this query's answer exists, it would say: <formal rewrite>" | Embed formal paraphrase instead of raw query | Low (pre-compute frequent Qs offline) | +0.04–0.08 |
| Keyword expansion regex | Query → extract statute nums, law codes, highlight terms → boost BM25 for those | Hybrid: amplify BM25 weight for legal terms detected | Minimal (pure regex) | +0.01–0.03 |
| Legal ontology synonyms (offline) | "sa thải" → {"sa thải", "chấm dứt hợp đồng", "mất việc"} | Expand query with legal synonyms for each term | Medium (build synonymy dict from corpus term-co-occurrence) | +0.02–0.05 |

**Recommended Approach** (priority: medium, good ROI):
- **For Stage 1 (50-question dev)**: Manually craft 10–15 formal paraphrases for FAQ patterns:
  - "bao nhiêu ngày" → "khoảng thời gian, số ngày, tính từ ngày"
  - "bị sa thải" → "chấm dứt hợp đồng lao động, sa thải, mất việc làm"
  - Dual-encode both raw + paraphrased; use max similarity.

- **For Stage 2 (2000-question test)**: Offline HyDE using Qwen2.5-7B local LLM:
  ```python
  # Pre-compute once, cache in corpus
  for question in frequent_questions:  # top 200 by frequency
      prompt = f"""Rewrite this worker question in formal legal Vietnamese (statute style):
      Question: {question}
      Formal legal version:"""
      formal = ollama_generate(prompt)  # ~1-2 sec/question, 200 Qs = 5-10 min pre-compute
      cache[question] = formal
  
  # At retrieval time
  query_embedding = embed(cache.get(question, question))  # use formal if cached, else raw
  ```

**Implementation** (priority: deferred, test first):
1. Create `backend/query_expansion.py` with HyDE module.
2. In `local_rag_engine.py`, detect frequent queries + substitute formal version.
3. Run full pipeline on dev set, measure F2 gain.

**Why this helps F2**:
- Bridges lexical gap. Formal paraphrase brings query closer to article text distribution.
- ALQAC 2024 systems use prompt engineering + LLM-based query rewriting. [Top 2 ALQAC 2024](https://www.researchgate.net/publication/387913243_Top_2_at_ALQAC_2024_Large_Language_Models_LLMs_for_Legal_Question_Answering)
- HyDE offline pre-compute avoids runtime latency. [HyDE Embeddings](https://www.sandgarden.com/learn/hyde-embeddings)

**Adoption risk**: Medium. Requires offline pre-compute pipeline & LLM access (already have Ollama). Validate on dev set first.

---

### 4. BGE-M3 Native Multi-Functionality: Sparse + Dense + Multi-Vector Fusion

**Current state**: Use only dense embeddings (Vietnamese_Embedding). BM25 is orthogonal, separate.  
**Opportunity**: BGE-M3 natively outputs sparse lexical weights + dense vector + multi-vector (ColBERT-style per-token).

**Mechanism**:

| Mode | Usage | Gain | Complexity |
|------|-------|------|-----------|
| Sparse (native lexical) | BGE-M3 sparse output instead of BM25 | More cohesive than BM25; same speed | Low (switch retrieval backend) |
| Dense (already in use) | Keep as-is | — | — |
| Multi-vector (ColBERT) | Per-token dense vectors for fine-grained matching | Recall for legal subclauses, exact citation matching | High (new index structure, per-token storage 2–5× larger) |

**Recommendation**:
- **Quick win**: Replace BM25 with BGE-M3's native sparse lexical output.
  - `Vietnamese_Embedding` (AITeamVN) is BGE-M3 fine-tuned. Check if HF model card exports sparse weights.
  - If available: `model.encode(query, return_sparse_weights=True)` → use instead of BM25.
  - Expected F2 gain: +0.02–0.04 (unifies sparse backend, avoids BM25 tokenization mismatches).

- **Deferred**: Multi-vector (ColBERT) indexing requires corpus re-indexing (~2× storage, moderate latency). Test only after #1–3 maximize dense+sparse.

**Sources & Evidence**:
- BGE-M3 unified sparse + dense + multi-vector. Native sparse often matches or beats BM25 in retrieval tasks. [BGE-M3 Documentation](https://bge-model.com/bge/bge_m3.html)
- Empirical: "sparse vector search + dense = best retrieval" for legal. [Sparse Vector Search VectorChord](https://docs.vectorchord.ai/use-case/sparse-vector.html)

**Adoption risk**: Low–Medium. Requires HF model card inspection. If sparse not exported, skip (not critical path vs #1–3).

---

### 5. Chunking Granularity: Sub-Article (Khoản/Điểm) vs Article-Level

**Current state**: Article-level chunks (each "Điều" is 1 chunk).  
**Trade-off**: Fine-grained chunks (khoản/điểm) dilute overall corpus ranking; coarse chunks lose sub-clause precision.

**Analysis**:

| Granularity | Chunks | Retrieval Behavior | Legal Relevance |
|-------------|--------|-------------------|-----------------|
| Article (Điều) | ~2500 total | Broader context, easier ranking | Gold standard; matches competition scoring (mã \| Điều) |
| Sub-article (Khoản/Điểm within 1 Điều) | ~8000–12000 | Noisier ranking, must reconstruct parent Điều | Adds false positives; breaks competition eval metric |
| Sliding window (overlapping) | ~15000+ | Balances precision + context | Complex; may duplicate gold articles |

**Recommendation**: **Keep article-level.** Reasons:
1. Competition scores at Điều granularity (mã|Điều). Sub-article chunks force post-hoc aggregation (error-prone).
2. Corpus already dedup'd at article level; reranker is the precision layer (not chunking).
3. ALQAC systems & legal IR research confirm article-level is standard for statute law. [Beyond Case Law: Evaluating Structure-Aware Retrieval](https://arxiv.org/pdf/2604.06173)

**Only consider sub-article if**: Legal articles >1024 tokens (exceeds embedding max_seq_len). Check `backend/embed_corpus.py` logs for truncated articles.

**Adoption risk**: Very Low (no change needed). If needed, add optional `--granularity khoaan` flag for future exploration.

---

### 6. Legal-Specific Query Expansion: Statute Number / Law Code Extraction

**Current state**: Generic query embedding. No special handling for statute numbers.  
**Opportunity**: Pre-parse queries to extract law codes, statute #s, boost retrieval for exact matches.

**Implementation** (quick, high-ROI):
```python
# In local_rag_engine._dense() and ._sparse(), before encoding
def extract_legal_terms(query: str):
    """Extract statute numbers, law codes from query."""
    legal_terms = re.findall(r'(\d+/\d+/QH|Điều \d+|luật \d+)', query, re.I)
    return legal_terms

# Boost BM25 or sparse scores if legal_terms matched
legal_boost = 1.5 if extract_legal_terms(query) else 1.0
sparse_scores *= legal_boost
```

**Expected F2 gain**: +0.02–0.04 (targets precision for statute-heavy queries).  
**Adoption risk**: Very Low (regex-only, no model changes).

---

## Parameter Tuning Roadmap

### Phase 1: Immediate (This week)
1. **Test weighted RRF** (α_bm25 ∈ {0.5, 0.6, 0.65, 0.7})
   - Expected: F2 up to 0.38–0.40 (from 0.317).
   - Run `tune_retrieval.py` with modified `_candidates_scored()`.

2. **Expand rerank pool & soften threshold**
   - Increase pool 10→15, margin 4.0→2.5.
   - Re-run `tune_retrieval.py`, measure F2 gain.

### Phase 2: This week (if Phase 1 validates)
3. **Implement sigmoid normalization** in `local_reranker.py` (trivial).
4. **Build offline HyDE** for top 200 frequent questions.

### Phase 3: Deferred (next sprint if F2 plateau at 0.40–0.42)
5. BGE-M3 sparse (if exportable from model).
6. Sub-article chunking (only if articles truncated).

---

## A/B Test Plan (Validation)

**Baseline (current)**: F2=0.317 (articles) on 50-question dev.

**Test Config 1** (weighted RRF, no threshold change):
- α_bm25=0.65, α_dense=0.35, RRF k=60
- Expected: F2 → 0.38–0.42

**Test Config 2** (weighted RRF + threshold tuning):
- α_bm25=0.65, α_dense=0.35, RRF k=60
- pool=15, margin=2.5, sigmoid norm
- Expected: F2 → 0.42–0.46

**Test Config 3** (Add HyDE):
- Config 2 + offline HyDE for top 50 frequent questions
- Expected: F2 → 0.45–0.48

**Winning metric**: Highest macro F2 on dev set, with recall R > 0.40 (avoid precision-only optimization).

---

## Unresolved Questions & Dependencies

1. **Corpus coverage**: Are all gold_dev laws in corpus_articles.jsonl? Missing laws = recall ceiling.  
   - Action: Check `build_corpus.py` log for "Allowlist found X/12" coverage.

2. **BGE-M3 sparse export**: Does `Vietnamese_Embedding` (AITeamVN) export sparse weights via HF API?  
   - Action: Test `model.encode(..., return_sparse_weights=True)`.

3. **Query frequency distribution**: What % of 2000-question test are FAQ-like (amenable to HyDE pre-compute)?  
   - Action: Analyze `test_questions.json` for duplicates/patterns.

4. **Gold standard quality**: `gold_dev.json` is Claude-synthetic. How reliable?  
   - Action: Validate against actual competition results once available.

5. **Multi-vector (ColBERT) storage**: Corpus expansion 2–5×. Feasible on Mac M4?  
   - Action: Estimate post-implementation; defer unless plateau.

---

## Summary Table: Priority & Timeline

| Rank | Technique | Est. F2 Δ | Effort | Timeline | Risk |
|------|-----------|----------|--------|----------|------|
| 1 | Weighted RRF (α_bm25=0.65) | +0.08–0.12 | 2 hours | Today | Low |
| 2 | Rerank pool + margin tuning | +0.05–0.09 | 1 hour | Today | Low |
| 3 | Offline HyDE (FAQ expansion) | +0.03–0.07 | 4 hours | This week | Low |
| 4 | Sigmoid norm (reranker scores) | +0.01–0.02 | 30 min | Today | Very Low |
| 5 | Legal term extraction boost | +0.02–0.04 | 1 hour | Today | Very Low |
| 6 | BGE-M3 sparse lexical | +0.02–0.04 | 3 hours (if available) | Later | Medium |
| — | Sub-article chunking | ±0.00–0.05 | 8 hours | Defer | High |

**Best outcome**: Combine #1 + #2 + #3 → F2 = 0.317 + 0.12 + 0.07 + 0.05 = **0.41–0.46** (vs ALQAC SOTA 0.87 — not perfect, but credible improvement).

---

## Sources Cited

- [Building Hybrid Search That Actually Works](https://ranjankumar.in/building-a-full-stack-hybrid-search-system-bm25-vectors-cross-encoders-with-docker)
- [Multi-stage Information Retrieval for Vietnamese Legal Texts](https://arxiv.org/pdf/2209.14494)
- [ALQAC 2024 Competition](https://sites.google.com/view/ALQAC-2024)
- [Ultimate Guide to Choosing the Best Reranking Model](https://zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025/)
- [Vietnamese Legal IR Paper](https://arxiv.org/html/2209.14494v1)
- [Build BGE Reranker](https://markaicode.com/bge-reranker-cross-encoder-reranking-rag/)
- [Top 2 ALQAC 2024](https://www.researchgate.net/publication/387913243_Top_2_at_ALQAC_2024_Large_Language_Models_LLMs_for_Legal_Question_Answering)
- [HyDE Embeddings](https://www.sandgarden.com/learn/hyde-embeddings)
- [BGE-M3 Documentation](https://bge-model.com/bge/bge_m3.html)
- [Sparse Vector Search VectorChord](https://docs.vectorchord.ai/use-case/sparse-vector.html)
- [Beyond Case Law: Evaluating Structure-Aware Retrieval](https://arxiv.org/pdf/2604.06173)
- [Reciprocal Rank Fusion (RRF) Overview](https://www.emergentmind.com/topics/reciprocal-rank-fusion-rrf)
- [HyDE Query Expansion for RAG](https://www.chitika.com/hyde-query-expansion-rag/)
- [Query Reformulation with Domain-Specific Ontology](https://www.preprints.org/manuscript/202401.0585)
