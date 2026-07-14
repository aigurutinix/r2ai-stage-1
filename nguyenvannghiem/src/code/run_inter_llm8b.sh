#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Intersection → Reranker top-15 → LLM 8B rerank ==="
echo "$(date): Starting..."

PYTHONUNBUFFERED=1 python3 -c "
import json, re, pickle, time
import numpy as np
from pathlib import Path
from retrieval_llm_rerank import load_chunk_content_cache, llm_rerank_all, format_submission

BASE = Path('.')

# Load reranker scores (already sorted by reranker_score desc)
with open(BASE / 'rerank_intersection_scores.pkl', 'rb') as f:
    score_cache = pickle.load(f)

questions = json.loads((BASE / 'R2AIStage1DATA.json').read_text())

# Build candidates: reranker top-15 per query
candidates = []
for qi in range(len(questions)):
    scored = score_cache.get(qi, [])
    top15 = scored[:15]
    # Convert back to candidate format
    cands = []
    for s in top15:
        cands.append({
            'dense_score': s.get('dense_score', 0),
            'chunk_id': s.get('chunk_id', ''),
            'law_id': s.get('law_id', ''),
            'article_number': s.get('article_number', ''),
            'article_title': s.get('article_title', ''),
            'doc_title': s.get('doc_title', ''),
            'doc_description': s.get('doc_description', ''),
            'van_ban_id': s.get('van_ban_id', ''),
            'hieu_luc': s.get('hieu_luc', ''),
        })
    candidates.append(cands)

avg_c = np.mean([len(c) for c in candidates])
print(f'Reranker top-15 candidates: avg {avg_c:.1f}/query')

content_cache = load_chunk_content_cache()
queries = [q['question'] for q in questions]

print('LLM 8B reranking (8 workers, max_candidates=15)...')
rm_cache = BASE / 'llm_rerank_cache_inter_top15.pkl'
t0 = time.time()
llm_results = llm_rerank_all(queries, candidates, content_cache, n_workers=8, max_candidates=15, questions=questions, source='inter_top15')
elapsed = time.time() - t0
print(f'  Done in {elapsed:.1f}s')

with open(rm_cache, 'wb') as f:
    pickle.dump({'questions': questions, 'llm_results': llm_results}, f)

submission = [format_submission(q['id'], q['question'], r or []) for q, r in zip(questions, llm_results)]
out = BASE / 'submission_3_1_inter_rerank_llm8b.json'
out.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding='utf-8')

avg_art = np.mean([len(s['relevant_articles']) for s in submission])
print(f'Saved to {out}, avg articles: {avg_art:.1f}')
"

echo "$(date): DONE - submission_3_1_inter_rerank_llm8b.json"
