#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Intersection pool → Vietnamese_Reranker (vLLM) → top-5 submit ==="
echo "$(date): Starting..."

PYTHONUNBUFFERED=1 python3 -c "
import json, re, pickle, time
import numpy as np
import requests
from pathlib import Path
from retrieval_llm_rerank import make_doc_name, load_chunk_content_cache

BASE = Path('.')

with open(BASE / 'hyde_candidates.pkl', 'rb') as f:
    hyde_data = pickle.load(f)
hyde_candidates = hyde_data['candidates']
questions = hyde_data['questions']

with open(BASE / 'submission_3_1_decompose_bm25_top150.json') as f:
    bm25_sub = json.load(f)

def extract_pairs(arts):
    pairs = set()
    for art_key in arts:
        parts = art_key.split('|')
        if len(parts) >= 3:
            lid = parts[0]
            m = re.search(r'\d+', parts[2])
            ano = m.group() if m else ''
            pairs.add((lid, ano))
    return pairs

# Load content
print('Loading content cache...')
content_cache = load_chunk_content_cache()

def get_text(c, max_chars=10000):
    cid = c.get('chunk_id','')
    content = content_cache.get(cid, '')
    if not content and cid.startswith('qa_'):
        vbid = c.get('van_ban_id','')
        ano = c.get('article_number','')
        if vbid and ano:
            content = content_cache.get(f'{vbid}#dieu_{ano}', '')
    doc_title = c.get('doc_title','')
    art_no = c.get('article_number','')
    art_title = c.get('article_title','')
    header = f'{doc_title} - Điều {art_no}'
    if art_title: header += f'. {art_title}'
    if content: return f'{header}\n{content[:max_chars]}'
    return header

# Build intersection candidates
print('Building intersection candidates...')
all_candidates = []
for qi in range(len(questions)):
    bm25_set = extract_pairs(bm25_sub[qi]['relevant_articles'])
    inter = [h for h in hyde_candidates[qi] if (h.get('law_id',''), str(h.get('article_number',''))) in bm25_set]
    all_candidates.append(inter)

avg_c = np.mean([len(c) for c in all_candidates])
print(f'Intersection: avg {avg_c:.1f}/query')

# Rerank via vLLM
print('Reranking via Vietnamese_Reranker (port 8012)...')
from concurrent.futures import ThreadPoolExecutor, as_completed

def score_query(qi):
    q = questions[qi]
    cands = all_candidates[qi]
    if not cands:
        return qi, []
    texts = [get_text(c) for c in cands]
    try:
        resp = requests.post(
            'http://localhost:8012/v1/score',
            headers={'Authorization': 'Bearer token-abc123', 'Content-Type': 'application/json'},
            json={'model': 'Reranker', 'text_1': q['question'], 'text_2': texts},
            timeout=120,
        )
        resp.raise_for_status()
        scores = [d['score'] for d in sorted(resp.json()['data'], key=lambda x: x['index'])]
    except:
        scores = [0.0] * len(cands)

    scored = list(zip(scores, cands))
    scored.sort(key=lambda x: x[0], reverse=True)
    return qi, scored

t0 = time.time()
results = [None] * len(questions)
done = 0

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(score_query, qi): qi for qi in range(len(questions))}
    for future in as_completed(futures):
        qi, scored = future.result()
        results[qi] = scored
        done += 1
        if done % 100 == 0 or done == len(questions):
            print(f'  {done}/{len(questions)} ({time.time()-t0:.1f}s)', flush=True)

print(f'Reranked in {time.time()-t0:.1f}s')

# Build submissions: top-3, top-5, top-8
for top_k in [3, 5, 8]:
    submission = []
    for qi, q in enumerate(questions):
        scored = results[qi] or []
        top = [c for _, c in scored[:top_k] if c.get('law_id','')]

        answer_parts, relevant_docs, relevant_articles = [], [], []
        seen_docs, seen_articles = set(), set()
        for r in top:
            law_id = r.get('law_id','')
            doc_name = make_doc_name(r.get('doc_title',''), r.get('doc_description',''))
            art_no = r.get('article_number','')
            doc_key = f'{law_id}|{doc_name}'
            if doc_key not in seen_docs and law_id:
                seen_docs.add(doc_key); relevant_docs.append(doc_key)
            art_key = f'{law_id}|{doc_name}|Điều {art_no}'
            if art_key not in seen_articles and law_id and art_no:
                seen_articles.add(art_key); relevant_articles.append(art_key)
                title = r.get('article_title','')
                answer_parts.append(f'Theo Điều {art_no} {doc_name}: {title}' if title else f'Căn cứ Điều {art_no} {doc_name}')

        submission.append({
            'id': q['id'], 'question': q['question'],
            'answer': '. '.join(answer_parts) + '.' if answer_parts else '',
            'relevant_docs': relevant_docs, 'relevant_articles': relevant_articles,
        })

    out = BASE / f'submission_3_1_inter_rerank_top{top_k}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    avg = np.mean([len(s['relevant_articles']) for s in submission])
    print(f'  top-{top_k}: avg {avg:.1f} articles -> {out.name}')

print(f'\n$(date): DONE')
"

echo "$(date): DONE"
