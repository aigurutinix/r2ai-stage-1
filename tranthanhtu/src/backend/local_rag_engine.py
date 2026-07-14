"""Offline hybrid retrieval — numpy cosine (corpus_emb.npy) + BM25 → RRF → rerank → cutoff.

No ChromaDB: corpus is small (~15MB float32) so numpy brute-force is instant AND the
embeddings file is trivially incremental (embed_corpus.py appends new rows). Same backend
local ↔ Colab notebook.

Pipeline:  dense (Vietnamese_Embedding, raw query) ‖ sparse (BM25)
        →  RRF fusion  →  cross-encoder rerank  →  score-threshold cutoff.
"""
import json
import re
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import local_models_config as cfg
from local_reranker import LocalReranker
from retrieval_cutoff import apply_cutoff

# Optional Vietnamese word segmentation improves BM25 (dense model uses raw text).
try:
    from pyvi import ViTokenizer
    _HAS_PYVI = True
except ImportError:
    _HAS_PYVI = False


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    if _HAS_PYVI:
        text = ViTokenizer.tokenize(text)
    return re.findall(r"\w+", text, flags=re.UNICODE)


class LocalLegalRAGEngine:
    def __init__(self, use_reranker: bool = True, candidate_k: int = 50):
        self.candidate_k = candidate_k

        # Load corpus text + precomputed embeddings (from embed_corpus.py)
        self.corpus: List[Dict[str, Any]] = []
        with open(cfg.CORPUS_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and json.loads(line).get("doc_number"):
                    self.corpus.append(json.loads(line))
        self.docs_text = [f"{r['title']}\n{r['text']}" for r in self.corpus]
        self.corpus_emb = np.load(cfg.CORPUS_EMB).astype("float32")
        assert len(self.corpus_emb) == len(self.corpus), (
            f"corpus_emb ({len(self.corpus_emb)}) != corpus ({len(self.corpus)}) "
            f"— chạy lại: python embed_corpus.py"
        )
        print(f"Loaded {len(self.corpus)} articles + embeddings {self.corpus_emb.shape}")

        self.device = cfg.get_device()
        print(f"Loading embedding model '{cfg.EMBEDDING_MODEL}' on {self.device} ...")
        self.model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=self.device)
        self.model.max_seq_length = cfg.EMBED_MAX_SEQ_LEN

        self.reranker = LocalReranker() if use_reranker else None
        print(f"BM25 indexing {len(self.docs_text)} articles (pyvi={'on' if _HAS_PYVI else 'off'}) ...")
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.docs_text])

    def _dense(self, query: str) -> List[int]:
        qv = self.model.encode(query, normalize_embeddings=True).astype("float32")
        scores = self.corpus_emb @ qv               # cosine (vectors are normalized)
        return [int(i) for i in np.argsort(scores)[::-1][:self.candidate_k]]

    def _sparse(self, query: str) -> List[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        return [int(i) for i in np.argsort(scores)[::-1][:self.candidate_k] if scores[i] > 0.0]

    def _candidates_scored(self, query: str, pool: int = 10) -> List[Dict[str, Any]]:
        """Dense+sparse → RRF → rerank → top `pool` deduped, each carrying rerank_score."""
        if not self.corpus:
            return []
        rrf: Dict[int, float] = {}
        for w, ids in ((cfg.RRF_W_DENSE, self._dense(query)), (cfg.RRF_W_BM25, self._sparse(query))):
            for rank, i in enumerate(ids):
                rrf[i] = rrf.get(i, 0.0) + w / (cfg.RRF_K + rank + 1)
        if not rrf:
            return []
        fused = sorted(rrf, key=rrf.get, reverse=True)[:self.candidate_k]
        candidates = [dict(self.corpus[i]) for i in fused]
        ranked = (self.reranker.rerank(query, candidates, top_k=pool)
                  if self.reranker else candidates[:pool])
        seen, out = set(), []
        for d in ranked:
            k = (d.get("doc_number", ""), d.get("article", ""))
            if k not in seen:
                seen.add(k)
                out.append(d)
        return out

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Return relevant article dicts, cut by reranker-score threshold (config-driven)."""
        top_k = top_k or cfg.RETRIEVE_TOP_K
        ranked = self._candidates_scored(query, pool=max(top_k, cfg.RETRIEVE_CAND_SAVE))
        return apply_cutoff(ranked, top_k, cfg.RETRIEVE_MIN_SCORE, cfg.RETRIEVE_MARGIN,
                            score_key="rerank_score")


if __name__ == "__main__":
    eng = LocalLegalRAGEngine()
    q = "Người lao động tự ý bỏ việc bao nhiêu ngày thì bị sa thải?"
    print(f"\nQuery: {q}\n")
    for i, r in enumerate(eng.retrieve(q), 1):
        s = r.get("rerank_score")
        head = f"    score={s:.3f} " if s is not None else "    "
        print(f"[{i}] {r.get('doc_number')} | {r.get('article')} | {r.get('clean_name')}")
        print(head + r.get("title", "")[:80])
