"""Local cross-encoder reranker (highest-ROI retrieval stage).

Loads AITeamVN/Vietnamese_Reranker (BGE-reranker-v2-m3, an XLM-R sequence
classifier) directly via transformers — NOT sentence-transformers CrossEncoder,
which in ST 5.x routes through AutoProcessor and fails on this text-only model.

Given a query and candidate docs (from hybrid RRF), scores each (query, doc)
pair and returns the top-k re-sorted.
"""
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import local_models_config as cfg

_BATCH = 16


class LocalReranker:
    def __init__(self):
        self.device = cfg.get_device()
        print(f"Loading reranker '{cfg.RERANKER_MODEL}' on {self.device} ...")
        self.tok = AutoTokenizer.from_pretrained(cfg.RERANKER_MODEL)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(cfg.RERANKER_MODEL)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top_k docs re-sorted by cross-encoder relevance score."""
        if not docs:
            return []
        pairs = [[query, f"{d.get('title', '')} {d.get('text', '')}"] for d in docs]
        scores: List[float] = []
        for i in range(0, len(pairs), _BATCH):
            batch = pairs[i:i + _BATCH]
            inputs = self.tok(
                batch, padding=True, truncation=True,
                max_length=cfg.RERANK_MAX_LEN, return_tensors="pt",
            ).to(self.device)
            logits = self.model(**inputs).logits.view(-1).float()
            scores.extend(logits.cpu().tolist())
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:top_k]
