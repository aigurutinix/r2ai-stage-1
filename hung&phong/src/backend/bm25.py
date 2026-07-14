"""BM25 sparse index cho hybrid retrieval (lexical — khớp chính xác số ký hiệu/Điều).

Bù điểm yếu của dense embedding: bge-m3 "mờ" với mã văn bản (22/2023/QH15), số Điều.
Tokenizer giữ nguyên mã VB dạng "22/2023/QH15" + từ tiếng Việt + số đứng riêng.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

# "22/2023/QH15" | từ (có dấu TV) | số đứng riêng
_TOKEN = re.compile(r"\d+/[\d/\w.\-]+|[^\W\d_]+|\d+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


class BM25Index:
    """Index BM25 + metadata căn theo thứ tự corpus. search() trả format giống dense hit."""

    def __init__(self, bm25: BM25Okapi, metas: list[dict]) -> None:
        self.bm25 = bm25
        self.metas = metas  # mỗi phần tử = payload-like dict (kèm 'text')

    @classmethod
    def build(cls, texts: list[str], metas: list[dict]) -> "BM25Index":
        corpus = [tokenize(t) for t in texts]
        return cls(BM25Okapi(corpus), metas)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "metas": self.metas}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["bm25"], d["metas"])

    def search(self, query: str, top_k: int = 30) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        if not len(scores):
            return []
        k = min(top_k, len(scores))
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        out = []
        for i in idx:
            if scores[i] <= 0:
                continue
            m = self.metas[int(i)]
            out.append({"score": float(scores[i]), "payload": m, "id": m.get("_id")})
        return out
