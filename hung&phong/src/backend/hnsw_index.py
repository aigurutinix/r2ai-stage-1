"""Dense ANN index (hnswlib) — thay Qdrant embedded brute-force để retrieval NHANH
(ms/câu thay vì 2-3s/câu). Dùng cho lặp nhanh khi tinh chỉnh.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import hnswlib
import numpy as np


class HnswDense:
    def __init__(self, index: "hnswlib.Index", metas: list[dict], dim: int) -> None:
        self.index = index
        self.metas = metas
        self.dim = dim

    @classmethod
    def build(cls, vectors, metas: list[dict], dim: int,
              ef_construction: int = 200, M: int = 32) -> "HnswDense":
        idx = hnswlib.Index(space="cosine", dim=dim)
        idx.init_index(max_elements=len(vectors), ef_construction=ef_construction, M=M)
        idx.add_items(np.asarray(vectors, dtype=np.float32), np.arange(len(vectors)))
        idx.set_ef(256)
        return cls(idx, metas, dim)

    def save(self, idx_path: str | Path, meta_path: str | Path) -> None:
        self.index.save_index(str(idx_path))
        with open(meta_path, "wb") as f:
            pickle.dump({"metas": self.metas, "dim": self.dim}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, idx_path: str | Path, meta_path: str | Path, ef: int = 256) -> "HnswDense":
        with open(meta_path, "rb") as f:
            d = pickle.load(f)
        idx = hnswlib.Index(space="cosine", dim=d["dim"])
        idx.load_index(str(idx_path), max_elements=len(d["metas"]))
        idx.set_ef(ef)
        return cls(idx, d["metas"], d["dim"])

    def search(self, vec, top_k: int = 40) -> list[dict]:
        labels, dists = self.index.knn_query(np.asarray(vec, dtype=np.float32), k=top_k)
        out = []
        for lab, dist in zip(labels[0], dists[0]):
            m = self.metas[int(lab)]
            out.append({"score": float(1.0 - dist), "payload": m, "id": m.get("_id")})
        return out
