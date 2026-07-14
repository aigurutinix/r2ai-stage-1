"""Cross-encoder reranker (bge-reranker-v2-m3) — chấm lại (câu hỏi, đoạn) sâu hơn
bi-encoder, tăng precision@k cho cấp Điều. Tải model lần đầu (~2.2GB), chạy GPU fp16.
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")  # tránh xung đột Keras 3 trong transformers

import time
from functools import lru_cache

# Đổi reranker qua env RERANKER_MODEL:
#   - HF cross-encoder local: vd AITeamVN/Vietnamese_Reranker (VN-specific) hoặc BAAI/bge-reranker-v2-m3
#   - Cohere API:  RERANKER_MODEL=cohere  hoặc  cohere:rerank-v3.5  (cần COHERE_API_KEY trong env)
_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
_MAX_CHARS = 2000  # cắt đoạn dài cho reranker

_IS_COHERE = _MODEL.lower().startswith("cohere")
_COHERE_MODEL = _MODEL.split(":", 1)[1] if ":" in _MODEL else "rerank-v3.5"


@lru_cache(maxsize=1)
def _get_reranker():
    from FlagEmbedding import FlagReranker
    return FlagReranker(_MODEL, use_fp16=True)


def _rerank_cohere(query: str, hits: list[dict], top_k: int | None = None) -> list[dict]:
    """Chấm điểm bằng Cohere Rerank API (REST). KHÔNG re-embed — chỉ sắp lại tập đã fetch."""
    import requests
    key = os.environ.get("COHERE_API_KEY", "")
    if not key:  # fallback: đọc trực tiếp .env (reranker dùng os.environ, không qua pydantic)
        try:
            for _ln in open(".env", encoding="utf-8"):
                if _ln.strip().startswith("COHERE_API_KEY="):
                    key = _ln.strip().split("=", 1)[1]; os.environ["COHERE_API_KEY"] = key; break
        except Exception:  # noqa: BLE001
            pass
    docs = [(h.get("payload", {}).get("text") or "")[:_MAX_CHARS] for h in hits]
    payload = {"model": _COHERE_MODEL, "query": query, "documents": docs,
               "top_n": len(docs)}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(8):
        try:
            r = requests.post("https://api.cohere.com/v2/rerank", json=payload,
                              headers=headers, timeout=60)
            if r.status_code == 429:  # rate limit (trial=10/phút) → chờ cửa sổ phút reset
                time.sleep(62)
                continue
            r.raise_for_status()
            for h in hits:
                h["rerank_score"] = 0.0
            for item in r.json().get("results", []):
                hits[item["index"]]["rerank_score"] = float(item["relevance_score"])
            ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
            return ranked[:top_k] if top_k else ranked
        except requests.HTTPError:
            raise
        except Exception:  # noqa: BLE001 — lỗi mạng tạm thời → thử lại
            time.sleep(2 ** attempt)
    raise RuntimeError("Cohere rerank thất bại sau 5 lần thử")


def rerank(query: str, hits: list[dict], top_k: int | None = None) -> list[dict]:
    """Gán h['rerank_score'] (0..1) cho từng hit, trả về đã sắp giảm dần (cắt top_k)."""
    if not hits:
        return []
    if _IS_COHERE:
        return _rerank_cohere(query, hits, top_k)
    rr = _get_reranker()
    pairs = [[query, (h.get("payload", {}).get("text") or "")[:_MAX_CHARS]] for h in hits]
    scores = rr.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    for h, sc in zip(hits, scores):
        h["rerank_score"] = float(sc)
    ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
    return ranked[:top_k] if top_k else ranked
