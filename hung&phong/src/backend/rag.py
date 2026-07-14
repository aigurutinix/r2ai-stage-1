"""RAG pipeline: embed query → retrieve → build prompt → call LLM."""
from __future__ import annotations

import logging
import re
from typing import Iterator

from backend.config import get_settings
from backend.embed import EmbeddingClient
from backend.llm import LLMClient
from backend.prompts import SYSTEM_PROMPT, USER_TEMPLATE, format_context
from backend.qdrant_store import QdrantStore
from backend.query_analyzer import analyze
from backend.textnorm import normalize_vn

logger = logging.getLogger(__name__)

# Re-rank mềm (không lọc cứng): ưu tiên văn bản còn hiệu lực + luật lõi quốc gia.
# Chỉ phạt "HẾT HIỆU LỰC TOÀN BỘ" (đã bị thay thế hẳn). KHÔNG phạt "một phần" vì
# luật sửa đổi một phần VẪN là văn bản hiện hành (vd Luật DN 2020, SHTT 2005) —
# phạt sẽ hạ oan căn cứ đúng (đã đo bằng A/B trên gold set).
_PENALTY_TOAN_BO = 0.10
_NQ_LOCAL_PENALTY = 0.05      # hạ Nghị quyết HĐND địa phương (nhiễu)
_PRIMARY_BOOST = 0.02        # ưu tiên Luật/Bộ luật/Pháp lệnh quốc gia
_CORE_PRIMARY = frozenset({"Luật", "Bộ luật", "Pháp lệnh"})


def _rerank_adjust(payload: dict) -> float:
    """Lượng điều chỉnh điểm (cộng vào cosine) theo hiệu lực + loại văn bản."""
    adj = 0.0
    st = str(payload.get("tinh_trang_hieu_luc") or "").lower()
    if "hết hiệu lực toàn bộ" in st:
        adj -= _PENALTY_TOAN_BO
    loai = str(payload.get("loai_van_ban") or "")
    sk = str(payload.get("so_ky_hieu") or "")
    if loai == "Nghị quyết" and "HĐND" in sk:
        adj -= _NQ_LOCAL_PENALTY
    elif loai in _CORE_PRIMARY:
        adj += _PRIMARY_BOOST
    # Ưu tiên bản MỚI (vbpl_v2 không có cột hiệu lực → dùng năm ban hành để
    # đẩy bản hiện hành lên trên bản cũ cùng tên, vd 59/2020 > 68/2014).
    nam = payload.get("nam") or ""
    m = re.search(r"(19|20)\d{2}", str(nam))
    if m:
        adj += min(0.05, max(0.0, (int(m.group()) - 2000) * 0.002))
    return adj


class RAGPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = EmbeddingClient()
        self.llm = LLMClient()
        self.store = QdrantStore()
        self._bm25 = None
        if self.settings.hybrid_search:
            from backend.bm25 import BM25Index
            path = self.settings.bm25_index_path or f"data/bm25_{self.settings.qdrant_collection}.pkl"
            logger.info("Nạp BM25 index: %s", path)
            self._bm25 = BM25Index.load(path)
        self._hnsw = None
        if self.settings.use_hnsw:
            from backend.hnsw_index import HnswDense
            base = (self.settings.hnsw_index_path
                    or f"C:/Users/PHONG/vbpl_idx/hnsw_{self.settings.qdrant_collection}")
            logger.info("Nạp HNSW index: %s", base)
            self._hnsw = HnswDense.load(f"{base}.bin", f"{base}_meta.pkl")

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Hybrid retrieve: filter-first khi detect tín hiệu cứng, fallback vector-only.

        Pure-vector miss khi query chứa số ký hiệu cụ thể (vd "159/SL").
        Strategy:
          1. Detect số ký hiệu / Điều / loại văn bản từ query.
          2. Nếu có filter → search có filter trước.
          3. Nếu 0 hit (filter quá chặt hoặc payload chưa index) → fallback
             sang vector-only để không trả về rỗng.
        """
        query = normalize_vn(query)  # đối xứng với corpus đã chuẩn hoá (khỏan→khoản)
        k = top_k or self.settings.top_k
        s = self.settings
        hybrid = s.hybrid_search and self._bm25 is not None
        fetch_k = s.hybrid_fetch if (hybrid or s.use_reranker) else max(k * 3, k + 20)
        vec = self.embedder.embed_one(query)

        # --- Dense: HNSW (nhanh) nếu bật, ngược lại Qdrant (có filter-first) ---
        dense: list[dict] = []
        if self._hnsw is not None:
            dense = self._hnsw.search(vec, top_k=fetch_k)
        else:
            filters = analyze(query)
            if filters.has_any():
                dense = self.store.search(query_vector=vec, top_k=fetch_k,
                                          filter_must=filters.to_qdrant_filter())
            if not dense:
                dense = self.store.search(query_vector=vec, top_k=fetch_k)

        # --- Gộp ứng viên dense ∪ BM25 (lexical) ---
        cands: dict[str, dict] = {self._hid(h): h for h in dense}
        bm25_hits: list[dict] = []
        if hybrid:
            bm25_hits = self._bm25.search(query, top_k=fetch_k)
            for h in bm25_hits:
                cands.setdefault(self._hid(h), h)
        cand_list = list(cands.values())

        # --- Chấm điểm: reranker > RRF > cosine ---
        if s.use_reranker and cand_list:
            from backend import reranker
            reranker.rerank(query, cand_list)               # gán rerank_score
            base = lambda h: h.get("rerank_score", 0.0)
        elif hybrid:
            rrf = self._rrf_scores(dense, bm25_hits)
            base = lambda h: rrf.get(self._hid(h), 0.0)
        else:
            base = lambda h: (h.get("score") or 0.0)

        for h in cand_list:
            h["adj_score"] = base(h) + _rerank_adjust(h.get("payload", {}))
        ranked = sorted(cand_list, key=lambda h: h["adj_score"], reverse=True)

        # Dedup theo (so_ky_hieu, dieu_so) — bỏ chunk/doc trùng, giữ điểm cao nhất,
        # giải phóng slot cho điều đa dạng hơn → recall tốt hơn.
        seen: set = set()
        out: list[dict] = []
        for h in ranked:
            p = h.get("payload", {})
            key = (str(p.get("so_ky_hieu")), p.get("dieu_so"))
            if key in seen:
                continue
            seen.add(key)
            out.append(h)
            if len(out) >= k:
                break
        return out

    @staticmethod
    def _hid(h: dict) -> str:
        """Khoá định danh ứng viên để dedup dense ∪ bm25."""
        p = h.get("payload", {})
        return str(h.get("id") or f"{p.get('so_ky_hieu')}#{p.get('dieu_so')}")

    @classmethod
    def _rrf_scores(cls, dense: list[dict], bm25: list[dict], k: int = 60) -> dict[str, float]:
        """Reciprocal Rank Fusion theo chunk id → {hid: score}."""
        agg: dict[str, float] = {}
        for hits in (dense, bm25):
            for rank, h in enumerate(hits):
                hid = cls._hid(h)
                agg[hid] = agg.get(hid, 0.0) + 1.0 / (k + rank)
        return agg

    @staticmethod
    def _rerank_by_status(hits: list[dict]) -> list[dict]:
        """Re-rank: điểm hiệu chỉnh = cosine + điều chỉnh(hiệu lực, loại VB)."""
        for h in hits:
            h["adj_score"] = (h.get("score") or 0.0) + _rerank_adjust(h.get("payload", {}))
        return sorted(hits, key=lambda h: h["adj_score"], reverse=True)

    def build_prompt(self, query: str, hits: list[dict]) -> tuple[str, str]:
        context = format_context(hits)
        user = USER_TEMPLATE.format(context=context, query=query)
        return SYSTEM_PROMPT, user

    def answer(self, query: str, top_k: int | None = None) -> dict:
        hits = self.retrieve(query, top_k=top_k)
        system, user = self.build_prompt(query, hits)
        answer = self.llm.complete(system, user)
        return {
            "answer": answer,
            "sources": [self._source_summary(h) for h in hits],
        }

    def stream_answer(self, query: str, top_k: int | None = None) -> Iterator[dict]:
        hits = self.retrieve(query, top_k=top_k)
        sources = [self._source_summary(h) for h in hits]
        yield {"type": "sources", "data": sources}

        system, user = self.build_prompt(query, hits)
        for token in self.llm.stream(system, user):
            yield {"type": "token", "data": token}

        yield {"type": "done", "data": None}

    @staticmethod
    def _source_summary(hit: dict) -> dict:
        p = hit.get("payload", {})
        return {
            "score": hit.get("score"),
            "doc_id": p.get("doc_id"),
            "so_ky_hieu": p.get("so_ky_hieu"),
            "loai_van_ban": p.get("loai_van_ban"),
            "title": p.get("title"),
            "dieu_so": p.get("dieu_so"),
            "dieu_tieu_de": p.get("dieu_tieu_de"),
            "tinh_trang_hieu_luc": p.get("tinh_trang_hieu_luc"),
            "ngay_ban_hanh": p.get("ngay_ban_hanh"),
        }
