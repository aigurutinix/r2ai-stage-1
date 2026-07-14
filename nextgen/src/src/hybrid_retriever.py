"""
src/hybrid_retriever.py
Giai đoạn 2: Tầng Truy Hồi Lai (Hybrid Retrieval) 
TỐI ƯU HOÀN TOÀN THEO TÀI LIỆU CHUẨN 2-STAGE SEARCH CỦA QDRANT
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams, QuantizationSearchParams, Prefetch

from config.settings import Settings
from src.index_bm25 import BM25IndexBuilder, tokenize_legal_text

logger = logging.getLogger(__name__)

DOC_NUMBER_PATTERN = re.compile(r'\d{1,3}/\d{4}/[A-ZĐ0-9/-]+')

class MultiQueryExpander:
    def __init__(self, llm_pipeline=None):
        self.llm = llm_pipeline

    def expand(self, query: str, num_variants: int = Settings.NUM_QUERY_VARIANTS) -> List[str]:
        variants = [query]
        variants.extend(self._rule_based_expand(query))
        # if self.llm is not None:
        #     try:
        #         prompt = f"<system>Bạn là chuyên gia ngôn ngữ. Hãy viết đúng {num_variants} câu hỏi tương đương với câu hỏi của người dùng. Không giải thích, không suy nghĩ dài dòng, không đánh số, mỗi câu 1 dòng.</system>\nNgười dùng: {query}\nCác biến thể:"
        #         output = self.llm(prompt, max_new_tokens=150, temperature=0.2, do_sample=True)
        #         generated = output[0]["generated_text"].replace(prompt, "").strip()
        #         generated = re.sub(r'<think>.*?</think>', '', generated, flags=re.DOTALL).strip()
        #         lines = [l.strip() for l in generated.split("\n") if l.strip() and len(l.strip()) > 10 and not l.strip().startswith(("<", "Đầu tiên", "Tôi có thể"))]
        #         variants.extend(lines[:num_variants])
        #     except Exception as e:
        #         logger.warning(f"LLM expansion thất bại, dùng fallback: {e}")
        #         variants.extend(self._rule_based_expand(query))
        # else:
        #     variants.extend(self._rule_based_expand(query))
        # return list(dict.fromkeys(variants))
        return list(dict.fromkeys(variants))[:num_variants]

    def _rule_based_expand(self, query: str) -> List[str]:
        expansions = []
        q_lower = query.lower()
        if "doanh nghiệp nhỏ và vừa" in q_lower:
            expansions.append(query.replace("doanh nghiệp nhỏ và vừa", "SME"))
        if "thuế" in q_lower:
            expansions.append("Quy định pháp luật về " + query)
        return expansions[:Settings.NUM_QUERY_VARIANTS - 1]

class HybridRetriever:
    def __init__(self, bm25_builder: BM25IndexBuilder, embedding_fn, llm_pipeline=None):
        self.bm25 = bm25_builder
        self.embed = embedding_fn
        self.expander = MultiQueryExpander(llm_pipeline)
        self.qdrant = QdrantClient(url=Settings.QDRANT_URL, api_key=Settings.QDRANT_API_KEY, timeout=30)
        logger.info("✅ HybridRetriever khởi tạo thành công với cấu trúc 2-Stage Prefetch & Rescore.")

    def _extract_doc_filter(self, query: str) -> Optional[Filter]:
        match = DOC_NUMBER_PATTERN.search(query)
        if match:
            doc_number = match.group(0)
            # Payload thực tế trên Qdrant Cloud là field phẳng "doc_id" (không có wrapper "metadata")
            return Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_number))])
        return None

    @staticmethod
    def _normalize_payload(payload: Dict) -> Dict:
        """
        Chuẩn hóa payload Qdrant về cùng cấu trúc metadata với corpus BM25 (law_corpus_clean.json):
        {"doc_number", "title", "article_id", "unique_article_id", "clause_num", "context"}.
        Payload thực tế trên Qdrant Cloud là field phẳng (doc_id, title, article_id, ...),
        không có wrapper "metadata" lồng bên trong.
        """
        if isinstance(payload.get("metadata"), dict):
            return payload["metadata"]
        return {
            "doc_number": payload.get("doc_id", ""),
            "title": payload.get("title", ""),
            "article_id": payload.get("article_id", ""),
            "unique_article_id": payload.get("unique_article_id", payload.get("chunk_id", "")),
            "clause_num": payload.get("clause_num"),
            "context": payload.get("context", {}),
        }

    def _dense_search(self, query: str, query_vector: List[float], limit: int = Settings.TOP_K_RAW) -> List[Dict]:
        """
        Tìm kiếm ngữ nghĩa sử dụng cấu trúc truy vấn 2 giai đoạn (Prefetch + Rescore) 
        đúng chính xác theo tài liệu hướng dẫn kỹ thuật của Qdrant.
        """
        meta_filter = self._extract_doc_filter(query)
        
        # Cấu hình hệ số Oversampling để quét rộng trên RAM trước khi load từ đĩa (disk)
        # Giả định lấy rộng gấp 10 lần giới hạn cần lấy thô để đảm bảo Recall
        rescore_limit = limit * 10 

        try:
            response = self.qdrant.query_points(
                collection_name=Settings.COLLECTION_NAME,
                query=query_vector,          # Truyền vector trực tiếp qua trường query
                query_filter=meta_filter,
                limit=limit,                 # Số lượng kết quả thực tế sau khi đã rescore ở Disk
                
                # Giai đoạn 2: Bật rescore trên Đĩa cứng (Disk)
                search_params=SearchParams(
                    quantization=QuantizationSearchParams(
                        rescore=True,
                    ),
                ),
                
                # Giai đoạn 1: Quét nhanh diện rộng (Prefetch) dữ liệu được lưu trên RAM
                prefetch=Prefetch(
                    query=query_vector,
                    limit=rescore_limit,
                    params=SearchParams(
                        quantization=QuantizationSearchParams(
                            rescore=False,   # Tránh kích hoạt rescore sớm ở tầng prefetch trên RAM
                        ),
                    )
                ),
                with_payload=True
            )
            
            # Cơ chế Fallback an toàn: Nếu bộ lọc metadata quá chặt dẫn đến rỗng kết quả, tự động search diện rộng
            if not response.points and meta_filter is not None:
                logger.warning(f"[Qdrant] Filter cứng bị rỗng kết quả, tiến hành kích hoạt Fallback mở rộng (No Filter).")
                response = self.qdrant.query_points(
                    collection_name=Settings.COLLECTION_NAME,
                    query=query_vector,
                    limit=limit,
                    search_params=SearchParams(
                        quantization=QuantizationSearchParams(rescore=True)
                    ),
                    prefetch=Prefetch(
                        query=query_vector,
                        limit=rescore_limit,
                        params=SearchParams(quantization=QuantizationSearchParams(rescore=False))
                    ),
                    with_payload=True
                )

            return [
                {
                    # Dùng unique_article_id (== id corpus, vd "04/2017/QH14_Điều 12") làm khóa
                    # thay cho point-id UUID của Qdrant, để RRF dedupe khớp với nhánh BM25/sparse
                    # (cùng một Điều ở 2 nhánh được cộng điểm thay vì bị coi là 2 tài liệu khác nhau).
                    "id": (p.payload.get("unique_article_id")
                           or p.payload.get("chunk_id")
                           or str(p.id)),
                    "text": p.payload.get("text", ""),
                    "dense_score": float(p.score if p.score else 0.0),
                    "metadata": self._normalize_payload(p.payload)
                }
                for p in response.points
            ]
        except Exception as e:
            logger.error(f"❌ Lỗi thực thi Qdrant query_points (2-Stage): {e}")
            return []

    def _sparse_search(self, query: str, limit: int = Settings.TOP_K_RAW) -> List[Dict]:
        return self.bm25.search(query, top_k=limit)

    def _rrf_merge(self, ranked_lists: List[List[Dict]], k: int = Settings.RRF_K) -> List[Dict]:
        rrf_scores: Dict[str, float] = {}
        doc_store: Dict[str, Dict] = {}
        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list, start=1):
                doc_id = doc["id"]
                if doc_id not in doc_store:
                    doc_store[doc_id] = doc
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in sorted_docs[:Settings.TOP_K_RAW]:
            doc = dict(doc_store[doc_id])
            doc["rrf_score"] = score
            results.append(doc)
        return results

    def _multi_query_retrieve(self, query: str) -> Tuple[List[Dict], List[Dict]]:
        query_variants = self.expander.expand(query)
        logger.info(f"Query variants: {query_variants}")
        all_dense: List[Dict] = []
        all_sparse: List[Dict] = []
        seen_dense_ids = set()
        seen_sparse_ids = set()

        for variant in query_variants:
            query_vector = self.embed(variant)
            dense_res = self._dense_search(variant, query_vector)
            for doc in dense_res:
                if doc["id"] not in seen_dense_ids:
                    all_dense.append(doc)
                    seen_dense_ids.add(doc["id"])
            sparse_res = self._sparse_search(variant)
            for doc in sparse_res:
                if doc["id"] not in seen_sparse_ids:
                    all_sparse.append(doc)
                    seen_sparse_ids.add(doc["id"])
        return all_dense, all_sparse

    def retrieve(self, query: str) -> List[Dict]:
        logger.info(f"[Retrieval] Query: {query[:80]}...")
        dense_results, sparse_results = self._multi_query_retrieve(query)
        logger.info(f"Dense: {len(dense_results)} docs, Sparse: {len(sparse_results)} docs")
        merged = self._rrf_merge([dense_results, sparse_results])
        logger.info(f"Sau RRF: {len(merged)} docs")
        return merged