"""
src/reranker.py
Giai đoạn 3: Tái Xếp Hạng (Reranking)

Dùng Cross-Encoder BAAI/bge-reranker-large để tinh lọc
Top 30 thô → Top 5-7 văn bản liên quan nhất.

Tự động phát hiện GPU (Kaggle T4/P100) hoặc fallback CPU.
"""

import logging
import time
from typing import List, Dict, Tuple

import torch
from sentence_transformers import CrossEncoder

from config.settings import Settings

logger = logging.getLogger(__name__)


class LegalReranker:
    """
    Cross-Encoder Reranker cho văn bản pháp lý tiếng Việt.
    Sử dụng BAAI/bge-reranker-large (hỗ trợ tiếng Việt tốt).
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[Reranker] Đang load model trên: {self.device.upper()}")

        # Khởi tạo Cross-Encoder
        self.model = CrossEncoder(
            Settings.RERANKER_MODEL,
            device=self.device,
            max_length=512  # Truncate để tránh OOM trên T4
        )
        logger.info(f"✅ Reranker '{Settings.RERANKER_MODEL}' load thành công.")

    def rerank(
        self,
        query: str,
        candidate_documents: List[Dict],
        threshold: float = Settings.RERANKER_THRESHOLD,
        top_k: int = Settings.TOP_K_FINAL
    ) -> List[Dict]:
        """
        Chấm điểm lại và lọc văn bản ứng viên.

        Args:
            query: Câu hỏi gốc
            candidate_documents: Top 30 từ tầng Hybrid
            threshold: Ngưỡng điểm tối thiểu để giữ văn bản
            top_k: Số văn bản tối đa trả về

        Returns:
            Danh sách văn bản đã rerank, lọc theo threshold, giới hạn top_k
        """
        if not candidate_documents:
            logger.warning("[Reranker] Danh sách candidate rỗng!")
            return []

        start_time = time.time()

        # Chuẩn bị cặp (query, passage) cho Cross-Encoder
        pairs = [[query, doc.get("text", "")] for doc in candidate_documents]

        # Predict scores (batch inference)
        with torch.no_grad():
            scores = self.model.predict(
                pairs,
                batch_size=16,     # Phù hợp với VRAM T4 (16GB)
                show_progress_bar=False
            )

        elapsed = time.time() - start_time
        logger.debug(f"[Reranker] Predict {len(pairs)} pairs trong {elapsed:.2f}s")

        # Gán điểm rerank vào document
        scored_docs = []
        for idx, (doc, score) in enumerate(zip(candidate_documents, scores)):
            doc_copy = dict(doc)
            doc_copy["rerank_score"] = float(score)
            scored_docs.append(doc_copy)

        # Sắp xếp theo điểm giảm dần
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Log phân phối điểm để debug
        all_scores = [d["rerank_score"] for d in scored_docs]
        logger.info(
            f"[Reranker] Score range: [{min(all_scores):.3f}, {max(all_scores):.3f}] "
            f"| Threshold: {threshold}"
        )

        # Lọc theo ngưỡng
        filtered = [d for d in scored_docs if d["rerank_score"] >= threshold]

        # Fallback: Nếu lọc quá chặt → giữ ít nhất top_k/2 kết quả tốt nhất
        if len(filtered) < max(1, top_k // 2):
            logger.warning(
                f"[Reranker] Chỉ {len(filtered)} doc vượt ngưỡng {threshold}. "
                f"Fallback: giữ top {top_k // 2} docs."
            )
            filtered = scored_docs[:max(1, top_k // 2)]

        result = filtered[:top_k]
        logger.info(f"[Reranker] Trả về {len(result)} docs sau lọc.")
        return result

    def hard_negative_analysis(
        self,
        query: str,
        positive_docs: List[Dict],
        negative_docs: List[Dict]
    ) -> Dict:
        """
        Phân tích Hard Negatives: Văn bản từ khóa gần nhưng không liên quan.
        Dùng để căn chỉnh ngưỡng Threshold.

        Args:
            query: Câu hỏi
            positive_docs: Văn bản đúng (ground truth)
            negative_docs: Văn bản nhiễu có từ khóa giống

        Returns:
            Dict thống kê để tinh chỉnh threshold
        """
        all_docs = positive_docs + negative_docs
        pairs = [[query, doc.get("text", "")] for doc in all_docs]

        with torch.no_grad():
            scores = self.model.predict(pairs, show_progress_bar=False)

        pos_scores = scores[:len(positive_docs)]
        neg_scores = scores[len(positive_docs):]

        analysis = {
            "positive_scores": {
                "min": float(min(pos_scores)) if pos_scores.size > 0 else 0,
                "max": float(max(pos_scores)) if pos_scores.size > 0 else 0,
                "mean": float(pos_scores.mean()) if pos_scores.size > 0 else 0,
            },
            "negative_scores": {
                "min": float(min(neg_scores)) if neg_scores.size > 0 else 0,
                "max": float(max(neg_scores)) if neg_scores.size > 0 else 0,
                "mean": float(neg_scores.mean()) if neg_scores.size > 0 else 0,
            },
            # Ngưỡng đề xuất: trung điểm giữa max(neg) và min(pos)
            "suggested_threshold": float(
                (max(neg_scores) + min(pos_scores)) / 2
            ) if neg_scores.size > 0 and pos_scores.size > 0 else Settings.RERANKER_THRESHOLD
        }

        logger.info(f"[Hard Negative Analysis] {analysis}")
        return analysis
