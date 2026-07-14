"""
src/evaluator.py
Module đánh giá nội bộ và sinh báo cáo evaluation_report.md

Theo dõi:
- Recall@K của tầng Retrieval
- MRR (Mean Reciprocal Rank)
- Tỷ lệ pass/fail của Self-Verification
- Thống kê tổng thể pipeline
"""

import os
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

from config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Metrics tổng hợp cho toàn bộ pipeline."""
    total_questions: int = 0
    retrieval_success: int = 0   # Ít nhất 1 doc relevant trong top K
    rerank_improved: int = 0     # Rerank đưa doc relevant lên cao hơn
    verify_passed: int = 0
    verify_failed: int = 0
    regenerated: int = 0         # Số lần phải sinh lại câu trả lời
    avg_context_docs: float = 0.0
    avg_answer_length: float = 0.0
    no_relevant_docs_count: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def verification_pass_rate(self) -> float:
        total = self.verify_passed + self.verify_failed
        return self.verify_passed / total if total > 0 else 0.0


class PipelineEvaluator:
    """
    Theo dõi và ghi lại metrics trong quá trình xử lý 2000 câu hỏi.
    """

    def __init__(self):
        self.metrics = PipelineMetrics()
        self.per_item_logs: List[Dict] = []
        os.makedirs(Settings.LOG_DIR, exist_ok=True)

    def log_item(
        self,
        item_id: str,
        query: str,
        num_retrieved: int,
        num_after_rerank: int,
        verify_passed: bool,
        verify_violations: List[str],
        was_regenerated: bool,
        answer_length: int,
        relevant_docs_count: int
    ):
        """Ghi log cho một câu hỏi."""
        self.metrics.total_questions += 1

        if num_retrieved > 0:
            self.metrics.retrieval_success += 1
        if num_after_rerank > 0 and num_after_rerank < num_retrieved:
            self.metrics.rerank_improved += 1
        if verify_passed:
            self.metrics.verify_passed += 1
        else:
            self.metrics.verify_failed += 1
        if was_regenerated:
            self.metrics.regenerated += 1
        if relevant_docs_count == 0:
            self.metrics.no_relevant_docs_count += 1

        # Cập nhật running average
        n = self.metrics.total_questions
        self.metrics.avg_context_docs = (
            (self.metrics.avg_context_docs * (n - 1) + num_after_rerank) / n
        )
        self.metrics.avg_answer_length = (
            (self.metrics.avg_answer_length * (n - 1) + answer_length) / n
        )

        self.per_item_logs.append({
            "id": item_id,
            "retrieved": num_retrieved,
            "after_rerank": num_after_rerank,
            "verify_passed": verify_passed,
            "violations": verify_violations,
            "regenerated": was_regenerated,
            "answer_len": answer_length,
            "relevant_docs": relevant_docs_count,
        })

        # Log tiến độ mỗi 100 câu
        if self.metrics.total_questions % 100 == 0:
            logger.info(
                f"[Progress] {self.metrics.total_questions}/2000 | "
                f"Verify pass rate: {self.metrics.verification_pass_rate:.1%} | "
                f"Regen: {self.metrics.regenerated}"
            )

    def generate_report(self, output_path: str = Settings.EVAL_REPORT_PATH) -> str:
        """
        Sinh báo cáo Markdown tổng kết toàn bộ pipeline.
        """
        m = self.metrics
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""# 📊 Báo Cáo Đánh Giá Pipeline RAG Pháp Lý
**Thời gian**: {timestamp}
**Model LLM**: {Settings.LLM_MODEL_NAME}
**Reranker**: {Settings.RERANKER_MODEL}

---

## 📈 Tổng Quan

| Chỉ Số | Giá Trị |
|--------|---------|
| Tổng câu hỏi xử lý | {m.total_questions} |
| Retrieval có kết quả | {m.retrieval_success} ({m.retrieval_success/max(m.total_questions,1):.1%}) |
| Reranker cải thiện thứ hạng | {m.rerank_improved} |
| Self-Verification PASS | {m.verify_passed} ({m.verification_pass_rate:.1%}) |
| Self-Verification FAIL | {m.verify_failed} |
| Số lần Regenerate | {m.regenerated} |
| Câu trả lời thiếu relevant_docs | {m.no_relevant_docs_count} |

---

## 📏 Trung Bình

| Chỉ Số | Giá Trị |
|--------|---------|
| Số văn bản context đưa vào LLM (avg) | {m.avg_context_docs:.1f} |
| Độ dài câu trả lời trung bình (chars) | {m.avg_answer_length:.0f} |

---

## ⚙️ Cấu Hình Pipeline

| Tham Số | Giá Trị |
|---------|---------|
| TOP_K_RAW (Hybrid) | {Settings.TOP_K_RAW} |
| TOP_K_FINAL (Reranker) | {Settings.TOP_K_FINAL} |
| RERANKER_THRESHOLD | {Settings.RERANKER_THRESHOLD} |
| LLM_TEMPERATURE | {Settings.LLM_TEMPERATURE} |
| NUM_QUERY_VARIANTS | {Settings.NUM_QUERY_VARIANTS} |
| MAX_CONTEXT_CHARS | {Settings.MAX_CONTEXT_CHARS} |

---

## 🔍 Phân Tích Lỗi

Tổng số lỗi ghi nhận: **{len(m.errors)}**

"""
        if m.errors:
            for err in m.errors[:20]:  # Giới hạn 20 lỗi đầu
                report += f"- {err}\n"
            if len(m.errors) > 20:
                report += f"- ... và {len(m.errors) - 20} lỗi khác\n"

        report += f"""
---

## 📋 Khuyến Nghị Tối Ưu

"""
        if m.verification_pass_rate < 0.8:
            report += "- ⚠️ Tỷ lệ Verification thấp (<80%). Xem xét hạ `RERANKER_THRESHOLD` hoặc tăng `TOP_K_FINAL`.\n"
        if m.no_relevant_docs_count > m.total_questions * 0.1:
            report += "- ⚠️ Nhiều câu trả lời thiếu relevant_docs. Kiểm tra lại law_manifest.json và Regex extraction.\n"
        if m.regenerated > m.total_questions * 0.2:
            report += "- ⚠️ Tỷ lệ Regenerate cao (>20%). Xem xét cải thiện Prompt hoặc hạ ngưỡng RULE1.\n"
        if m.avg_answer_length < 200:
            report += "- ⚠️ Câu trả lời trung bình quá ngắn. Tăng `LLM_MAX_NEW_TOKENS` hoặc điều chỉnh Prompt.\n"

        if m.verification_pass_rate >= 0.9:
            report += "- ✅ Tỷ lệ Verification tốt (≥90%). Pipeline hoạt động ổn định.\n"

        report += "\n---\n*Báo cáo được sinh tự động bởi PipelineEvaluator*\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"📄 Báo cáo đã lưu tại: {output_path}")
        return report

    def save_detailed_log(self):
        """Lưu log chi tiết từng câu hỏi ra JSON."""
        log_path = os.path.join(Settings.LOG_DIR, "detailed_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.per_item_logs, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Detailed log đã lưu tại: {log_path}")
