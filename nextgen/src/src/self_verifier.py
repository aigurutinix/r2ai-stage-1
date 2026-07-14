"""
src/self_verifier.py
Giai đoạn 5: Tự Kiểm Tra Chống Ảo Giác (Self-Verification)

5 Quy tắc vàng kiểm tra câu trả lời trước khi xuất ra.
Nếu vi phạm → kích hoạt Regenerate với temperature thấp hơn.
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from config.settings import Settings

logger = logging.getLogger(__name__)

# =========================================================
# Regex patterns
# =========================================================
# Nhận diện tham chiếu điều luật: "Điều 5", "Điều 12a", "Điều 3, Khoản 2"
ARTICLE_REF_PATTERN = re.compile(
    r'[Đđ]iều\s+\d+[a-z]?(?:\s*[,;]\s*[Kk]hoản\s+\d+)?',
    re.IGNORECASE
)

# Nhận diện cụm trích dẫn bắt buộc: "Căn cứ Điều X" hoặc "Theo quy định tại Điều X"
CITATION_TRIGGER_PATTERN = re.compile(
    r'(Căn\s+cứ\s+[Đđ]iều|Theo\s+quy\s+định\s+tại\s+[Đđ]iều|[Tt]heo\s+[Đđ]iều)',
    re.IGNORECASE
)

# Nhận diện số hiệu văn bản trong text
DOC_NUMBER_IN_TEXT = re.compile(r'\d{1,3}/\d{4}/[A-ZĐ0-9/-]+')


@dataclass
class VerificationResult:
    """Kết quả kiểm tra từng câu trả lời."""
    passed: bool = True
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extracted_articles: List[str] = field(default_factory=list)
    extracted_doc_numbers: List[str] = field(default_factory=list)

    def add_violation(self, rule_id: str, message: str):
        self.passed = False
        self.violations.append(f"[{rule_id}] {message}")
        logger.warning(f"⚠️ Vi phạm {rule_id}: {message}")

    def add_warning(self, message: str):
        self.warnings.append(message)
        logger.debug(f"[Warning] {message}")


class SelfVerifier:
    """
    Kiểm tra câu trả lời theo 5 quy tắc vàng chống ảo giác pháp lý.
    """

    def __init__(self, law_manifest: Optional[Dict] = None):
        """
        Args:
            law_manifest: Dict ánh xạ số hiệu → tên văn bản chuẩn.
                         Load từ data/law_manifest.json
        """
        self.law_manifest = law_manifest or {}
        logger.info(f"✅ SelfVerifier khởi tạo với {len(self.law_manifest)} văn bản trong manifest.")

    @classmethod
    def from_manifest_file(cls, path: str = Settings.LAW_MANIFEST_PATH) -> "SelfVerifier":
        """Factory method: Load manifest từ file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            return cls(law_manifest=manifest)
        except FileNotFoundError:
            logger.warning(f"Không tìm thấy law_manifest.json tại {path}. Chạy không có manifest.")
            return cls(law_manifest={})

    # =========================================================
    # RULE 1: Điều được trích dẫn có trong Context không?
    # =========================================================
    def _rule1_articles_in_context(
        self, answer: str, context_docs: List[Dict]
    ) -> Tuple[bool, List[str]]:
        """
        Kiểm tra: Mỗi "Điều X" trong câu trả lời phải xuất hiện
        trong ít nhất một trong các văn bản context.
        """
        answer_articles = ARTICLE_REF_PATTERN.findall(answer)
        if not answer_articles:
            return True, []  # Không có tham chiếu → pass (sẽ bị Rule 5 bắt)

        # Gộp toàn bộ text context VÀ article_id trong metadata. Nhiều chunk có số Điều
        # nằm ở metadata.article_id (vd "Điều 12") chứ KHÔNG nằm trong thân text — nếu chỉ
        # đối chiếu text sẽ báo nhầm "ảo giác" cho câu trả lời đúng (false positive).
        context_text = " ".join(
            (doc.get("text", "") + " " + (doc.get("metadata", {}) or {}).get("article_id", ""))
            for doc in context_docs
        ).lower()

        hallucinated = []
        for art_ref in answer_articles:
            # Normalize: "Điều 5" → "điều 5"
            art_lower = art_ref.lower().strip()
            # Kiểm tra số điều có trong context không
            art_num = re.search(r'\d+', art_lower)
            if art_num:
                search_pattern = f"điều {art_num.group()}"
                if search_pattern not in context_text:
                    hallucinated.append(art_ref)

        return len(hallucinated) == 0, hallucinated

    # =========================================================
    # RULE 2: Số hiệu điều luật có tồn tại trong manifest?
    # =========================================================
    def _rule2_doc_numbers_valid(self, answer: str) -> Tuple[bool, List[str]]:
        """
        Kiểm tra: Số hiệu văn bản pháp luật được đề cập phải có trong law_manifest.
        """
        if not self.law_manifest:
            return True, []  # Không có manifest → bỏ qua rule này

        found_numbers = DOC_NUMBER_IN_TEXT.findall(answer)
        if not found_numbers:
            return True, []

        invalid_numbers = []
        for doc_num in found_numbers:
            # Tìm trong manifest (kiểm tra theo key hoặc nested value)
            if not self._doc_number_in_manifest(doc_num):
                invalid_numbers.append(doc_num)

        return len(invalid_numbers) == 0, invalid_numbers

    def _doc_number_in_manifest(self, doc_number: str) -> bool:
        """Kiểm tra số hiệu có trong manifest không."""
        doc_num_clean = doc_number.strip()

        # Manifest có thể là dict dạng {"80/2021/NĐ-CP": {...}} hoặc List
        if isinstance(self.law_manifest, dict):
            for key in self.law_manifest:
                if doc_num_clean in key or key in doc_num_clean:
                    return True
        elif isinstance(self.law_manifest, list):
            for item in self.law_manifest:
                if isinstance(item, dict):
                    identifier = item.get("doc_number", "") or item.get("so_hieu", "")
                    if doc_num_clean in identifier or identifier in doc_num_clean:
                        return True
        return False

    # =========================================================
    # RULE 3: Tên văn bản khớp với law_manifest?
    # =========================================================
    def _rule3_doc_names_consistent(
        self, answer: str, context_docs: List[Dict]
    ) -> Tuple[bool, List[str]]:
        """
        Kiểm tra: Tên văn bản pháp luật trong câu trả lời phải
        khớp với tên trong manifest, không được sai trích dẫn.
        """
        # Lấy các số hiệu có trong context
        context_doc_numbers = set()
        context_doc_names = {}
        for doc in context_docs:
            meta = doc.get("metadata", {})
            dn = meta.get("doc_number", "")
            dname = meta.get("title", "")
            if dn:
                context_doc_numbers.add(dn)
                context_doc_names[dn] = dname

        if not context_doc_numbers or not self.law_manifest:
            return True, []

        inconsistencies = []
        for doc_num in context_doc_numbers:
            if doc_num in answer:
                # Lấy tên chuẩn từ manifest
                manifest_entry = self.law_manifest.get(doc_num, {})
                if isinstance(manifest_entry, dict):
                    canonical_name = manifest_entry.get("law_name", "")
                else:
                    canonical_name = str(manifest_entry)

                # Chỉ cảnh báo nếu manifest có tên nhưng không tìm thấy trong câu trả lời
                if canonical_name and len(canonical_name) > 5:
                    # Lấy 4-5 từ đầu của tên chuẩn để kiểm tra
                    name_fragment = " ".join(canonical_name.split()[:4]).lower()
                    if name_fragment not in answer.lower():
                        inconsistencies.append(
                            f"{doc_num}: expected '{canonical_name[:50]}...'"
                        )

        return len(inconsistencies) == 0, inconsistencies

    # =========================================================
    # RULE 4: Câu trả lời không chứa thông tin ngoài Context?
    # =========================================================
    def _rule4_no_out_of_context_claims(
        self, answer: str, context_docs: List[Dict]
    ) -> Tuple[bool, List[str]]:
        """
        Kiểm tra heuristic: Câu trả lời không chứa số liệu cụ thể
        (tỷ lệ %, ngày tháng, mức tiền) mà không có trong context.

        Đây là Rule heuristic nhẹ (dùng để cảnh báo, không block hard).
        """
        context_text = " ".join(doc.get("text", "") for doc in context_docs)

        # Tìm số liệu tỷ lệ phần trăm trong câu trả lời
        answer_percentages = re.findall(r'\d+(?:[,.]\d+)?%', answer)

        suspicious = []
        for pct in answer_percentages:
            pct_clean = pct.replace(",", ".")
            if pct_clean not in context_text and pct not in context_text:
                suspicious.append(pct)

        # Chỉ cảnh báo (warning), không fail hard
        return True, suspicious

    # =========================================================
    # RULE 5: Câu trả lời phải có ít nhất 1 tham chiếu "Điều X"
    # =========================================================
    def _rule5_has_article_reference(self, answer: str) -> bool:
        """
        Kiểm tra câu trả lời có chứa ít nhất một cụm "Điều X" không.
        """
        has_article = bool(ARTICLE_REF_PATTERN.search(answer))
        has_trigger = bool(CITATION_TRIGGER_PATTERN.search(answer))
        return has_article and has_trigger

    # =========================================================
    # MAIN VERIFY
    # =========================================================
    def verify(self, answer: str, query: str, context_docs: List[Dict]) -> VerificationResult:
        """
        Chạy toàn bộ 5 quy tắc kiểm tra.

        Args:
            answer: Câu trả lời từ LLM
            query: Câu hỏi gốc
            context_docs: Danh sách văn bản context đã dùng

        Returns:
            VerificationResult với trạng thái pass/fail và chi tiết vi phạm
        """
        result = VerificationResult()

        # Trích xuất thông tin cơ bản
        result.extracted_articles = ARTICLE_REF_PATTERN.findall(answer)
        result.extracted_doc_numbers = DOC_NUMBER_IN_TEXT.findall(answer)

        # --- RULE 1 ---
        r1_pass, hallucinated_arts = self._rule1_articles_in_context(answer, context_docs)
        if not r1_pass:
            result.add_violation(
                "RULE1",
                f"Các điều luật sau không có trong context: {hallucinated_arts}"
            )

        # --- RULE 2 ---
        r2_pass, invalid_nums = self._rule2_doc_numbers_valid(answer)
        if not r2_pass:
            result.add_violation(
                "RULE2",
                f"Số hiệu văn bản không tồn tại trong manifest: {invalid_nums}"
            )

        # --- RULE 3 ---
        r3_pass, inconsistencies = self._rule3_doc_names_consistent(answer, context_docs)
        if not r3_pass:
            # Rule 3 chỉ warning, không hard fail
            for inc in inconsistencies:
                result.add_warning(f"[RULE3] Tên văn bản có thể không nhất quán: {inc}")

        # --- RULE 4 ---
        _, suspicious_nums = self._rule4_no_out_of_context_claims(answer, context_docs)
        for sus in suspicious_nums:
            result.add_warning(f"[RULE4] Số liệu có thể ngoài context: {sus}")

        # --- RULE 5 ---
        r5_pass = self._rule5_has_article_reference(answer)
        if not r5_pass:
            result.add_violation(
                "RULE5",
                "Câu trả lời không chứa cụm trích dẫn pháp lý bắt buộc (Căn cứ Điều X / Theo quy định tại Điều X)"
            )

        if result.passed:
            logger.info(f"✅ Verification PASSED | {len(result.extracted_articles)} điều luật trích dẫn")
        else:
            logger.warning(
                f"❌ Verification FAILED | {len(result.violations)} vi phạm: "
                + " | ".join(result.violations)
            )

        return result
