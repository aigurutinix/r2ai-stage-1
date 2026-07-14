"""
src/post_processor.py
Giai đoạn 6: Hậu Xử Lý & Đóng Gói Bài Nộp (Post-Processing)

1. Regex Extraction: Bóc tách "Điều X", "Khoản Y", tên văn bản từ câu trả lời
2. Hard-Mapping: Tra cứu law_manifest.json để điền chuỗi chuẩn
3. Data Validation: Kiểm tra toàn bộ 2000 bản ghi
4. Submission Packaging: Xuất file results.json + zip
"""

import re
import os
import json
import zipfile
import logging
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime

from config.settings import Settings

logger = logging.getLogger(__name__)

# =========================================================
# Patterns trích xuất thông tin pháp lý từ text
# =========================================================

# Tham chiếu điều, khoản, điểm
ARTICLE_FULL_PATTERN = re.compile(
    r'(?:Căn\s+cứ\s+|Theo\s+quy\s+định\s+tại\s+|[Tt]heo\s+|[Tt]ại\s+)?'
    r'([Đđ]iều\s+\d+[a-zA-Z]?)'
    r'(?:,?\s*[Kk]hoản\s+(\d+))?'
    r'(?:,?\s*[Đđ]iểm\s+([a-zA-Z]))?',
    re.IGNORECASE
)

# Số hiệu văn bản pháp luật Việt Nam
# Lưu ý: KHÔNG dùng alternation kiểu (?:QH|NĐ|TT|...|[A-ZĐ0-9/-]+) vì regex sẽ khớp
# với alternative đầu tiên (ví dụ "QH") rồi dừng, làm mất phần hậu tố số hiệu (ví dụ "13"
# trong "QH13"). Chỉ cần một charset gộp vì nó đã bao trùm mọi tiền tố loại văn bản.
DOC_NUMBER_PATTERN = re.compile(
    r'(\d{1,3}/\d{4}/[A-ZĐ0-9/-]+)',
    re.IGNORECASE
)

# Loại văn bản
DOC_TYPE_KEYWORDS = {
    "luật": "Luật",
    "nghị định": "Nghị định",
    "thông tư": "Thông tư",
    "quyết định": "Quyết định",
    "nghị quyết": "Nghị quyết",
    "chỉ thị": "Chỉ thị",
    "thông tư liên tịch": "Thông tư liên tịch",
    "pháp lệnh": "Pháp lệnh",
    "hiến pháp": "Hiến pháp",
}


class PostProcessor:
    """
    Xử lý hậu kỳ câu trả lời: Trích xuất → Hard-Map → Đóng gói JSON.
    """

    def __init__(self, law_manifest: Optional[Dict] = None):
        """
        Args:
            law_manifest: Dict ánh xạ số hiệu → metadata chuẩn.
                         Cấu trúc mẫu:
                         {
                           "80/2021/NĐ-CP": {
                             "loai_van_ban": "Nghị định",
                             "so_hieu": "80/2021/NĐ-CP",
                             "trich_yeu": "Quy định chi tiết...",
                             "ten_day_du": "Nghị định 80/2021/NĐ-CP ngày..."
                           }
                         }
        """
        self.law_manifest = law_manifest or {}
        logger.info(f"✅ PostProcessor init với {len(self.law_manifest)} entries trong manifest.")

    @classmethod
    def from_manifest_file(cls, path: str = Settings.LAW_MANIFEST_PATH) -> "PostProcessor":
        try:
            with open(path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            return cls(law_manifest=manifest)
        except FileNotFoundError:
            logger.warning(f"Không tìm thấy law_manifest.json tại {path}")
            return cls(law_manifest={})

    # =========================================================
    # BƯỚC 1: REGEX EXTRACTION
    # =========================================================
    def extract_legal_references(
        self, answer: str
    ) -> Tuple[List[str], List[Tuple[Optional[str], str]]]:
        """
        Bóc tách toàn bộ tham chiếu pháp lý từ câu trả lời, đồng thời ghép mỗi
        "Điều X" với số hiệu văn bản gần nó nhất về khoảng cách ký tự (xét cả phía
        trước và phía sau, không chỉ phía trước). Lý do: prompt yêu cầu LLM trích dẫn
        theo dạng "Theo quy định tại Điều X Luật/Nghị định Y..." — tức số hiệu văn bản
        thường nằm SAU "Điều X" trong câu, nên chỉ xét "gần nhất phía trước" sẽ gán sai
        văn bản khi câu trả lời nhắc tới nhiều văn bản.

        Returns:
            Tuple(doc_numbers, doc_article_pairs)
            - doc_numbers: ["80/2021/NĐ-CP", ...] theo thứ tự xuất hiện
            - doc_article_pairs: [(doc_number_hoặc_None, "Điều 5, Khoản 2"), ...]
        """
        doc_events = [(m.start(), m.group(1)) for m in DOC_NUMBER_PATTERN.finditer(answer)]
        doc_numbers: List[str] = []
        for _, val in doc_events:
            if val not in doc_numbers:
                doc_numbers.append(val)

        article_events = []
        for m in ARTICLE_FULL_PATTERN.finditer(answer):
            dieu, khoan, diem = m.groups()
            if not dieu:
                continue
            ref = dieu.strip()
            if khoan:
                ref += f", Khoản {khoan}"
            if diem:
                ref += f", Điểm {diem.lower()}"
            article_events.append((m.start(), ref))

        doc_article_pairs: List[Tuple[Optional[str], str]] = []
        for pos, ref in article_events:
            nearest_doc = None
            if doc_events:
                nearest_doc = min(doc_events, key=lambda e: abs(e[0] - pos))[1]
            pair = (nearest_doc, ref)
            if pair not in doc_article_pairs:
                doc_article_pairs.append(pair)

        return doc_numbers, doc_article_pairs

    # =========================================================
    # BƯỚC 2: HARD-MAPPING → relevant_docs & relevant_articles
    # =========================================================
    def _build_canonical_doc_string(self, doc_number: str) -> str:
        """
        Tra law_manifest.json để lấy chuỗi chuẩn theo format BTC: "<Số hiệu>|<Tên văn bản>".
        manifest thực tế có cấu trúc {"doc_id", "document_type", "law_name", "btc_standard_string"},
        trong đó "btc_standard_string" đã đúng định dạng "<số hiệu>|<loại văn bản> <số hiệu> <tên>"
        nên chỉ cần dùng trực tiếp, không cần tự dựng lại chuỗi.
        """
        entry = self.law_manifest.get(doc_number)

        if entry is None:
            # Tìm fuzzy trong manifest
            for key, val in self.law_manifest.items():
                if doc_number in key or key in doc_number:
                    entry = val
                    break

        if isinstance(entry, dict) and entry.get("btc_standard_string"):
            return entry["btc_standard_string"]
        if isinstance(entry, str) and "|" in entry:
            return entry

        # Fallback: Không có trong manifest -> tự sinh chuỗi "<số hiệu>|<loại văn bản suy luận>"
        return self._infer_doc_type_from_number(doc_number)

    def _infer_doc_type_from_number(self, doc_number: str) -> str:
        """Suy luận loại văn bản từ ký hiệu số hiệu, trả về đúng format '<số hiệu>|<tên suy luận>'."""
        upper = doc_number.upper()
        if "QH" in upper:
            guess = f"Luật {doc_number}"
        elif "NĐ-CP" in upper or "ND-CP" in upper:
            guess = f"Nghị định {doc_number}"
        elif "TT-BTC" in upper or "TT-" in upper:
            guess = f"Thông tư {doc_number}"
        elif "QĐ" in upper:
            guess = f"Quyết định {doc_number}"
        else:
            guess = f"Văn bản {doc_number}"
        return f"{doc_number}|{guess}"

    def build_relevant_docs(self, doc_numbers: List[str]) -> List[str]:
        """Xây dựng trường relevant_docs theo chuẩn BTC: '<Số hiệu>|<Tên văn bản>'."""
        relevant_docs = []
        seen: Set[str] = set()

        for doc_num in doc_numbers:
            canonical = self._build_canonical_doc_string(doc_num)
            if canonical not in seen:
                relevant_docs.append(canonical)
                seen.add(canonical)

        return relevant_docs

    def build_relevant_articles(
        self,
        doc_article_pairs: List[Tuple[Optional[str], str]],
    ) -> List[str]:
        """
        Xây dựng trường relevant_articles theo chuẩn BTC.
        Format: "<Số hiệu>|<Tên văn bản>|<Điều X, Khoản Y>".
        Mỗi Điều được ghép với đúng số hiệu văn bản đứng gần nó nhất trong câu trả lời
        (xem extract_legal_references), tránh gán nhầm khi có nhiều văn bản được trích dẫn.
        """
        relevant_articles = []
        seen: Set[str] = set()

        for doc_number, art_ref in doc_article_pairs:
            # Bỏ qua "Điều X" trần không gắn được số hiệu văn bản: chuỗi dạng "Điều 12"
            # không khớp format BTC "<số hiệu>|<tên>|<Điều X>" nên chỉ là nhiễu, làm tụt precision.
            if not doc_number:
                continue
            canonical_doc = self._build_canonical_doc_string(doc_number)
            full_ref = f"{canonical_doc}|{art_ref}"

            if full_ref not in seen:
                relevant_articles.append(full_ref)
                seen.add(full_ref)

        return relevant_articles

    # =========================================================
    # PROCESS SINGLE ITEM
    # =========================================================
    def process_single(
        self,
        item_id: str,
        query: str,
        answer: str,
        context_docs: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Xử lý hậu kỳ một bản ghi đơn lẻ.

        Returns:
            Dict đúng chuẩn format BTC:
            {
                "id": str,
                "question": str,
                "answer": str,
                "relevant_docs": [...],
                "relevant_articles": [...]
            }
        """
        doc_numbers, doc_article_pairs = self.extract_legal_references(answer)

        # Bổ sung doc_numbers từ context metadata (nếu LLM quên trích dẫn số hiệu)
        if context_docs:
            for doc in context_docs:
                meta = doc.get("metadata", {})
                dn = meta.get("doc_number", "")
                if dn and dn not in doc_numbers:
                    # Chỉ thêm nếu số hiệu xuất hiện trong câu trả lời
                    if dn in answer or dn.split("/")[0] in answer:
                        doc_numbers.append(dn)

        relevant_docs = self.build_relevant_docs(doc_numbers)
        relevant_articles = self.build_relevant_articles(doc_article_pairs)

        return {
            "id": item_id,
            "question": query,
            "answer": answer,
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
        }

    # =========================================================
    # BƯỚC 3: DATA VALIDATION
    # =========================================================
    def validate_results(self, results: List[Dict]) -> Dict:
        """
        Kiểm tra tổng thể file kết quả trước khi nộp.

        Checks:
        - Đủ 2000 bản ghi
        - Không trùng lặp ID
        - Không có trường null
        - Cú pháp JSON hợp lệ
        - Tất cả answer có tham chiếu pháp lý
        """
        report = {
            "total_records": len(results),
            "errors": [],
            "warnings": [],
            "stats": {
                "no_relevant_docs": 0,
                "no_relevant_articles": 0,
                "no_answer": 0,
                "short_answers": 0,  # Câu trả lời < 50 ký tự
            }
        }

        seen_ids: Set[str] = set()

        for i, item in enumerate(results):
            item_id = item.get("id", f"item_{i}")

            # Kiểm tra trùng ID
            if item_id in seen_ids:
                report["errors"].append(f"Trùng lặp ID: {item_id}")
            seen_ids.add(str(item_id))

            # Kiểm tra null fields
            required_fields = ["id", "question", "answer", "relevant_docs", "relevant_articles"]
            for field in required_fields:
                if item.get(field) is None:
                    report["errors"].append(f"[{item_id}] Trường '{field}' là null")

            # Kiểm tra answer rỗng
            answer = item.get("answer", "")
            if not answer or len(answer) < 10:
                report["stats"]["no_answer"] += 1
                report["errors"].append(f"[{item_id}] Answer rỗng hoặc quá ngắn")
            elif len(answer) < 50:
                report["stats"]["short_answers"] += 1
                report["warnings"].append(f"[{item_id}] Answer ngắn ({len(answer)} chars)")

            # Kiểm tra relevant_docs
            if not item.get("relevant_docs"):
                report["stats"]["no_relevant_docs"] += 1
                report["warnings"].append(f"[{item_id}] relevant_docs rỗng")

            # Kiểm tra relevant_articles
            if not item.get("relevant_articles"):
                report["stats"]["no_relevant_articles"] += 1
                report["warnings"].append(f"[{item_id}] relevant_articles rỗng")

        # Kiểm tra số lượng
        if len(results) != 2000:
            report["errors"].append(
                f"Số bản ghi không đúng: có {len(results)}, cần 2000"
            )

        report["is_valid"] = len(report["errors"]) == 0
        return report

    # =========================================================
    # BƯỚC 4: SUBMISSION PACKAGING
    # =========================================================
    def package_submission(
        self,
        results: List[Dict],
        output_dir: str = Settings.OUTPUT_DIR
    ) -> Tuple[str, str]:
        """
        Xuất results.json và đóng gói thành submission.zip.

        Returns:
            Tuple(json_path, zip_path)
        """
        os.makedirs(output_dir, exist_ok=True)

        # Xuất results.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, "results.json")
        zip_path = os.path.join(output_dir, f"submission_{timestamp}.zip")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Đã xuất results.json: {json_path}")

        # Đóng gói zip (results.json ở thư mục gốc của zip)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname="results.json")
        logger.info(f"📦 Đã đóng gói: {zip_path}")

        return json_path, zip_path
