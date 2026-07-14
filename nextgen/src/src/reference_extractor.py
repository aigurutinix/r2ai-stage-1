"""
src/reference_extractor.py
Trích relevant_docs / relevant_articles từ danh sách context ĐÃ RERANK.

Tách riêng khỏi AnswerGenerator để pipeline retrieval-only (fast_retrieval.py — KHÔNG nạp LLM)
tái dùng được CÙNG một logic, tránh lệch code giữa hai đường.

Chiến lược: lấy TOP-N theo thứ tự rerank (Settings.RELEVANT_ARTICLES_MAX / RELEVANT_DOCS_MAX).
Bằng chứng (mô phỏng 50 câu GT): top-2 rerank cho F2 cao nhất; giao với citation LLM làm giảm F2.
"""
import json
import logging
from typing import List, Dict, Tuple

from config.settings import Settings

logger = logging.getLogger(__name__)


def load_manifest(path: str = Settings.LAW_MANIFEST_PATH) -> Dict:
    """Nạp law_manifest.json (dict keyed by số hiệu văn bản)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Không tìm thấy law_manifest.json tại {path}. Chạy không có manifest.")
        return {}


def canonical_doc_string(doc_number: str, fallback_title: str, manifest: Dict) -> str:
    """
    Tra law_manifest.json để lấy chuỗi chuẩn "<Số hiệu>|<Tên văn bản>" theo format BTC.
    Ưu tiên field "btc_standard_string"; fallback sang title của chunk khi không có trong manifest.
    """
    entry = manifest.get(doc_number)
    if isinstance(entry, dict) and entry.get("btc_standard_string"):
        return entry["btc_standard_string"]
    if fallback_title:
        return f"{doc_number}|{fallback_title}"
    return f"{doc_number}|Văn bản {doc_number}"


def extract_references_topn(contexts: List[Dict], manifest: Dict) -> Tuple[List[str], List[str]]:
    """
    Lấy TOP-N văn bản/Điều đầu tiên theo thứ tự rerank.
    Yêu cầu: `contexts` giữ nguyên thứ tự rerank (điểm cao -> thấp).
    Định dạng:
    - relevant_docs: ["mã văn bản|tên văn bản"]
    - relevant_articles: ["mã văn bản|tên văn bản|Điều X"]
    """
    relevant_docs: List[str] = []
    relevant_articles: List[str] = []
    seen_docs = set()
    seen_articles = set()

    for doc in contexts:
        metadata = doc.get("metadata", {})
        doc_number = metadata.get("doc_number", "").strip()
        doc_title = metadata.get("title", "").strip()
        article_id = metadata.get("article_id", "").strip()  # Ví dụ: "Điều 4"

        if not doc_number:
            continue

        cdoc = canonical_doc_string(doc_number, doc_title, manifest)

        if cdoc not in seen_docs and len(relevant_docs) < Settings.RELEVANT_DOCS_MAX:
            seen_docs.add(cdoc)
            relevant_docs.append(cdoc)

        if article_id:
            astr = f"{cdoc}|{article_id}"
            if astr not in seen_articles and len(relevant_articles) < Settings.RELEVANT_ARTICLES_MAX:
                seen_articles.add(astr)
                relevant_articles.append(astr)

        if (len(relevant_docs) >= Settings.RELEVANT_DOCS_MAX
                and len(relevant_articles) >= Settings.RELEVANT_ARTICLES_MAX):
            break

    return relevant_docs, relevant_articles


def extract_references_all(contexts: List[Dict], manifest: Dict) -> Tuple[List[str], List[str]]:
    """
    Lấy TẤT CẢ văn bản/Điều phân biệt từ `contexts` (KHÔNG cap theo Settings).
    Dùng cho luồng LLM-select số lượng biến thiên: `contexts` ở đây đã là tập LLM chọn ra
    (đã giới hạn bởi max_select), nên chỉ cần chuyển thành chuỗi chuẩn + loại trùng.
    Giữ nguyên thứ tự đầu vào (= thứ tự ưu tiên LLM chọn).
    """
    relevant_docs: List[str] = []
    relevant_articles: List[str] = []
    seen_docs = set()
    seen_articles = set()

    for doc in contexts:
        metadata = doc.get("metadata", {})
        doc_number = metadata.get("doc_number", "").strip()
        doc_title = metadata.get("title", "").strip()
        article_id = metadata.get("article_id", "").strip()
        if not doc_number:
            continue
        cdoc = canonical_doc_string(doc_number, doc_title, manifest)
        if cdoc not in seen_docs:
            seen_docs.add(cdoc)
            relevant_docs.append(cdoc)
        if article_id:
            astr = f"{cdoc}|{article_id}"
            if astr not in seen_articles:
                seen_articles.add(astr)
                relevant_articles.append(astr)

    return relevant_docs, relevant_articles
