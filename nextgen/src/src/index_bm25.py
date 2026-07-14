"""
src/index_bm25.py
Xây dựng và lưu chỉ mục BM25 cho corpus pháp lý tiếng Việt.
Có thể chạy offline một lần và serialize để tái sử dụng.
"""

import os
import re
import pickle
import logging
from typing import List, Dict

from rank_bm25 import BM25Okapi
from underthesea import word_tokenize

from config.settings import Settings

logger = logging.getLogger(__name__)

# =========================================================
# Bộ Stopwords pháp lý tiếng Việt
# Loại bỏ các từ không mang giá trị phân biệt ngữ nghĩa
# =========================================================
LEGAL_STOPWORDS = {
    "căn_cứ", "quy_định", "tại", "điều", "khoản", "điểm", "của", "và",
    "theo", "về", "có", "được", "là", "các", "này", "đó", "cho", "với",
    "trong", "trên", "từ", "đến", "khi", "mà", "không", "phải", "hoặc",
    "thì", "nếu", "bởi", "vì", "do", "để", "như", "đây", "những", "một",
    "hai", "ba", "số", "ngày", "năm", "tháng", "quyết", "định", "luật",
    "nghị", "thông", "tư", "văn", "bản", "ban", "hành", "hướng", "dẫn",
}

# Từ đồng nghĩa chuyên ngành pháp lý (normalize trước khi tokenize)
SYNONYM_MAP = {
    "mặt bằng sản xuất": "địa điểm kinh doanh",
    "cơ sở sản xuất": "địa điểm kinh doanh",
    "doanh nghiệp nhỏ": "doanh nghiệp nhỏ và vừa",
    "doanh nghiệp vừa": "doanh nghiệp nhỏ và vừa",
    "sme": "doanh nghiệp nhỏ và vừa",
    "dnnvv": "doanh nghiệp nhỏ và vừa",
    "tnhh": "trách nhiệm hữu hạn",
    "cổ phần": "công ty cổ phần",
    "tncn": "thu nhập cá nhân",
    "tndn": "thu nhập doanh nghiệp",
    "bhxh": "bảo hiểm xã hội",
    "bhyt": "bảo hiểm y tế",
    "bhtn": "bảo hiểm thất nghiệp",
    "gtgt": "giá trị gia tăng",
    "vat": "giá trị gia tăng",
}


def normalize_synonyms(text: str) -> str:
    """Thay thế từ đồng nghĩa để chuẩn hóa truy vấn."""
    text_lower = text.lower()
    for src, tgt in SYNONYM_MAP.items():
        text_lower = text_lower.replace(src, tgt)
    return text_lower


def tokenize_legal_text(text: str) -> List[str]:
    """
    Tokenize văn bản pháp lý tiếng Việt:
    1. Chuẩn hóa từ đồng nghĩa
    2. Tách từ bằng underthesea
    3. Loại bỏ stopwords pháp lý
    4. Giữ lại số hiệu văn bản (ví dụ: 80/2021)
    """
    text = normalize_synonyms(text)

    # Giữ nguyên số hiệu trước khi word_tokenize
    doc_numbers = re.findall(r'\d+/\d+(?:/[A-ZĐ-]+)?', text)

    tokens = word_tokenize(text.lower(), format="text").split()
    tokens = [t for t in tokens if t not in LEGAL_STOPWORDS and len(t) > 1]

    # Thêm lại số hiệu vào danh sách token (tránh bị tách vụn)
    tokens.extend(doc_numbers)

    return tokens


class BM25IndexBuilder:
    """
    Xây dựng và quản lý chỉ mục BM25 cho corpus văn bản pháp lý.
    Hỗ trợ serialize/deserialize để tránh re-build mỗi lần chạy.
    """

    def __init__(self):
        self.bm25: BM25Okapi = None
        self.documents: List[Dict] = []

    def build(self, documents: List[Dict]) -> "BM25IndexBuilder":
        """
        Xây dựng chỉ mục BM25 từ danh sách văn bản.

        Args:
            documents: List[Dict] với cấu trúc:
                {"id": str, "text": str, "metadata": dict}
        """
        logger.info(f"Đang build BM25 index từ {len(documents)} văn bản...")
        self.documents = documents

        tokenized_corpus = []
        for i, doc in enumerate(documents):
            tokens = tokenize_legal_text(doc.get("text", ""))
            tokenized_corpus.append(tokens)
            if (i + 1) % 500 == 0:
                logger.info(f"  Đã tokenize {i + 1}/{len(documents)} văn bản")

        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("✅ BM25 index build thành công.")
        return self

    def save(self, path: str = Settings.BM25_INDEX_PATH) -> None:
        """Serialize BM25 index ra file để tái sử dụng."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "documents": self.documents}, f)
        logger.info(f"💾 Đã lưu BM25 index tại: {path}")

    def load(self, path: str = Settings.BM25_INDEX_PATH) -> "BM25IndexBuilder":
        """Deserialize BM25 index từ file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.documents = data["documents"]
        logger.info(f"✅ Đã load BM25 index từ: {path} ({len(self.documents)} docs)")
        return self

    def search(self, query: str, top_k: int = Settings.TOP_K_RAW) -> List[Dict]:
        """
        Tìm kiếm BM25 và trả về Top K văn bản có điểm cao nhất.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index chưa được build hoặc load!")

        tokenized_query = tokenize_legal_text(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            doc = dict(self.documents[idx])
            doc["bm25_score"] = float(scores[idx])
            results.append(doc)

        return results


# =========================================================
# Script chạy độc lập để build index lần đầu
# =========================================================
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    # Giả sử bạn có file corpus đã chuẩn hóa
    corpus_path = os.path.join(Settings.DATA_DIR, "law_corpus_clean.json")
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        builder = BM25IndexBuilder()
        builder.build(corpus).save()
        print(f"Done! Index đã lưu tại {Settings.BM25_INDEX_PATH}")
    else:
        print(f"❌ Không tìm thấy corpus tại {corpus_path}")
        print("Hãy chạy sau khi đã có file law_corpus_clean.json từ Giai đoạn 0.")
