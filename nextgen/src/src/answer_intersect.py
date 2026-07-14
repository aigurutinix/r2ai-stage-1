"""
src/answer_intersect.py  (LUỒNG RIÊNG cho fast_retrieval --llm-answer)

Pipeline: LLM sinh answer THẬT -> lấy GIAO giữa các điều/số hiệu LLM trích dẫn trong answer
với TOP pool ứng viên rerank. Mục tiêu:
  - relevant_docs/articles bám đúng điều LLM thực sự dùng (grounded trong pool rerank).
  - DERIVE docs TỪ các điều được chọn => mọi văn bản đều có ít nhất 1 Điều (hết "doc mồ côi").
  - answer khác rỗng => còn ăn điểm QA (5 tiêu chí).

LƯU Ý: giao-citation từng cho F2 thấp hơn top-2 rerank trên GT-50 -> BẮT BUỘC đo lại
bằng score_retrieval.py trước khi nộp.
"""
import re
import logging
from typing import List, Dict, Set, Tuple

from config.settings import Settings

logger = logging.getLogger(__name__)

_ARTICLE_NUM_RE = re.compile(r"[Đđ]iều\s+(\d+)")
_DOC_NUM_RE = re.compile(r"\d{1,4}/\d{4}/[A-Za-zĐđ\-]+")


def parse_citations(answer_text: str) -> Tuple[Set[str], Set[str]]:
    """Bóc các số Điều và số hiệu văn bản LLM nhắc tới trong answer."""
    art_nums = set(_ARTICLE_NUM_RE.findall(answer_text or ""))
    doc_nums = set(_DOC_NUM_RE.findall(answer_text or ""))
    return art_nums, doc_nums


def _article_num(article_id: str) -> str:
    m = re.search(r"(\d+)", article_id or "")
    return m.group(1) if m else ""


def intersect_select(
    answer_text: str,
    ranked: List[Dict],
    pool_k: int = 8,
    max_out: int = 5,
) -> List[Dict]:
    """
    Trả về danh sách context (con) = GIAO giữa điều LLM trích dẫn và pool rerank[:pool_k],
    giữ thứ tự rerank, tối đa max_out phần tử.
    Fallback: nếu giao rỗng -> top-N rerank (Settings.RELEVANT_ARTICLES_MAX) để không mất recall.
    """
    pool = ranked[:pool_k]
    if not pool:
        return []

    cited_arts, cited_docs = parse_citations(answer_text)

    def is_cited(c: Dict) -> bool:
        md = c.get("metadata", {})
        anum = _article_num(md.get("article_id", ""))
        if not anum or anum not in cited_arts:
            return False
        # Nếu answer có nêu số hiệu, bắt khớp đúng văn bản để tránh nhầm 'Điều X' giữa các luật
        if cited_docs:
            return md.get("doc_number", "").strip() in cited_docs
        return True

    kept = [c for c in pool if is_cited(c)]

    if not kept:
        # LLM không trích được điều nào khớp pool -> giữ top theo rerank
        kept = pool[: max(1, Settings.RELEVANT_ARTICLES_MAX)]

    return kept[:max_out]
