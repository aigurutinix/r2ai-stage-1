"""Chuẩn hoá văn bản tiếng Việt cho RAG — áp ĐỐI XỨNG cho corpus (chunk) và query.

Vấn đề: nguồn vbpl.vn dùng kiểu đặt DẤU THANH CŨ trên nguyên âm đầu của oa/oe/uy
("khỏan, họat, tóan, hòan, ủy") trong khi câu hỏi BTC viết kiểu MỚI ("khoản, hoạt,
toán, hoàn, uỷ"). Lệch chuẩn này làm BM25 (khớp lexical) trượt ~48% corpus ở những từ
pháp lý phổ biến nhất ("khoản"). Chuẩn hoá về MỘT dạng (đặt dấu trên nguyên âm sau) rồi
áp cho cả hai phía → khớp chắc. Không đụng tới số ký hiệu/Điều nên citation an toàn.
"""
from __future__ import annotations

import re
import unicodedata

# Cặp dấu thanh CŨ (dấu trên o/u của oa/oe/uy) -> MỚI (dấu trên a/e/y).
_PAIRS = {
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
}
# Thêm biến thể viết HOA (HỌAT ĐỘNG, ỦY BAN...) và Hoa-đầu.
_MAP: dict[str, str] = {}
for _k, _v in _PAIRS.items():
    _MAP[_k] = _v
    _MAP[_k.upper()] = _v.upper()
    _MAP[_k.capitalize()] = _v.capitalize()

_RE_TONE = re.compile("|".join(sorted(_MAP, key=len, reverse=True)))
_RE_WS = re.compile(r"[ \t ]{2,}")


def normalize_vn(text: str) -> str:
    """NFC + chuyển dấu thanh oa/oe/uy về kiểu mới + gộp khoảng trắng thừa."""
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = _RE_TONE.sub(lambda m: _MAP[m.group()], text)
    text = _RE_WS.sub(" ", text)
    return text
