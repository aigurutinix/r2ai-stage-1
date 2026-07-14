"""Chỉ số đánh giá truy hồi theo thể lệ cuộc thi.

Thể lệ (project.md §4.1): scorer tự động tìm pattern `Điều X` trong trường
`answer`, chuẩn hoá đáp án về `Điều X`, rồi tính **Precision / Recall / F2 macro**
(tính cho từng truy vấn rồi lấy trung bình).

    Precision = (số điều đúng) / (số điều đã truy hồi)
    Recall    = (số điều đúng) / (số điều liên quan)
    F2        = (5 · P · R) / (4 · P + R)      # ưu tiên Recall

Module thuần (không I/O) để dễ unit-test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# "Điều 12", "Điều 112a", "Điều 4đ" → "12" / "112a" / "4đ".
# Bắt cả hậu tố chữ (điều bổ sung trong luật sửa đổi: "Điều 112a") — nếu không
# regex `\d+\b` sẽ TRƯỢT hoàn toàn các điều dạng này.
_RE_DIEU = re.compile(r"(?i)\bĐiều\s+(\d{1,4})([a-zđ]?)(?![\wà-ỹ])")


def extract_dieu_numbers(text: str) -> set[str]:
    """Rút tập định danh điều luật ('Điều X') từ văn bản — giống scorer BTC.

    Trả về set CHUỖI đã chuẩn hoá (vd "12", "112a") để phân biệt 'Điều 112'
    với 'Điều 112a'.
    """
    if not text:
        return set()
    return {m.group(1) + (m.group(2) or "").lower() for m in _RE_DIEU.finditer(text)}


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f2: float
    n_correct: int
    n_retrieved: int
    n_relevant: int


def prf2(retrieved: set, relevant: set) -> PRF:
    """Precision/Recall/F2 cho MỘT truy vấn trên hai tập bất kỳ (hashable)."""
    n_ret = len(retrieved)
    n_rel = len(relevant)
    correct = retrieved & relevant
    n_cor = len(correct)

    # Quy ước biên: không có đáp án liên quan → recall=1 nếu cũng không truy hồi gì.
    precision = n_cor / n_ret if n_ret else (1.0 if n_rel == 0 else 0.0)
    recall = n_cor / n_rel if n_rel else 1.0

    denom = 4 * precision + recall
    f2 = (5 * precision * recall) / denom if denom else 0.0
    return PRF(precision, recall, f2, n_cor, n_ret, n_rel)


@dataclass(frozen=True)
class MacroScore:
    precision: float
    recall: float
    f2: float
    n_queries: int


def macro_average(per_query: list[PRF]) -> MacroScore:
    """Trung bình macro: tính chỉ số mỗi truy vấn rồi lấy trung bình cộng."""
    n = len(per_query)
    if n == 0:
        return MacroScore(0.0, 0.0, 0.0, 0)
    return MacroScore(
        precision=sum(p.precision for p in per_query) / n,
        recall=sum(p.recall for p in per_query) / n,
        f2=sum(p.f2 for p in per_query) / n,
        n_queries=n,
    )
