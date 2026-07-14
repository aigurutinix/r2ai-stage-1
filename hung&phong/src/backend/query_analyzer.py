"""Phân tích câu hỏi → trích keyword/metadata để filter trước khi vector search.

Pure-vector RAG hay miss khi query có số ký hiệu cụ thể (vd "159/SL").
Module này detect các tín hiệu cứng — số ký hiệu, số Điều, loại văn bản —
để build Qdrant filter, tăng độ chính xác đáng kể.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Số ký hiệu VBPL phổ biến:
#   45/2019/QH14   - Luật / Nghị quyết Quốc hội
#   127/2020/NĐ-CP - Nghị định
#   01/2021/TT-BCA - Thông tư
#   87/1999/QĐ-UB  - Quyết định UBND
#   28/1999/CT-UB  - Chỉ thị
#   159/SL         - Sắc lệnh (không có năm)
#   134/SL (1950)  - Sắc lệnh thời kỳ đầu
# Pattern bắt cả 2 dạng có/không có năm; ký tự VN "Đ" được chấp nhận.
_RE_SO_KY_HIEU = re.compile(
    r"\b(\d{1,4}(?:/\d{4})?/[A-ZĐ][A-ZĐ0-9]+(?:-[A-ZĐ0-9]+)?)\b"
)

# "Điều 36" / "điều 1" / "Điều  12,"
_RE_DIEU = re.compile(r"[Đđ]i[eề]u\s+(\d{1,4})\b")

# Loại văn bản — order matters (cụ thể trước, chung sau):
# "Bộ luật" trước "Luật", v.v. Match case-insensitive trên Unicode.
_LOAI_VAN_BAN_PATTERNS: list[tuple[str, re.Pattern]] = [
    (label, re.compile(rf"(?<![\wÀ-ỹ]){re.escape(label)}(?![\wÀ-ỹ])", re.IGNORECASE))
    for label in [
        "Hiến pháp",
        "Bộ luật",
        "Pháp lệnh",
        "Nghị quyết",
        "Nghị định",
        "Thông tư",
        "Quyết định",
        "Chỉ thị",
        "Sắc lệnh",
        "Công văn",
        "Luật",  # phải sau "Bộ luật" để không bị nuốt
        "Lệnh",  # phải sau "Sắc lệnh", "Pháp lệnh"
    ]
]


@dataclass(frozen=True)
class QueryFilters:
    """Tín hiệu cứng trích từ câu hỏi để build Qdrant filter."""

    so_ky_hieu: str | None = None
    dieu_so: int | None = None
    loai_van_ban: str | None = None
    all_so_ky_hieu: tuple[str, ...] = field(default_factory=tuple)

    def has_any(self) -> bool:
        return any([self.so_ky_hieu, self.dieu_so, self.loai_van_ban])

    def to_qdrant_filter(self) -> dict[str, str | int]:
        """Build dict cho `QdrantStore.search(filter_must=...)`.

        KHÔNG bao gồm loai_van_ban nếu đã có so_ky_hieu (số ký hiệu đã đủ
        unique). Tránh over-filter.
        """
        f: dict[str, str | int] = {}
        if self.so_ky_hieu:
            f["so_ky_hieu"] = self.so_ky_hieu
        else:
            if self.loai_van_ban:
                f["loai_van_ban"] = self.loai_van_ban
        if self.dieu_so is not None:
            f["dieu_so"] = self.dieu_so
        return f


def analyze(query: str) -> QueryFilters:
    """Trích filters từ câu hỏi tiếng Việt."""
    if not query:
        return QueryFilters()

    skh_matches = _RE_SO_KY_HIEU.findall(query)
    skh_primary = skh_matches[0] if skh_matches else None

    dieu_match = _RE_DIEU.search(query)
    dieu_so = int(dieu_match.group(1)) if dieu_match else None

    loai = None
    for label, pat in _LOAI_VAN_BAN_PATTERNS:
        if pat.search(query):
            loai = label
            break

    return QueryFilters(
        so_ky_hieu=skh_primary,
        dieu_so=dieu_so,
        loai_van_ban=loai,
        all_so_ky_hieu=tuple(skh_matches),
    )
