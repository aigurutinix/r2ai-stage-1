"""Bộ lọc phạm vi corpus về đúng domain cuộc thi: Luật Doanh nghiệp & SME.

Cuộc thi chỉ chấm trên mảng Luật Doanh nghiệp & doanh nghiệp nhỏ và vừa (SME).
Ingest toàn bộ ~153k văn bản luật VN vừa chậm (≈8h embed) vừa thêm nhiễu các
lĩnh vực không liên quan → kéo tụt precision. Bộ lọc này giữ lại đúng tập cần:

  - `loai_van_ban` thuộc nhóm văn bản quy phạm "lõi" (Luật, Nghị định, ...).
  - title / lĩnh vực / ngành khớp ít nhất một từ khoá DN-SME.

Giữ TẤT CẢ tình trạng hiệu lực (kể cả hết hiệu lực) để tối đa recall — F2 của
cuộc thi ưu tiên recall; tình trạng hiệu lực vẫn nằm trong payload để lớp trả
lời cân nhắc.
"""
from __future__ import annotations

import json
from pathlib import Path

# Whitelist số hiệu VB thuộc phạm trù (crawl từ luatvietnam/thuvienphapluat — danh sách
# hướng dẫn các luật trụ cột). Tầng 1 của in_scope_vbpl: VB trong list LUÔN được giữ,
# kể cả title không khớp từ khoá (vd 135/2020 "tuổi nghỉ hưu" — chủ đề lao động).
_WL_PATH = Path(__file__).resolve().parents[1] / "data" / "scope_whitelist.json"
try:
    _WHITELIST: frozenset[str] = frozenset(
        json.loads(_WL_PATH.read_text(encoding="utf-8")).get("so_ky_hieu", []))
except Exception:
    _WHITELIST = frozenset()

# Loại văn bản "lõi" — văn bản quy phạm pháp luật có giá trị pháp lý cao.
CORE_TYPES: frozenset[str] = frozenset(
    {
        "Luật",
        "Bộ luật",
        "Nghị định",
        "Pháp lệnh",
        "Thông tư",
        "Thông tư liên tịch",
        "Văn bản hợp nhất",
        "Nghị quyết",
    }
)

# Từ khoá lĩnh vực DN/SME — khớp trên title + linh_vuc + nganh (đã lower()).
SME_KEYWORDS: tuple[str, ...] = (
    "doanh nghiệp",
    "đầu tư",
    "thuế",
    "lao động",
    "hợp đồng",
    "thương mại",
    "kinh doanh",
    "kế toán",
    "tài chính",
    "phá sản",
    "chứng khoán",
    "sở hữu trí tuệ",
    "cạnh tranh",
    "bảo hiểm xã hội",
    "tiền lương",
)

_BLOB_FIELDS = ("title", "linh_vuc", "nganh")

# ===== Dành cho dataset vbpl-vn (tmquan) =====
# Loại VB lõi (theo legal_type của vbpl-vn).
VBPL_CORE_TYPES: frozenset[str] = frozenset(
    {
        "Luật", "Bộ luật", "Pháp lệnh", "Nghị định", "Thông tư",
        "Thông tư liên tịch", "Văn bản hợp nhất", "Nghị quyết",
        # BỎ "Quyết định" — không phải VBPL quy phạm; lọt 5960 QĐ hành chính (gồm QĐ-BGDĐT
        # về đề cương môn học CNXH/giáo trình → 769 doc rác). (audit 2026-06-20)
    }
)
# Từ khoá DN/SME mở rộng — phủ đủ các domain đề thi (gồm đấu thầu, hải quan, hoá đơn...).
SME_KEYWORDS_VBPL: tuple[str, ...] = SME_KEYWORDS + (
    "đấu thầu", "hải quan", "hóa đơn", "ngân sách", "việc làm", "hộ kinh doanh",
    "hợp tác xã", "đăng ký kinh doanh", "công đoàn", "an toàn", "đối tác công tư",
    "quản lý thuế", "khởi nghiệp", "giá cả", "định giá", "khung giá", "đất đai", "xây dựng",
    # Bổ sung sau khi phát hiện scope lọc nhầm VB phạm trù (vd 99/2013 SHCN, 198/2025 KTTN):
    "sở hữu công nghiệp", "kinh tế tư nhân", "công nghệ", "chuyển đổi số", "thương nhân",
    "xuất khẩu", "nhập khẩu", "tín dụng", "viên chức", "công chức", "phí", "lệ phí",
    "ưu đãi", "vốn", "tài sản", "nhà ở", "môi trường",
    # Chủ đề lao động/an sinh dễ rớt từ khoá (vd 135/2020 "tuổi nghỉ hưu"):
    "nghỉ hưu", "hưu trí", "người lao động", "tai nạn lao động", "công đoàn",
    "an toàn vệ sinh lao động", "trợ cấp", "bảo hiểm thất nghiệp", "bảo hiểm y tế",
)

# Loại văn bản GỐC cấp cao — giữ HẾT (không cần từ khoá). Số lượng ít, là nền tảng pháp lý;
# F2 ưu tiên recall nên thà giữ rộng còn hơn lọc nhầm luật trong phạm trù (vd Luật Đầu tư,
# Luật SHTT... dễ rớt từ khoá).
_KEEP_ALL_TYPES: frozenset[str] = frozenset({"Luật", "Bộ luật", "Pháp lệnh"})


# Loại thẳng tài liệu GIÁO DỤC/ĐÀO TẠO phi-pháp-luật (đề cương môn học, giáo trình CNXH...)
# lọt qua keyword. audit 2026-06-20.
_EXCLUDE_TITLE = (
    "đề cương", "môn học", "giáo trình", "chủ nghĩa xã hội khoa học", "chủ nghĩa mác",
    "tư tưởng hồ chí minh", "lịch sử đảng", "triết học mác", "kinh tế chính trị mác",
    "chương trình đào tạo", "chương trình môn", "bài giảng",
)


def in_scope_vbpl(row: dict, keyword_filter: bool = True) -> bool:
    _title = str(row.get("title") or "").lower()
    if any(x in _title for x in _EXCLUDE_TITLE):
        return False
    """Lọc record vbpl-vn về phạm vi DN/SME cấp trung ương.

    - scope == 'trung_uong' (bỏ văn bản địa phương → bớt nhiễu).
    - legal_type thuộc nhóm lõi.
    - Văn bản GỐC cấp cao (Luật/Bộ luật/Pháp lệnh) → giữ hết (tránh lọc nhầm).
    - Còn lại (NĐ/TT/NQ/QĐ): title/legal_area khớp từ khoá DN/SME mở rộng.
    """
    if str(row.get("scope") or "").strip() != "trung_uong":
        return False
    lt = str(row.get("legal_type") or "").strip()
    if lt not in VBPL_CORE_TYPES:
        return False
    # Tầng 1: whitelist (số hiệu thuộc phạm trù — giữ kể cả title không khớp từ khoá)
    dn = row.get("doc_number")
    dn = dn[0] if isinstance(dn, (list, tuple)) and dn else dn
    if str(dn or "").strip() in _WHITELIST:
        return True
    # Tầng 2: văn bản gốc cấp cao (Luật/Bộ luật/Pháp lệnh)
    if not keyword_filter or lt in _KEEP_ALL_TYPES:
        return True
    # Tầng 3: khớp từ khoá DN/SME mở rộng
    blob = " ".join(str(row.get(k) or "") for k in ("title", "legal_area")).lower()
    return any(kw in blob for kw in SME_KEYWORDS_VBPL)


def _blob(row: dict) -> str:
    return " ".join(str(row.get(k) or "") for k in _BLOB_FIELDS).lower()


def in_scope(row: dict) -> bool:
    """True nếu doc thuộc phạm vi DN/SME (core type + khớp từ khoá)."""
    if (row.get("loai_van_ban") or "").strip() not in CORE_TYPES:
        return False
    blob = _blob(row)
    return any(kw in blob for kw in SME_KEYWORDS)
