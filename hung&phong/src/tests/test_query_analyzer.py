"""Unit tests cho backend.query_analyzer."""
from __future__ import annotations

import pytest

from backend.query_analyzer import analyze


# ---------- Số ký hiệu ----------


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Bộ luật Lao động 45/2019/QH14", "45/2019/QH14"),
        ("Sắc lệnh 159/SL năm 1950 quy định gì?", "159/SL"),
        ("Nghị định 127/2020/NĐ-CP có hiệu lực khi nào", "127/2020/NĐ-CP"),
        ("Thông tư 01/2021/TT-BCA", "01/2021/TT-BCA"),
        ("Quyết định 87/1999/QĐ-UB Lâm Đồng", "87/1999/QĐ-UB"),
        ("Chỉ thị 28/1999/CT-UB về niêm yết giá", "28/1999/CT-UB"),
    ],
)
def test_so_ky_hieu_detection(query: str, expected: str) -> None:
    f = analyze(query)
    assert f.so_ky_hieu == expected, f"Got {f.so_ky_hieu!r}"


def test_so_ky_hieu_not_confused_with_date() -> None:
    """Ngày tháng `28/12/2023` KHÔNG được nhận làm số ký hiệu."""
    f = analyze("Văn bản ban hành ngày 28/12/2023 có hiệu lực không?")
    assert f.so_ky_hieu is None


def test_so_ky_hieu_none_when_no_match() -> None:
    f = analyze("Thời gian thử việc tối đa là bao lâu?")
    assert f.so_ky_hieu is None


# ---------- Điều số ----------


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Điều 36 Bộ luật Lao động", 36),
        ("điều 1 sắc lệnh", 1),
        ("Theo Điều 100, Khoản 2", 100),
        ("Quy định tại Điều  7 của luật", 7),
    ],
)
def test_dieu_detection(query: str, expected: int) -> None:
    assert analyze(query).dieu_so == expected


def test_dieu_none_when_absent() -> None:
    assert analyze("Luật doanh nghiệp 59/2020/QH14 nói gì?").dieu_so is None


# ---------- Loại văn bản ----------


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Bộ luật Lao động nói gì về thử việc", "Bộ luật"),
        ("luật doanh nghiệp 2020", "Luật"),
        ("Sắc lệnh năm 1950 về toà án", "Sắc lệnh"),
        ("Nghị định 127 điều chỉnh thuế", "Nghị định"),
        ("Quyết định UBND Lâm Đồng", "Quyết định"),
        ("Hiến pháp 2013", "Hiến pháp"),
        ("Thông tư 01 BCA", "Thông tư"),
    ],
)
def test_loai_van_ban_detection(query: str, expected: str) -> None:
    assert analyze(query).loai_van_ban == expected


def test_loai_van_ban_bo_luat_beats_luat() -> None:
    """`Bộ luật` ưu tiên hơn `Luật` (substring trap)."""
    assert analyze("Bộ luật Hình sự").loai_van_ban == "Bộ luật"


# ---------- Composite & filter dict ----------


def test_filter_dict_with_so_ky_hieu_drops_loai_van_ban() -> None:
    """Khi có số ký hiệu, KHÔNG cần thêm loai_van_ban (over-filter)."""
    f = analyze("Nghị định 127/2020/NĐ-CP")
    d = f.to_qdrant_filter()
    assert d == {"so_ky_hieu": "127/2020/NĐ-CP"}


def test_filter_dict_combines_so_ky_hieu_and_dieu() -> None:
    f = analyze("Điều 36 Bộ luật Lao động 45/2019/QH14")
    d = f.to_qdrant_filter()
    assert d == {"so_ky_hieu": "45/2019/QH14", "dieu_so": 36}


def test_filter_dict_fallback_to_loai_van_ban_only() -> None:
    f = analyze("Có Nghị định nào về thuế bất động sản không?")
    d = f.to_qdrant_filter()
    assert d == {"loai_van_ban": "Nghị định"}


def test_empty_query_safe() -> None:
    f = analyze("")
    assert not f.has_any()
    assert f.to_qdrant_filter() == {}


def test_has_any_true_when_any_signal() -> None:
    assert analyze("Điều 1").has_any()
    assert analyze("Bộ luật").has_any()
    assert analyze("45/2019/QH14").has_any()
    assert not analyze("xin chào").has_any()
