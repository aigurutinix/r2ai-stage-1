"""Unit test cho tests/metrics.py — kiểm chứng công thức F2 & trích Điều."""
from __future__ import annotations

import math

from tests.metrics import extract_dieu_numbers, macro_average, prf2


def test_extract_dieu_basic():
    assert extract_dieu_numbers("Theo Điều 12 và Điều 24 của Luật...") == {"12", "24"}


def test_extract_dieu_case_and_spacing():
    assert extract_dieu_numbers("điều 5, Điều  17") == {"5", "17"}


def test_extract_dieu_ignores_non_dieu_numbers():
    # "khoản 3" và năm "2020" không phải Điều
    assert extract_dieu_numbers("khoản 3 Điều 7 năm 2020") == {"7"}


def test_extract_dieu_letter_suffix():
    # Điều bổ sung trong luật sửa đổi: "Điều 112a" phải bắt được, KHÁC "Điều 112"
    assert extract_dieu_numbers("theo Điều 112a và Điều 112 của Luật") == {"112a", "112"}


def test_extract_dieu_empty():
    assert extract_dieu_numbers("") == set()
    assert extract_dieu_numbers("không trích dẫn gì") == set()


def test_prf2_perfect():
    r = prf2({4, 17}, {4, 17})
    assert r.precision == 1.0 and r.recall == 1.0 and r.f2 == 1.0


def test_prf2_half():
    # truy hồi {4,5}, liên quan {4,17} → đúng {4}
    r = prf2({4, 5}, {4, 17})
    assert r.precision == 0.5 and r.recall == 0.5
    # F2 = 5*0.25 / (4*0.5+0.5) = 1.25/2.5 = 0.5
    assert math.isclose(r.f2, 0.5)


def test_prf2_recall_weighted():
    # P=1, R=0.5 → F2 nghiêng về recall, phải < 0.75 (F1) ... thực ra F2=0.555
    r = prf2({4}, {4, 17})
    assert r.precision == 1.0 and r.recall == 0.5
    assert math.isclose(r.f2, (5 * 1.0 * 0.5) / (4 * 1.0 + 0.5))  # 2.5/4.5


def test_prf2_high_recall_beats_high_precision_under_f2():
    # F2 ưu tiên recall: (P=0.5,R=1.0) phải > (P=1.0,R=0.5)
    hi_recall = prf2({4, 17, 99}, {4, 17})  # P=2/3, R=1.0
    hi_prec = prf2({4}, {4, 17})            # P=1.0, R=0.5
    assert hi_recall.f2 > hi_prec.f2


def test_prf2_empty_relevant():
    # không có đáp án liên quan & không truy hồi gì → coi như đúng
    r = prf2(set(), set())
    assert r.recall == 1.0 and r.precision == 1.0


def test_macro_average():
    a = prf2({4}, {4})          # 1,1,1
    b = prf2({5}, {4, 17})      # 0,0,0
    m = macro_average([a, b])
    assert m.n_queries == 2
    assert math.isclose(m.precision, 0.5)
    assert math.isclose(m.recall, 0.5)
    assert math.isclose(m.f2, 0.5)
