from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_v66_arch_pipeline import (  # noqa: E402
    OBSOLETE_PREFIXES,
    acceptable_reason,
    llm_verify_selection,
    prefix,
    select_candidates,
)


def _cand(key: str, art: str, text: str, raw: float, arch: float) -> dict:
    return {
        "_key": key,
        "art": art,
        "title": art.split("|")[1].strip() if "|" in art else art,
        "text": text,
        "prefix": prefix(art),
        "score_raw": raw,
        "score_arch": arch,
    }


def test_selector_rejects_obsolete_penalty_docs_even_with_high_raw() -> None:
    question = "Diem ban le thuoc la trung bay qua mot hop cua mot nhan hieu thuoc la thi bi xu phat bao nhieu tien?"
    pool = [
        _cand(
            "old",
            "06/2008/ND-CP|Nghi dinh Quy dinh ve xu phat vi pham hanh chinh trong hoat dong thuong mai|Dieu 31",
            "Dieu 31. Vi pham quy dinh ve kinh doanh thuoc la.",
            raw=0.98,
            arch=1.10,
        ),
        _cand(
            "new",
            "98/2020/ND-CP|Nghi dinh Quy dinh xu phat vi pham hanh chinh trong hoat dong thuong mai, san xuat, buon ban hang gia, hang cam va bao ve quyen loi nguoi tieu dung|Dieu 34",
            "Dieu 34. Hanh vi vi pham ve ban san pham thuoc la.",
            raw=0.70,
            arch=0.82,
        ),
    ]

    out = select_candidates(question, pool)

    assert all(not c["art"].startswith("06/2008") for c in out)
    assert any(c["art"].startswith("98/2020") for c in out)


def test_selector_prefers_advertising_law_for_general_advertising_ban() -> None:
    question = "Viec quang cao san pham cua minh la tot nhat tren thi truong co bi coi la hanh vi bi cam trong quang cao khong?"
    pool = [
        _cand(
            "tm",
            "36/2005/QH11|Luat Thuong mai|Dieu 109",
            "Dieu 109. Cac quang cao thuong mai bi cam.",
            raw=0.95,
            arch=1.05,
        ),
        _cand(
            "qc",
            "16/2012/QH13|Luat Quang cao|Dieu 8",
            "Dieu 8. Hanh vi cam trong hoat dong quang cao.",
            raw=0.65,
            arch=0.72,
        ),
    ]

    out = select_candidates(question, pool)

    assert all(not c["art"].startswith("36/2005") for c in out)
    assert any(c["art"].startswith("16/2012") for c in out)


def test_still_current_old_doc_is_not_rejected_as_obsolete() -> None:
    question = (
        "Cong ty toi nhan quyen thuong mai nhung co su co bat kha khang, "
        "can xu ly thong bao the nao de duoc mien trach nhiem?"
    )
    cand = _cand(
        "franchise",
        "35/2006/ND-CP|Nghi dinh Quy dinh chi tiet Luat Thuong mai ve hoat dong nhuong quyen thuong mai|Dieu 16",
        "Dieu 16. Don phuong cham dut hop dong nhuong quyen thuong mai.",
        raw=0.9437,
        arch=0.5587,
    )
    OBSOLETE_PREFIXES.add("35/2006")
    try:
        assert acceptable_reason(question, cand) == "ok"
    finally:
        OBSOLETE_PREFIXES.discard("35/2006")


def test_llm_drop_cannot_remove_strong_clause_coverage() -> None:
    question = (
        "Cong ty dang nop don dang ky kieu dang cong nghiep va chi dan dia ly, "
        "quy trinh tham dinh noi dung khac nhau the nao va sau khi duoc cap van bang "
        "thi nhung thong tin nao se duoc cong bo tren Cong bao?"
    )
    selected = [
        _cand(
            "cd",
            "23/2023/TT-BKHCN|Thong tu SHTT|Dieu 30",
            "Dieu 30. Tham dinh noi dung don dang ky chi dan dia ly.",
            raw=0.989,
            arch=1.026,
        ),
        _cand(
            "pub",
            "23/2023/TT-BKHCN|Thong tu SHTT|Dieu 33",
            "Dieu 33. Cong bo quyet dinh cap van bang bao ho. Cac thong tin duoc cong bo tren Cong bao so huu cong nghiep.",
            raw=0.978,
            arch=1.015,
        ),
        _cand(
            "kd",
            "23/2023/TT-BKHCN|Thong tu SHTT|Dieu 23",
            "Dieu 23. Tham dinh noi dung don dang ky kieu dang cong nghiep.",
            raw=0.970,
            arch=1.007,
        ),
    ]

    class FakeLLM:
        def complete(self, *_args, **_kwargs) -> str:
            return '{"keep":["C1","C3"],"add":[],"drop":["C2"],"missing":[],"reason":"drop"}'

    out, _meta = llm_verify_selection(question, selected, [], "fake", llm=FakeLLM(), apply=True)

    assert any(c["art"].endswith("Dieu 33") for c in out)


def test_llm_add_rejects_near_zero_rerank_noise() -> None:
    question = "Cong ty thue don vi quang cao thi quyen va nghia vu quang cao duoc xu ly the nao?"
    selected = [
        _cand(
            "direct",
            "16/2012/QH13|Luat Quang cao|Dieu 13",
            "Dieu 13. Quyen va nghia vu cua nguoi kinh doanh dich vu quang cao.",
            raw=0.52,
            arch=0.61,
        )
    ]
    pool = [
        _cand(
            "noise",
            "342/2025/ND-CP|Nghi dinh Quang cao|Dieu 24",
            "Dieu 24. Trach nhiem cua Bo Van hoa, The thao va Du lich ve quan ly nha nuoc doi voi hoat dong quang cao.",
            raw=0.0,
            arch=0.04,
        )
    ]

    class FakeLLM:
        def complete(self, *_args, **_kwargs) -> str:
            return '{"keep":["C1"],"add":["C2"],"drop":[],"missing":[],"reason":"add noise"}'

    out, _meta = llm_verify_selection(question, selected, pool, "fake", llm=FakeLLM(), apply=True)

    assert all(not c["art"].startswith("342/2025") for c in out)


def test_llm_drop_cannot_remove_selected_direct_low_raw_article() -> None:
    question = (
        "Cong ty chuyen giao du lieu khach hang cho ben thu ba de quang cao san pham "
        "thi phai quan ly du lieu ca nhan trong kinh doanh dich vu quang cao the nao?"
    )
    selected = [
        _cand(
            "transfer",
            "91/2025/QH15|Luat Bao ve du lieu ca nhan|Dieu 17",
            "Dieu 17. Chuyen giao du lieu ca nhan khi co su dong y cua chu the du lieu.",
            raw=0.080,
            arch=0.121,
        ),
        _cand(
            "ads",
            "91/2025/QH15|Luat Bao ve du lieu ca nhan|Dieu 28",
            "Dieu 28. Bao ve du lieu ca nhan trong kinh doanh dich vu quang cao. To chuc kinh doanh dich vu quang cao chi duoc su dung du lieu ca nhan cua khach hang theo quy dinh.",
            raw=0.082,
            arch=0.107,
        ),
    ]

    class FakeLLM:
        def complete(self, *_args, **_kwargs) -> str:
            return '{"keep":["C1"],"add":[],"drop":["C2"],"missing":[],"reason":"drop"}'

    out, _meta = llm_verify_selection(question, selected, [], "fake", llm=FakeLLM(), apply=True)

    assert any(c["art"].endswith("Dieu 28") for c in out)


def test_llm_drop_cannot_remove_internal_document_charter_article() -> None:
    question = (
        "Cong ty thay doi so luong va phan chia quyen han cua nguoi dai dien theo phap luat "
        "thi phai dieu chinh trong van ban noi bo nao?"
    )
    selected = [
        _cand(
            "notice",
            "59/2020/QH14|Luat Doanh nghiep|Dieu 31",
            "Dieu 31. Thong bao thay doi noi dung dang ky doanh nghiep.",
            raw=0.751,
            arch=0.966,
        ),
        _cand(
            "charter",
            "59/2020/QH14|Luat Doanh nghiep|Dieu 24",
            "Dieu 24. Dieu le cong ty bao gom Dieu le khi dang ky doanh nghiep va Dieu le duoc sua doi, bo sung trong qua trinh hoat dong.",
            raw=0.729,
            arch=0.928,
        ),
    ]

    class FakeLLM:
        def complete(self, *_args, **_kwargs) -> str:
            return '{"keep":["C1"],"add":[],"drop":["C2"],"missing":[],"reason":"drop"}'

    out, _meta = llm_verify_selection(question, selected, [], "fake", llm=FakeLLM(), apply=True)

    assert any(c["art"].endswith("Dieu 24") for c in out)
