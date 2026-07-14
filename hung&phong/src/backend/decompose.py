"""Phân rã câu hỏi tình huống dài thành các câu hỏi pháp lý NGUYÊN TỬ.

Một câu thi thường gói 2–4 vấn đề pháp lý (vd: ký HĐLĐ sai + chậm trả lương +
trách nhiệm dân sự). Retrieval 1-shot chỉ khớp 1 vế → sót căn cứ các vế khác.
Tách thành câu con, mỗi câu tra được 1 điều cụ thể → truy hồi từng câu → hợp nhất.

Tách riêng khỏi retrieval để cache kết quả decompose (LLM) và lặp rẻ phần merge.
"""
from __future__ import annotations

import re

from backend.llm import LLMClient

_SYSTEM = (
    "Bạn là chuyên gia pháp luật Việt Nam, chuyên tách câu hỏi tình huống thành "
    "các câu hỏi pháp lý nguyên tử để tra cứu điều luật."
)

_USER = """Câu hỏi sau là một TÌNH HUỐNG có thể chứa NHIỀU vấn đề pháp lý độc lập. Hãy tách thành các câu hỏi pháp lý NGUYÊN TỬ — mỗi câu chỉ về ĐÚNG MỘT vấn đề, để tra cứu được một điều luật cụ thể.

Quy tắc:
- Tách theo các VẤN ĐỀ PHÁP LÝ KHÁC NHAU — mỗi vế cần một ĐIỀU LUẬT / CĂN CỨ PHÁP LÝ RIÊNG.
  Nhận diện cả vế ẨN: vd "vi phạm bị phạt bao nhiêu VÀ phải khắc phục thế nào" = 2 vế (mức phạt
  + biện pháp khắc phục); "ký hợp đồng sai + chậm trả lương + bồi thường" = 3 vế (3 căn cứ khác nhau).
- TUYỆT ĐỐI KHÔNG tách các MỤC CON / LIỆT KÊ của CÙNG một vấn đề thành nhiều câu. Ví dụ:
  "những CHI PHÍ nào được hỗ trợ" = 1 vế (dù có nhiều loại chi phí); "ĐIỀU KIỆN gì để được..." =
  1 vế (dù nhiều điều kiện); "hồ sơ GỒM những gì" = 1 vế. Các thứ này nằm trong MỘT điều luật.
- Mỗi câu con NGẮN GỌN, độc lập, trả lời được bằng một điều luật. GIỮ NGUYÊN thuật ngữ pháp lý
  quan trọng (vd "đơn phương chấm dứt hợp đồng lao động", "thời hiệu khởi kiện", "phạt vi phạm hợp đồng").
- Nếu câu hỏi vốn chỉ có MỘT vấn đề pháp lý, trả về đúng MỘT câu (chính nó, rút gọn).
- Tối đa 5 câu con.
- CHỈ xuất danh sách đánh số, mỗi câu một dòng: "1. ...", "2. ...". KHÔNG giải thích gì thêm.

Câu hỏi: {q}"""

_NUM = re.compile(r"^\s*\d+[\.\)]\s*(.+?)\s*$")
_MAX_SUBS = 5


def parse(text: str) -> list[str]:
    """Lấy các dòng đánh số thành list câu con (đã loại trùng, giữ thứ tự)."""
    subs: list[str] = []
    for line in (text or "").splitlines():
        m = _NUM.match(line)
        if not m:
            continue
        s = m.group(1).strip().strip('"').strip()
        if len(s) >= 8 and s.lower() not in {x.lower() for x in subs}:
            subs.append(s)
    return subs[:_MAX_SUBS]


def decompose(question: str, llm: LLMClient | None = None) -> list[str]:
    """Trả về list câu con (≥1). Lỗi/không tách được → trả [question]. think=True để
    tách kỹ hơn (nhận diện vế ẩn)."""
    llm = llm or LLMClient()
    try:
        out = llm.complete(_SYSTEM, _USER.format(q=question), think=True)
        subs = parse(out)
    except Exception:  # noqa: BLE001 — LLM lỗi thì fallback câu gốc
        subs = []
    return subs or [question]
