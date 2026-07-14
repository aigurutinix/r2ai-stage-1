"""Phân rã câu hỏi mức "VỪA" — bảo thủ, CHẶN trôi dạt (khác bản SÂU của decompose.py).

Bài học từ Review #5 (v22 sâu thất bại): phân rã sâu trôi dạt khỏi neo pháp lý vì
(1) viết lại/diễn giải mạnh, (2) thêm vế không có trong câu gốc, (3) biến câu hỏi thành
câu khẳng định, (4) tách liệt kê của 1 vấn đề. Prompt "vừa" giữ kiểu CHIA NHẸ của bản nông
nhưng ràng buộc cứng để KHÔNG trôi: tái dùng nguyên văn thuật ngữ, không thêm ý, mỗi vế là
câu hỏi, không tách liệt kê, chỉ tách khi thật sự đa-vấn-đề. think=False (bớt "sáng tạo").
"""
from __future__ import annotations

import re

from backend.llm import LLMClient

_SYSTEM = (
    "Bạn là chuyên gia pháp luật Việt Nam, tách câu hỏi tình huống thành các câu hỏi pháp lý "
    "nhỏ để TRA CỨU điều luật. Bạn tách RẤT BẢO THỦ, GIỮ NGUYÊN VĂN từ ngữ của câu gốc, "
    "không diễn giải lại. Bạn CHỈ viết bằng tiếng Việt có dấu, TUYỆT ĐỐI không dùng chữ Hán "
    "hay bất kỳ ký tự tiếng nước ngoài nào."
)

_USER = """Tách câu hỏi sau thành các câu hỏi pháp lý NHỎ để TRA CỨU điều luật (đây là bước truy hồi, KHÔNG phải trả lời).

QUY TẮC BẮT BUỘC (vi phạm là hỏng):
1. CHỈ DÙNG TIẾNG VIỆT CÓ DẤU: toàn bộ câu con viết HOÀN TOÀN bằng tiếng Việt. TUYỆT ĐỐI KHÔNG dùng chữ Hán / chữ Trung Quốc / chữ Nhật / chữ Hàn hay bất kỳ ký tự nước ngoài nào. Ví dụ SAI: "dịch vụ trung gian介绍 lao động" — PHẢI viết "dịch vụ trung gian giới thiệu lao động". Mọi thuật ngữ pháp lý dùng đúng từ tiếng Việt.
2. TÁI DÙNG NGUYÊN VĂN: mỗi câu con phải dùng lại CHÍNH XÁC các thuật ngữ pháp lý, cụm danh từ, số văn bản, tên điều luật có trong câu gốc. TUYỆT ĐỐI KHÔNG thay từ đồng nghĩa, KHÔNG diễn giải lại (ví dụ: KHÔNG đổi "cưỡng chế" thành "đình chỉ"; KHÔNG đổi "đơn phương chấm dứt hợp đồng lao động" thành "sa thải"; KHÔNG đổi "quá cảnh hàng hóa" thành "lưu trú").
3. KHÔNG THÊM Ý MỚI: chỉ tách những vế ĐÃ CÓ trong câu gốc. KHÔNG tự suy diễn thêm câu hỏi. Ví dụ: câu gốc chỉ hỏi "nguyên tắc" thì KHÔNG được thêm vế hỏi "điều kiện" hay "thủ tục". KHÔNG bịa thêm số văn bản / số nghị định không có trong câu gốc.
4. MỖI CÂU CON LÀ MỘT CÂU HỎI, kết thúc bằng dấu "?". KHÔNG viết câu khẳng định nêu sẵn đáp án (ví dụ KHÔNG viết "Công ty phải nộp đơn trong 30 ngày.").
5. KHÔNG TÁCH LIỆT KÊ của cùng một vấn đề: "hồ sơ gồm những gì", "điều kiện gồm những gì", "những chi phí nào được hỗ trợ", "tiêu chí xác định gồm những gì" = MỘT câu con (dù có nhiều mục), vì chúng nằm trong MỘT điều luật.
6. CHỈ tách khi câu gốc thật sự chứa TỪ 2 VẤN ĐỀ PHÁP LÝ KHÁC NHAU (cần điều luật / căn cứ khác nhau). Nếu chỉ MỘT vấn đề → trả về ĐÚNG MỘT câu (chính câu gốc, giữ nguyên thuật ngữ, có thể bỏ phần tình huống thừa).
7. Tối đa 3 câu con.
8. CHỈ xuất danh sách đánh số, mỗi câu một dòng: "1. ...", "2. ...". KHÔNG giải thích gì thêm.

Câu hỏi: {q}"""

_NUM = re.compile(r"^\s*\d+[\.\)]\s*(.+?)\s*$")
# Hán/Nhật/Hàn + ký tự CJK mở rộng — dùng để phát hiện rò chữ nước ngoài
_CJK = re.compile(r"[　-〿㐀-䶿一-鿿豈-﫿぀-ヿ가-힯]")
_MAX_SUBS = 3


def has_cjk(text: str) -> bool:
    """True nếu có chữ Hán/Nhật/Hàn (Qwen đôi khi rò) → cần retry/loại."""
    return bool(_CJK.search(text or ""))


def parse(text: str) -> list[str]:
    """Lấy các dòng đánh số thành list câu con (loại trùng, giữ thứ tự)."""
    subs: list[str] = []
    for line in (text or "").splitlines():
        m = _NUM.match(line)
        if not m:
            continue
        s = m.group(1).strip().strip('"').strip()
        if len(s) >= 8 and s.lower() not in {x.lower() for x in subs}:
            subs.append(s)
    return subs[:_MAX_SUBS]


_RETRY_NOTE = "\n\nLƯU Ý: lần trước bạn đã rò chữ Hán/nước ngoài — lần này CHỈ viết tiếng Việt có dấu."


def decompose_mid(question: str, llm: LLMClient | None = None) -> list[str]:
    """Trả về list câu con (>=1). think=False để bớt trôi.

    Nếu phát hiện rò chữ CJK → retry 1 lần với nhắc nhở; vẫn còn → loại các câu dính CJK.
    Loại sạch mà rỗng → fallback [question] (an toàn, không đẩy rác vào retrieval)."""
    llm = llm or LLMClient()
    subs: list[str] = []
    for attempt in range(2):
        prompt = _USER.format(q=question) + (_RETRY_NOTE if attempt else "")
        try:
            subs = parse(llm.complete(_SYSTEM, prompt, think=False))
        except Exception:  # noqa: BLE001 — LLM lỗi thì fallback câu gốc
            subs = []
        if not any(has_cjk(s) for s in subs):
            break
    clean = [s for s in subs if not has_cjk(s)]  # bỏ câu vẫn dính CJK sau retry
    return clean or [question]
