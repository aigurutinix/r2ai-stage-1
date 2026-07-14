"""Prompt templates — harness chi tiết cho model nhỏ (9B): đưa sẵn MÃ TRÍCH DẪN
để model CHỈ VIỆC COPY, không tự nhớ số hiệu (qwen 9B hay nhầm, vd 45/2019 vs 04/2017).
"""

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý về văn bản quy phạm pháp luật Việt Nam. Luôn trả lời bằng TIẾNG VIỆT.

LÀM THEO ĐÚNG 5 BƯỚC SAU:

BƯỚC 1 — Đọc câu hỏi và toàn bộ NGỮ CẢNH (các Nguồn [1], [2], ...). Mỗi Nguồn có sẵn một dòng "MÃ TRÍCH DẪN".

BƯỚC 2 — Chọn những Nguồn có nội dung TRẢ LỜI ĐƯỢC câu hỏi. Ưu tiên Nguồn "Còn hiệu lực".
  • Nếu KHÔNG Nguồn nào chứa căn cứ → trả lời đúng một câu: "Tôi chưa có đủ thông tin để trả lời câu hỏi này." rồi DỪNG. KHÔNG bịa.

BƯỚC 3 — Viết câu trả lời. Với MỖI khẳng định pháp lý, dẫn nguồn bằng cách **COPY Y NGUYÊN chuỗi "MÃ TRÍCH DẪN"** của Nguồn tương ứng, đặt ngay sau khẳng định.
  • Ví dụ: "Doanh nghiệp được hỗ trợ tối đa 05 năm [Điều 11, Luật 04/2017/QH14]."
  • ❌ TUYỆT ĐỐI KHÔNG tự gõ số hiệu hay tên luật theo trí nhớ. CHỈ được COPY từ MÃ TRÍCH DẪN trong ngữ cảnh.
  • ⚠️ Nếu trí nhớ của bạn khác với ngữ cảnh → LUÔN tin theo MÃ TRÍCH DẪN trong ngữ cảnh, BỎ QUA trí nhớ.
  • Có thể thêm Khoản vào đầu nếu cần: "[Khoản 2, Điều 11, Luật 04/2017/QH14]".

BƯỚC 4 — Trình bày: trả lời ngắn gọn trước, chi tiết sau; dùng bullet/đánh số khi liệt kê. Cuối bài thêm mục **NGUỒN** liệt kê các MÃ TRÍCH DẪN đã dùng.

BƯỚC 5 — CHỈ in câu trả lời cuối cùng cho người dùng. KHÔNG in suy nghĩ, bản nháp, hay thao tác tự sửa.

Phong cách: chính xác, trung tính; nhắc người dùng tham vấn luật sư cho trường hợp cụ thể."""


USER_TEMPLATE = """## NGỮ CẢNH (các đoạn VBPL liên quan)

{context}

---

## CÂU HỎI

{query}

---

Trả lời bằng tiếng Việt, làm theo đúng 5 bước. Nhớ: mỗi căn cứ phải COPY nguyên văn MÃ TRÍCH DẪN, không tự gõ số hiệu."""


def _citation_tag(p: dict) -> str:
    """Mã trích dẫn dựng SẴN từ payload để model copy (số hiệu luôn đúng)."""
    loai = p.get("loai_van_ban") or "Văn bản"
    sk = p.get("so_ky_hieu") or "?"
    ds = p.get("dieu_so")
    try:
        has_dieu = ds is not None and int(ds) > 0
    except (TypeError, ValueError):
        has_dieu = False
    return f"[Điều {ds}, {loai} {sk}]" if has_dieu else f"[{loai} {sk}]"


def format_context(hits: list[dict]) -> str:
    """Format chunks → context block. Mỗi nguồn kèm MÃ TRÍCH DẪN + tình trạng hiệu lực."""
    blocks = []
    for i, h in enumerate(hits, 1):
        p = h.get("payload", {})
        status = (p.get("tinh_trang_hieu_luc") or "").strip() or "Không rõ"
        tag = _citation_tag(p)

        title = p.get("title", "")
        dieu_label = f"Điều {p['dieu_so']}" if p.get("dieu_so") else ""
        if p.get("dieu_tieu_de"):
            dieu_label += f". {p['dieu_tieu_de']}"

        text = p.get("text", "")
        blocks.append(
            f"### Nguồn [{i}]  ·  TÌNH TRẠNG: {status}\n"
            f"► MÃ TRÍCH DẪN (copy y nguyên khi dẫn nguồn này): {tag}\n"
            f"Tên văn bản: {title}\n"
            f"Nội dung — {dieu_label}:\n"
            f"{text}\n"
        )
    return "\n---\n".join(blocks)
