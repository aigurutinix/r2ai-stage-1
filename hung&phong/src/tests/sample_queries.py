"""50 testcase mô phỏng câu hỏi nhân viên Bộ Tư pháp VN thường hỏi.

Mục đích: đánh giá end-to-end RAG (retrieval + LLM answer) trên corpus thực.
Phủ đa dạng: tra cứu cụ thể (số ký hiệu, Điều), khái niệm luật dân sự / lao
động / hình sự, thủ tục hành chính, so sánh lịch sử, và edge case.

Lưu ý: corpus hiện chỉ 50 docs (chủ yếu Sắc lệnh 1945-1950) → nhiều câu
hiện đại sẽ MISS. Đây cũng là thông tin cần đo.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TestCase:
    id: int
    category: str
    query: str
    # Doc nào ta KỲ VỌNG hệ thống trả về (None = không kỳ vọng cụ thể):
    expected_so_ky_hieu: str | None = None
    # Nếu corpus có doc liên quan thì True; nếu chắc chắn miss thì False.
    expected_in_corpus: bool = True


SAMPLE_QUERIES: list[TestCase] = [
    # ===== A. Tra cứu văn bản cụ thể (số ký hiệu) =====
    TestCase(1, "A_so_ky_hieu", "Sắc lệnh 159/SL ngày 17/11/1950 quy định những gì?", "159/SL", True),
    TestCase(2, "A_so_ky_hieu", "Cho tôi nội dung Sắc lệnh 85/SL về cải cách bộ máy tư pháp.", "85/SL", True),
    TestCase(3, "A_so_ky_hieu", "Sắc lệnh 76/SL về quy chế công chức áp dụng cho đối tượng nào?", "76/SL", True),
    TestCase(4, "A_so_ky_hieu", "Sắc lệnh 77/SL quy định chế độ công nhân tại các xí nghiệp quốc gia?", "77/SL", True),
    TestCase(5, "A_so_ky_hieu", "Bộ luật Lao động 45/2019/QH14 có hiệu lực từ khi nào?", "45/2019/QH14", False),
    TestCase(6, "A_so_ky_hieu", "Nghị định 145/2020/NĐ-CP hướng dẫn về vấn đề gì?", "145/2020/NĐ-CP", False),
    TestCase(7, "A_so_ky_hieu", "Thông tư 10/2020/TT-BLĐTBXH quy định mức lương tối thiểu vùng ra sao?", "10/2020/TT-BLĐTBXH", False),
    TestCase(8, "A_so_ky_hieu", "Quyết định 87/1999/QĐ-UB của UBND tỉnh Lâm Đồng nói về gì?", "87/1999/QĐ-UB", True),
    TestCase(9, "A_so_ky_hieu", "Chỉ thị 28/1999/CT-UB về niêm yết giá quy định trách nhiệm các bên thế nào?", "28/1999/CT-UB", True),
    TestCase(10, "A_so_ky_hieu", "Luật Doanh nghiệp 59/2020/QH14 có quy định mới gì so với luật cũ?", "59/2020/QH14", False),

    # ===== B. Tra cứu Điều khoản =====
    TestCase(11, "B_dieu", "Điều 36 Bộ luật Lao động 45/2019/QH14 quy định gì?", "45/2019/QH14", False),
    TestCase(12, "B_dieu", "Điều 1 Sắc lệnh 159/SL liệt kê những căn cứ ly hôn nào?", "159/SL", True),
    TestCase(13, "B_dieu", "Điều 5 Sắc lệnh 157/SL về tổ chức toà án nói gì?", "157/SL", True),
    TestCase(14, "B_dieu", "Điều 9 Sắc lệnh 77/SL quy định gì về điều kiện làm việc?", "77/SL", True),
    TestCase(15, "B_dieu", "Điều 27 Hiến pháp 2013 quy định về tuổi bầu cử là bao nhiêu?", None, False),
    TestCase(16, "B_dieu", "Điều 8 Sắc lệnh 76/SL về chế độ lương công chức ra sao?", "76/SL", True),
    TestCase(17, "B_dieu", "Điều 100 Bộ luật Dân sự về thừa kế quy định gì?", None, False),
    TestCase(18, "B_dieu", "Điều 15 Luật Hôn nhân Gia đình về thủ tục đăng ký kết hôn.", None, False),

    # ===== C. Luật dân sự / hôn nhân (concept) =====
    TestCase(19, "C_civil", "Trong luật pháp Việt Nam, vợ chồng có thể thuận tình ly hôn không?", None, True),
    TestCase(20, "C_civil", "Thủ tục ly hôn đơn phương cần những hồ sơ gì?", None, True),
    TestCase(21, "C_civil", "Quyền nuôi con sau ly hôn được pháp luật quy định ra sao?", None, True),
    TestCase(22, "C_civil", "Nghĩa vụ cấp dưỡng sau ly hôn áp dụng trong trường hợp nào?", None, True),
    TestCase(23, "C_civil", "Phân chia tài sản chung của vợ chồng khi ly hôn theo nguyên tắc nào?", None, True),
    TestCase(24, "C_civil", "Trẻ em dưới 16 tuổi có được nhận thừa kế không?", None, False),
    TestCase(25, "C_civil", "Di chúc miệng có giá trị pháp lý không?", None, False),
    TestCase(26, "C_civil", "Người không có khả năng nhận thức có được tham gia ký hợp đồng không?", None, False),
    TestCase(27, "C_civil", "Hợp đồng vô hiệu trong những trường hợp nào?", None, False),
    TestCase(28, "C_civil", "Phân biệt quyền nhân thân và quyền tài sản.", None, False),

    # ===== D. Lao động =====
    TestCase(29, "D_labor", "Thời gian thử việc tối đa cho lao động phổ thông là bao lâu?", None, False),
    TestCase(30, "D_labor", "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động trong trường hợp nào?", None, True),
    TestCase(31, "D_labor", "Trợ cấp thôi việc được tính như thế nào?", None, False),
    TestCase(32, "D_labor", "Thời gian làm thêm giờ tối đa trong một tuần là bao nhiêu?", None, False),
    TestCase(33, "D_labor", "Lao động nữ mang thai được hưởng những chế độ gì?", None, False),

    # ===== E. Hình sự =====
    TestCase(34, "E_crime", "Tội cố ý gây thương tích bị xử phạt như thế nào?", None, False),
    TestCase(35, "E_crime", "Tội lừa đảo chiếm đoạt tài sản theo Bộ luật Hình sự.", None, False),
    TestCase(36, "E_crime", "Thời hiệu truy cứu trách nhiệm hình sự là bao lâu?", None, False),
    TestCase(37, "E_crime", "Phân biệt tội phạm rất nghiêm trọng và đặc biệt nghiêm trọng.", None, False),
    TestCase(38, "E_crime", "Người dưới 18 tuổi phạm tội bị xử lý thế nào?", None, False),

    # ===== F. Thủ tục hành chính =====
    TestCase(39, "F_proc", "Hồ sơ đăng ký kết hôn cần những giấy tờ gì?", None, False),
    TestCase(40, "F_proc", "Quy trình khởi kiện vụ án dân sự bao gồm những bước nào?", None, False),
    TestCase(41, "F_proc", "Thời hạn giải quyết khiếu nại hành chính là bao lâu?", None, False),
    TestCase(42, "F_proc", "Thủ tục công chứng hợp đồng mua bán nhà ở.", None, False),
    TestCase(43, "F_proc", "Quy định về thẩm quyền giải quyết tranh chấp đất đai.", None, False),
    TestCase(44, "F_proc", "Trình tự kháng cáo bản án sơ thẩm trong vụ án dân sự.", None, False),

    # ===== G. So sánh / lịch sử =====
    TestCase(45, "G_compare", "So sánh quyền tự do hôn nhân theo Sắc lệnh 159/SL với Luật Hôn nhân Gia đình hiện hành.", "159/SL", True),
    TestCase(46, "G_compare", "Hiến pháp 2013 khác Hiến pháp 1992 ở những điểm chính nào?", None, False),
    TestCase(47, "G_compare", "Tổ chức toà án nhân dân thời kỳ 1950 khác hiện nay ra sao?", "157/SL", True),

    # ===== H. Edge cases =====
    TestCase(48, "H_edge", "ly hôn cần gì", None, True),
    TestCase(49, "H_edge", "Văn bản 159 năm 1950 ạ", None, True),
    TestCase(50, "H_edge", "Hôm nay tôi muốn hỏi về luật đất đai 2024 mới sửa đổi.", None, False),
]
