# -*- coding: utf-8 -*-
"""
crawl_and_chunk.py  —  Cào & băm nhỏ (chunking) văn bản luật trên vbpl.vn

Khác biệt so với bản cũ:
  1. URL đúng định dạng mới: /van-ban/chi-tiet/<slug>--<id>
  2. Bóc tách theo cấu trúc THẬT của trang: div.preview-content + các thẻ
     <p class="prov-chapter | prov-article | prov-clause | prov-point | prov-content">
     (KHÔNG còn class "toanvancontent" — class đó không tồn tại)
  3. vbpl.vn dùng Next.js (render bằng JavaScript). Script tự thử bằng `requests`
     trước cho nhanh; nếu trang trả về rỗng thì TỰ ĐỘNG chuyển sang Playwright
     để render đúng như trình duyệt.

Cài đặt:
    pip install requests beautifulsoup4
    # Bản dự phòng (khuyến nghị, vì trang render JS):
    pip install playwright
    playwright install chromium
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import time
import sys

# ============================================================
# 1. DANH SÁCH VĂN BẢN MỤC TIÊU
# ============================================================
TARGET_LAWS = [
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-ho-tro-doanh-nghiep-nho-va-vua-so-04-2017-qh14--128706",
        "ma_van_ban": "04/2017/QH14",
        "ten_van_ban_chuan": "Luật 04/2017/QH14 Hỗ trợ doanh nghiệp nhỏ và vừa",
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-doanh-nghiep-so-59-2020-qh14--142881",
        "ma_van_ban": "59/2020/QH14",
        "ten_van_ban_chuan": "Luật 59/2020/QH14 Doanh nghiệp",
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-doanh-nghiep-so-59-2020-qh14--142847",
        "ma_van_ban": "59/2020/QH14",
        "ten_van_ban_chuan": "Luật 59/2020/QH14 Doanh nghiệp",
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/bo-luat-lao-dong-so-45-2019-qh14--139264",
        "ma_van_ban": "45/2019/QH14",
        "ten_van_ban_chuan": "Luật 45/2019/QH14 Bộ luật Lao động"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-quyet-so-78-2025-ubtvqh15-ve-viec-ghi-nhan-thoi-gian-dong-bao-hiem-xa-hoi-bat-buoc-cua-chu-ho-kinh-doanh-da-tham-gia-truoc-ngay-luat-bao-hiem-xa-hoi-so-41-2024-qh15-co-hieu-luc-thi-hanh--179965",
        "ma_van_ban": "41/2024/QH15",
        "ten_van_ban_chuan": "Luật 41/2024/QH15 Bảo hiểm xã hội"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-sua-doi-bo-sung-mot-so-dieu-cua-luat-thue-gia-tri-gia-tang-so-13-2008-qh12-so-31-2013-qh13--30614",
        "ma_van_ban": "13/2008/QH12",
        "ten_van_ban_chuan": "Luật 13/2008/QH12 Thuế giá trị gia tăng"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-thue-gia-tri-gia-tang-so-13-2008-qh12--12806",
        "ma_van_ban": "13/2008/QH12",
        "ten_van_ban_chuan": "Luật sửa đổi, bổ sung 13/2008/QH12 Thuế giá trị gia tăng"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-thue-thu-nhap-doanh-nghiep-so-14-2008-qh12--12807",
        "ma_van_ban": "14/2008/QH12",
        "ten_van_ban_chuan": "Luật 14/2008/QH12 Thuế thu nhập doanh nghiệp"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/thong-tu-so-123-2012-tt-btc-huong-dan-thi-hanh-mot-so-dieu-cua-luat-thue-thu-nhap-doanh-nghiep-so-14-2008-qh12-va-huong-dan-thi-hanh-nghi-dinh-so-124-2008-nd-cp-ngay-11-12-2008-nghi-dinh-so-122-2011-nd-cp-ngay-27-12-2011-cua-chinh-phu-quy-dinh-chi-tiet-thi-hanh-mot-so-dieu-cua-luat-thue-thu-nhap-doanh-nghiep--46367",
        "ma_van_ban": "14/2008/QH12",
        "ten_van_ban_chuan": "Thông tư số 123/2012/TT-BTC"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/thong-tu-so-18-2011-tt-btc-sua-doi-bo-sung-thong-tu-so-130-2008-tt-btc-ngay-26-12-2008-cua-bo-tai-chinh-huong-dan-thi-hanh-mot-so-dieu-cua-luat-thue-thu-nhap-doanh-nghiep-so-14-2008-qh12-va-huong-dan-thi-hanh-nghi-dinh-so-124-2008-nd-cp-ngay-11-thang-12-nam-2008-cua-chinh-phu-quy-dinh-chi-tiet-thi-hanh-mot-so-dieu-cua-luat-thue-thu-nhap-doanh-nghiep--26201",
        "ma_van_ban": "14/2008/QH12",
        "ten_van_ban_chuan": "Thông tư số 18/2011/TT-BTC Sửa đổi, bổ sung"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/thong-tu-so-130-2008-tt-btc-huong-dan-thi-hanh-mot-so-dieu-cua-luat-thue-thu-nhap-doanh-nghiep-so-14-2008-qh12-va-huong-dan-thi-hanh-nghi-dinh-so-124-2008-nd-cp-ngay-11-thang-12-nam-2008-cua-chinh-phu-quy-dinh-chi-tiet-thi-hanh-mot-so-dieu-cua-luat-thue-thu-nhap-doanh-nghiep--12617",
        "ma_van_ban": "14/2008/QH12",
        "ten_van_ban_chuan": "Thông tư số 130/2008/TT-BTC"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/quyet-dinh-so-3027-qd-btc-ve-viec-dinh-chinh-thong-tu-so-130-2008-tt-btc-ngay-26-12-2008-cua-bo-tai-chinh-huong-dan-thi-hanh-mot-so-dieu-cua-luat-thue-thu-nhap-doanh-nghiep-so-14-2008-qh12-va-huong-dan-thi-hanh-nghi-dinh-so-124-2008-nd-cp-ngay-11-thang-12-nam-2008-cua-chinh-phu-quy-dinh-chi-tiet-thi-hanh-mot-so-dieu-cua-luat-thue-thu-nhap-doanh-nghiep--107966",
        "ma_van_ban": "14/2008/QH12",
        "ten_van_ban_chuan": "Quyết định số 3027/QĐ-BTC Về việc đính chính Thông tư số 130/2008/TT-BTC"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-80-2021-nd-cp-quy-dinh-chi-tiet-va-huong-dan-thi-hanh-mot-so-dieu-cua-luat-ho-tro-doanh-nghiep-nho-va-vua--158783",
        "ma_van_ban": "80/2021/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định số 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành một số điều của Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/thong-tu-so-52-2023-tt-btc-huong-dan-co-che-su-dung-kinh-phi-ngan-sach-nha-nuoc-chi-thuong-xuyen-ho-tro-doanh-nghiep-nho-va-vua-theo-quy-dinh-tai-nghi-dinh-so-80-2021-nd-cp-ngay-26-thang-8-nam-2021-cua-chinh-phu--163729",
        "ma_van_ban": "80/2021/NĐ-CP",
        "ten_van_ban_chuan": "Thông tư số 52/2023/TT-BTC Hướng dẫn cơ chế sử dụng kinh phí ngân sách nhà nước chi thường xuyên hỗ trợ doanh nghiệp nhỏ và vừa theo quy định tại Nghị định số 80/2021/NĐ-CP ngày 26 tháng 8 năm 2021 của Chính phủ"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/thong-tu-so-06-2022-tt-bkhdt-huong-dan-mot-so-dieu-cua-nghi-dinh-so-80-2021-nd-cp-ngay-26-8-2021-cua-chinh-phu-quy-dinh-chi-tiet-va-huong-dan-thi-hanh-mot-so-dieu-cua-luat-ho-tro-doanh-nghiep-nho-va-vua--158782",
        "ma_van_ban": "80/2021/NĐ-CP",
        "ten_van_ban_chuan": "Thông tư số 06/2022/TT-BKHĐT Hướng dẫn một số điều của Nghị định số 80/2021/NĐ-CP ngày 26/8/2021 của Chính phủ quy định chi tiết và hướng dẫn thi hành một số điều của Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-01-2021-nd-cp-ve-dang-ky-doanh-nghiep--153870",
        "ma_van_ban": "01/2021/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định 01/2021/NĐ-CP về đăng ký doanh nghiệp"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-47-2021-nd-cp-quy-dinh-chi-tiet-mot-so-dieu-cua-luat-doanh-nghiep--147722",
        "ma_van_ban": "47/2021/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định 47/2021/NĐ-CP Quy định chi tiết một số điều của Luật Doanh nghiệp"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-16-2023-nd-cp-ve-to-chuc-quan-ly-va-hoat-dong-cua-doanh-nghiep-truc-tiep-phuc-vu-quoc-phong-an-ninh-va-doanh-nghiep-ket-hop-kinh-te-voi-quoc-phong-an-ninh-sua-doi-quy-dinh-tai-diem-g-khoan-1-dieu-23-nghi-dinh-so-47-2021-nd-cp-ngay-01-thang-4-nam-2021-cua-chinh-phu-quy-dinh-chi-tiet-mot-so-dieu-cua-luat-doanh-nghiep--166416",
        "ma_van_ban": "47/2021/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định số 16/2023/NĐ-CP Về tổ chức quản lý và hoạt động của doanh nghiệp trực tiếp phục vụ quốc phòng, an ninh và doanh nghiệp kết hợp kinh tế với quốc phòng, an ninh"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-145-2020-nd-cp-quy-dinh-chi-tiet-va-huong-dan-thi-hanh-mot-so-dieu-cua-bo-luat-lao-dong-ve-dieu-kien-lao-dong-va-quan-he-lao-dong--152668",
        "ma_van_ban": "145/2020/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định 145/2020/NĐ-CP Quy định chi tiết và hướng dẫn thi hành một số điều của Bộ luật Lao động về điều kiện lao động và quan hệ lao động"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-126-2020-nd-cp-quy-dinh-chi-tiet-mot-so-dieu-cua-luat-quan-ly-thue--146459",
        "ma_van_ban": "126/2020/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định 126/2020/NĐ-CP Quy định chi tiết một số điều của Luật Quản lý thuế"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-91-2022-nd-cp-sua-doi-bo-sung-mot-so-dieu-cua-nghi-dinh-so-126-2020-nd-cp-ngay-19-thang-10-nam-2020-cua-chinh-phu-quy-dinh-chi-tiet-mot-so-dieu-cua-luat-quan-ly-thue--157326",
        "ma_van_ban": "126/2020/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định số 91/2022/NĐ-CP Sửa đổi, bổ sung một số điều của Nghị định số 126/2020/NĐ-CP ngày 19 tháng 10 năm 2020 của Chính phủ quy định chi tiết một số điều của Luật Quản lý thuế"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/thong-tu-21-2026-tt-btc-sua-doi-bo-sung-mot-so-dieu-cua-thong-tu-so-80-2021-tt-btc-ngay-29-thang-9-nam-2021-cua-bo-truong-bo-tai-chinh-huong-dan-thi-hanh-mot-so-dieu-cua-luat-quan-ly-thue-va-nghi-dinh-so-126-2020-nd-cp-ngay-19-thang-10-nam-2020-cua-chinh-phu-quy-dinh-chi-tiet-mot-so-dieu-cua-luat-quan-ly-thue--23cbb8e0-6006-11f1-bb86-3919c0b56476",
        "ma_van_ban": "126/2020/NĐ-CP",
        "ten_van_ban_chuan": "Thông tư 21/2026/TT-BTC Sửa đổi, bổ sung một số điều của Thông tư số 80/2021/TT-BTC ngày 29 tháng 9 năm 2021 của Bộ trưởng Bộ Tài chính hướng dẫn thi hành một số điều của Luật Quản lý thuế và Nghị định số 126/2020/NĐ-CP ngày 19 tháng 10 năm 2020 của Chính phủ quy định chi tiết một số điều của Luật Quản lý thuế"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/thong-tu-so-94-2025-tt-btc-sua-doi-bo-sung-mot-so-dieu-cua-thong-tu-so-80-2021-tt-btc-ngay-29-thang-9-nam-2021-cua-bo-tai-chinh-huong-dan-thi-hanh-mot-so-dieu-cua-luat-quan-ly-thue-va-nghi-dinh-so-126-2020-nd-cp-ngay-19-thang-10-nam-2020-cua-chinh-phu-quy-dinh-chi-tiet-mot-so-dieu-cua-luat-quan-ly-thue-va-sua-doi-bo-sung-mot-so-mau-bieu-cua-thong-tu-so-40-2021-tt-btc-ngay-01-thang-6-nam-2021-cua-bo-tai-chinh-huong-dan-thue-gia-tri-gia-tang-thue-thu-nhap-ca-nhan-va-quan-ly-thue-doi-voi-ho-kinh-doanh-ca-nhan-kinh-doanh--187379",
        "ma_van_ban": "126/2020/NĐ-CP",
        "ten_van_ban_chuan": "Thông tư số 94/2025/TT-BTC Sửa đổi, bổ sung một số điều của Thông tư số 80/2021/TT-BTC ngày 29 tháng 9 năm 2021 của Bộ Tài chính hướng dẫn thỉ hành một số điều của Luật Quản lý thuế và Nghị định số 126/2020/NĐ-CP ngày 19 tháng 10 năm 2020 của Chính phủ quy định chi tiết một số điều của Luật Quản lý thuế và sửa đổi, bổ sung một số mẫu biểu của Thông tư số 40/2021/TT-BTC ngày 01 tháng 6 năm 2021 của Bộ Tài chính hướng dẫn thuế giá trị gia tăng, thuế thu nhập cá nhân và quản lý thuế đối vói hộ kỉnh doanh, cá nhân kỉnh doanh"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/thong-tu-so-80-2021-tt-btc-huong-dan-thi-hanh-mot-so-dieu-cua-luat-quan-ly-thue-va-nghi-dinh-so-126-2020-nd-cp-ngay-19-thang-10-nam-2020-cua-chinh-phu-quy-dinh-chi-tiet-mot-so-dieu-cua-luat-quan-ly-thue--151086",
        "ma_van_ban": "126/2020/NĐ-CP",
        "ten_van_ban_chuan": "Thông tư số 80/2021/TT-BTC Hướng dẫn thi hành một số điều của Luật Quản lý thuế và Nghị định số 126/2020/NĐ-CP ngày 19 tháng 10 năm 2020 của Chính phủ quy định chi tiết một số điều của Luật Quản lý thuế"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/van-ban-hop-nhat-so-15-vbhn-btc-huong-dan-thi-hanh-mot-so-dieu-cua-luat-quan-ly-thue-va-nghi-dinh-so-126-2020-nd-cp-ngay-19-thang-10-nam-2020-cua-chinh-phu-quy-dinh-chi-tiet-mot-so-dieu-cua-luat-quan-ly-thue--167653",
        "ma_van_ban": "126/2020/NĐ-CP",
        "ten_van_ban_chuan": "Văn bản hợp nhất số 15/VBHN-BTC Hướng dẫn thi hành một số điều của Luật Quản lý thuế và Nghị định số 126/2020/NĐ-CP ngày 19 tháng 10 năm 2020 của Chính phủ quy định chi tiết một số điều của Luật Quản lý thuế"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-dau-tu-so-61-2020-qh14--142867",
        "ma_van_ban": "61/2020/QH14",
        "ten_van_ban_chuan": "Luật 61/2020/QH14 Đầu tư"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-sua-doi-bo-sung-mot-so-dieu-cua-luat-xu-ly-vi-pham-hanh-chinh-so-67-2020-qh14--146541",
        "ma_van_ban": "67/2020/QH14",
        "ten_van_ban_chuan": "Luật 67/2020/QH14 Sửa đổi, bổ sung một số điều của Luật Xử lý vi phạm hành chính"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-bao-ve-quyen-loi-nguoi-tieu-dung-so-19-2023-qh15--161263",
        "ma_van_ban": "19/2023/QH15",
        "ten_van_ban_chuan": "Luật 19/2023/QH15 Bảo vệ quyền lợi người tiêu dùng"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-125-2020-nd-cp-quy-dinh-xu-phat-vi-pham-hanh-chinh-ve-thue-hoa-don--146458",
        "ma_van_ban": "125/2020/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định số 125/2020/NĐ-CP Quy định xử phạt vi phạm hành chính về thuế, hóa đơn"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/van-ban-hop-nhat-nghi-dinh-so-125-2020-nd-cp-quy-dinh-xu-phat-vi-pham-hanh-chinh-ve-thue-hoa-don--ab302360-68a4-11f1-a9cd-836a3b797bec",
        "ma_van_ban": "125/2020/NĐ-CP",
        "ten_van_ban_chuan": "Văn bản hợp nhất Nghị định số 125/2020/NĐ-CP Quy định xử phạt vi phạm hành chính về thuế, hóa đơn"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-12-2022-nd-cp-quy-dinh-xu-phat-vi-pham-hanh-chinh-trong-linh-vuc-lao-dong-bao-hiem-xa-hoi-nguoi-lao-dong-viet-nam-di-lam-viec-o-nuoc-ngoai-theo-hop-dong--153913",
        "ma_van_ban": "12/2022/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định số 12/2022/NĐ-CP Quy định xử phạt vi phạm hành chính trong lĩnh vực lao động, bảo hiểm xã hội, người lao động Việt Nam đi làm việc ở nước ngoài theo hợp đồng"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/thong-tu-so-06-2022-tt-bkhdt-huong-dan-mot-so-dieu-cua-nghi-dinh-so-80-2021-nd-cp-ngay-26-8-2021-cua-chinh-phu-quy-dinh-chi-tiet-va-huong-dan-thi-hanh-mot-so-dieu-cua-luat-ho-tro-doanh-nghiep-nho-va-vua--158782",
        "ma_van_ban": "06/2022/TT-BKHĐT",
        "ten_van_ban_chuan": "Thông tư số 06/2022/TT-BKHĐT Hướng dẫn một số điều của Nghị định số 80/2021/NĐ-CP ngày 26/8/2021 của Chính phủ quy định chi tiết và hướng dẫn thi hành một số điều của Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
    },
    {
        "url": "http://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-65-2023-nd-cp-quy-dinh-chi-tiet-mot-so-dieu-va-bien-phap-thi-hanh-luat-so-huu-tri-tue-ve-so-huu-cong-nghiep-bao-ve-quyen-so-huu-cong-nghiep-quyen-doi-voi-giong-cay-trong-va-quan-ly-nha-nuoc-ve-so-huu-tri-tue--164372",
        "ma_van_ban": "65/2023/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định số 65/2023/NĐ-CP Quy định chi tiết một số điều và biện pháp thi hành Luật Sở hữu trí tuệ về sở hữu công nghiệp, bảo vệ quyền sở hữu công nghiệp, quyền đối với giống cây trồng và quản lý nhà nước về sở hữu trí tuệ"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/thong-tu-so-23-2023-tt-bkhcn-quy-dinh-chi-tiet-mot-so-dieu-cua-luat-so-huu-tri-tue-va-bien-phap-thi-hanh-nghi-dinh-so-65-2023-nd-cp-ngay-23-thang-8-nam-2023-cua-chinh-phu-quy-dinh-chi-tiet-mot-so-dieu-va-bien-phap-thi-hanh-luat-so-huu-tri-tue-ve-so-huu-cong-nghiep-bao-ve-quyen-so-huu-cong-nghiep-quyen-doi-voi-giong-cay-trong-va-quan-ly-nha-nuoc-ve-so-huu-tri-tue-lien-quan-den-thu-tuc-xac-lap-quyen-so-huu-cong-nghiep-va-bao-dam-thong-tin-so-huu-cong-nghiep--164373",
        "ma_van_ban": "165/2023/NĐ-CP",
        "ten_van_ban_chuan": "Thông tư số 23/2023/TT-BKHCN Quy định chi tiết một số điều của Luật Sở hữu trí tuệ và biện pháp thi hành Nghị định số 65/2023/NĐ-CP ngày 23 tháng 8 năm 2023 của Chính phủ quy định chi tiết một số điều và biện pháp thi hành Luật Sở hữu trí tuệ về sở hữu công nghiệp, bảo vệ quyền sở hữu công nghiệp, quyền đối với giống cây trồng và quản lý nhà nước về sở hữu trí tuệ liên quan đến thủ tục xác lập quyền sở hữu công nghiệp và bảo đảm thông tin sở hữu công nghiệp"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-70-2023-nd-cp-sua-doi-bo-sung-mot-so-dieu-cua-nghi-dinh-so-152-2020-nd-cp-ngay-30-thang-12-nam-2020-cua-chinh-phu-quy-dinh-ve-nguoi-lao-dong-nuoc-ngoai-lam-viec-tai-viet-nam-va-tuyen-dung-quan-ly-nguoi-lao-dong-viet-nam-lam-viec-cho-to-chuc-ca-nhan-nuoc-ngoai-tai-viet-nam--162330",
        "ma_van_ban": "152/2020/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định số 70/2023/NĐ-CP Sửa đổi, bổ sung một số điều của Nghị định số 152/2020/NĐ-CP ngày 30 tháng 12 năm 2020 của Chính phủ quy định về người lao động nước ngoài làm việc tại Việt Nam và tuyển dụng, quản lý người lao động Việt Nam làm việc cho tổ chức, cá nhân nước ngoài tại Việt Nam"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-152-2020-nd-cp-quy-dinh-ve-nguoi-lao-dong-nuoc-ngoai-lam-viec-tai-viet-nam-va-tuyen-dung-quan-ly-nguoi-lao-dong-viet-nam-lam-viec-cho-to-chuc-ca-nhan-nuoc-ngoai-tai-viet-nam--152669",
        "ma_van_ban": "152/2020/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định số 152/2020/NĐ-CP Quy định về người lao động nước ngoài làm việc tại Việt Nam và tuyển dụng, quản lý người lao động Việt Nam làm việc cho tổ chức, cá nhân nước ngoài tại Việt Nam"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-70-2023-nd-cp-sua-doi-bo-sung-mot-so-dieu-cua-nghi-dinh-so-152-2020-nd-cp-ngay-30-thang-12-nam-2020-cua-chinh-phu-quy-dinh-ve-nguoi-lao-dong-nuoc-ngoai-lam-viec-tai-viet-nam-va-tuyen-dung-quan-ly-nguoi-lao-dong-viet-nam-lam-viec-cho-to-chuc-ca-nhan-nuoc-ngoai-tai-viet-nam--162330",
        "ma_van_ban": "70/2023/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định số 70/2023/NĐ-CP Sửa đổi, bổ sung một số điều của Nghị định số 152/2020/NĐ-CP ngày 30 tháng 12 năm 2020 của Chính phủ quy định về người lao động nước ngoài làm việc tại Việt Nam và tuyển dụng, quản lý người lao động Việt Nam làm việc cho tổ chức, cá nhân nước ngoài tại Việt Nam"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-70-2025-nd-cp-sua-doi-bo-sung-mot-so-dieu-cua-nghi-dinh-so-123-2020-nd-cp-ngay-19-thang-10-nam-2020-cua-chinh-phu-quy-dinh-ve-hoa-don-chung-tu--177581",
        "ma_van_ban": "123/2020/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định số 70/2025/NĐ-CP Sửa đổi, bổ sung một số điều của Nghị định số 123/2020/NĐ-CP ngày 19 tháng 10 năm 2020 của Chính phủ quy định về hóa đơn, chứng từ"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-123-2020-nd-cp-quy-dinh-ve-hoa-don-chung-tu--146457",
        "ma_van_ban": "123/2020/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định số 123/2020/NĐ-CP Quy định về hóa đơn, chứng từ"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-41-2022-nd-cp-sua-doi-bo-sung-mot-so-dieu-cua-nghi-dinh-so-123-2020-nd-cp-ngay-19-thang-10-nam-2020-cua-chinh-phu-quy-dinh-ve-hoa-don-chung-tu-va-nghi-dinh-so-15-2022-nd-cp-ngay-28-thang-01-nam-2022-cua-chinh-phu-quy-dinh-chinh-sach-mien-giam-thue-theo-nghi-quyet-so-43-2022-qh15-cua-quoc-hoi-ve-chinh-sach-tai-khoa-tien-te-ho-tro-chuong-trinh-phuc-hoi-va-phat-trien-kinh-te-xa-hoi--154800",
        "ma_van_ban": "123/2020/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định số 41/2022/NĐ-CP Sửa đổi, bổ sung một số điều của Nghị định số 123/2020/NĐ-CP ngày 19 tháng 10 năm 2020 của Chính phủ quy định về hóa đơn, chứng từ và Nghị định số 15/2022/NĐ-CP ngày 28 tháng 01 năm 2022 của Chính phủ quy định chính sách miễn, giảm thuế theo Nghị quyết số 43/2022/QH15 của Quốc hội về chính sách tài khóa, tiền tệ hỗ trợ Chương trình phục hồi và phát triển kinh tế - xã hội"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/thong-tu-so-32-2025-tt-btc-huong-dan-thuc-hien-mot-so-dieu-cua-luat-quan-ly-thue-ngay-13-thang-6-nam-2019-nghi-dinh-so-123-2020-nd-cp-ngay-19-thang-10-nam-2020-cua-chinh-phu-quy-dinh-ve-hoa-don-chung-tu-nghi-dinh-so-70-2025-nd-cp-ngay-20-thang-3-nam-2025-sua-doi-bo-sung-mot-so-dieu-cua-nghi-dinh-so-123-2020-nd-cp--178309",
        "ma_van_ban": "123/2020/NĐ-CP",
        "ten_van_ban_chuan": "Thông tư số 32/2025/TT-BTC Hướng dẫn thực hiện một số điều của Luật Quản lý thuế ngày 13 tháng 6 năm 2019, Nghị định số 123/2020/NĐ-CP ngày 19 tháng 10 năm 2020 của Chính phủ quy định về hóa đơn, chứng từ, Nghị định số 70/2025/NĐ-CP ngày 20 tháng 3 năm 2025 sửa đổi, bổ sung một số điều của Nghị định số 123/2020/NĐ-CP"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/thong-tu-so-78-2021-tt-btc-huong-dan-thuc-hien-mot-so-dieu-cua-luat-quan-ly-thue-ngay-13-thang-6-nam-2019-nghi-dinh-so-123-2020-nd-cp-ngay-19-thang-10-nam-2020-cua-chinh-phu-quy-dinh-ve-hoa-don-chung-tu--151084",
        "ma_van_ban": "123/2020/NĐ-CP",
        "ten_van_ban_chuan": "Thông tư số 78/2021/TT-BTC Hướng dẫn thực hiện một số điều của Luật Quản lý thuế ngày 13 tháng 6 năm 2019, Nghị định số 123/2020/NĐ-CP ngày 19 tháng 10 năm 2020 của Chính phủ quy định về hóa đơn, chứng từ"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/thong-tu-so-78-2021-tt-btc-huong-dan-thuc-hien-mot-so-dieu-cua-luat-quan-ly-thue-ngay-13-thang-6-nam-2019-nghi-dinh-so-123-2020-nd-cp-ngay-19-thang-10-nam-2020-cua-chinh-phu-quy-dinh-ve-hoa-don-chung-tu--151084",
        "ma_van_ban": "78/2021/TT-BTC",
        "ten_van_ban_chuan": "Thông tư 78/2021/TT-BTC Hướng dẫn thực hiện một số điều của Luật Quản lý thuế, Nghị định 123/2020/NĐ-CP về hóa đơn, chứng từ"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/thong-tu-so-132-2018-tt-btc-huong-dan-che-do-ke-toan-cho-doanh-nghiep-sieu-nho--134338",
        "ma_van_ban": "132/2018/TT-BTC",
        "ten_van_ban_chuan": "Thông tư 132/2018/TT-BTC Hướng dẫn Chế độ kế toán cho doanh nghiệp siêu nhỏ"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-thuong-mai-so-36-2005-qh11--26117",
        "ma_van_ban": "36/2005/QH11",
        "ten_van_ban_chuan": "Luật Thương mại số 36/2005/QH11"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/bo-luat-dan-su-so-91-2015-qh13--95942",
        "ma_van_ban": "91/2015/QH13",
        "ten_van_ban_chuan": "Bộ luật Dân sự số 91/2015/QH13"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-quyet-so-91-2015-qh13-ve-chuong-trinh-hoat-dong-giam-sat-cua-quoc-hoi-nam-2016--70827",
        "ma_van_ban": "91/2015/QH13",
        "ten_van_ban_chuan": "Nghị quyết số 91/2015/QH13 Về chương trình hoạt động giám sát của Quốc hội năm 2016"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-dau-thau-so-22-2023-qh15--166581",
        "ma_van_ban": "22/2023/QH15",
        "ten_van_ban_chuan": "Luật 22/2023/QH15 Đấu thầu"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-dau-tu-theo-phuong-thuc-doi-tac-cong-tu-so-64-2020-qh14--142882",
        "ma_van_ban": "64/2020/QH14",
        "ten_van_ban_chuan": "Luật 64/2020/QH14 Đầu tư theo phương thức đối tác công tư"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-quan-ly-thue-so-38-2019-qh14--136036",
        "ma_van_ban": "38/2019/QH14",
        "ten_van_ban_chuan": "Luật Quản lý thuế số 38/2019/QH14",
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/van-ban-hop-nhat-luat-quan-ly-thue-so-38-2019-qh14--a3756ec0-64b0-11f1-be70-8f2e758b5f81",
        "ma_van_ban": "38/2019/QH14",
        "ten_van_ban_chuan": "Văn bản hợp nhất Luật Quản lý thuế số 38/2019/QH14",
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-bao-hiem-xa-hoi-so-58-2014-qh13--46744",
        "ma_van_ban": "58/2014/QH13",
        "ten_van_ban_chuan": "Luật Bảo hiểm xã hội số 58/2014/QH13",
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-pha-sa-n-so-51-2014-qh13--36869",
        "ma_van_ban": "51/2014/QH13",
        "ten_van_ban_chuan": "Luật Phá sản số 51/2014/QH13"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-doanh-nghiep-so-68-2014-qh13--46751",
        "ma_van_ban": "68/2014/QH13",
        "ten_van_ban_chuan": "Luật Doanh nghiệp số 68/2014/QH13"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-quyet-so-78-2025-ubtvqh15-ve-viec-ghi-nhan-thoi-gian-dong-bao-hiem-xa-hoi-bat-buoc-cua-chu-ho-kinh-doanh-da-tham-gia-truoc-ngay-luat-bao-hiem-xa-hoi-so-41-2024-qh15-co-hieu-luc-thi-hanh--179965",
        "ma_van_ban": "41/2024/QH15",
        "ten_van_ban_chuan": "Nghị quyết số 78/2025/UBTVQH15 Về việc ghi nhận thời gian đóng bảo hiểm xã hội bắt buộc của chủ hộ kinh doanh đã tham gia trước ngày Luật Bảo hiểm xã hội số 41/2024/QH15 có hiệu lực thi hành"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-sua-doi-bo-sung-mot-so-dieu-cua-luat-so-huu-tri-tue-so-07-2022-qh15--157722",
        "ma_van_ban": "07/2022/QH15",
        "ten_van_ban_chuan": "Luật Sửa đổi, bổ sung một số điều của Luật Sở hữu trí tuệ số 07/2022/QH15",
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-ke-toan-so-88-2015-qh13--95924",
        "ma_van_ban": "88/2015/QH13",
        "ten_van_ban_chuan": "Luật Kế toán số 88/2015/QH13",
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-an-toan-ve-sinh-lao-dong-so-84-2015-qh13--70811",
        "ma_van_ban": "84/2015/QH13",
        "ten_van_ban_chuan": "Luật An toàn, vệ sinh lao động số 84/2015/QH13",
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-quang-cao-so-16-2012-qh13--27617",
        "ma_van_ban": "16/2012/QH13",
        "ten_van_ban_chuan": "Luật Quảng cáo số 16/2012/QH13"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-hai-quan-so-54-2014-qh13--36878",
        "ma_van_ban": "54/2014/QH13",
        "ten_van_ban_chuan": "LUẬT HẢI QUAN SỐ 54/2014/QH13"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-trong-tai-thuong-mai-so-54-2010-qh12--25700",
        "ma_van_ban": "54/2010/QH12",
        "ten_van_ban_chuan": "Luật Trọng tài thương mại số 54/2010/QH12"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-xay-dung-so-50-2014-qh13--36867",
        "ma_van_ban": "50/2014/QH13",
        "ten_van_ban_chuan": "Luật Xây dựng số 50/2014/QH13"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-sua-doi-bo-sung-mot-so-dieu-cua-luat-dat-dai-so-31-2024-qh15-luat-nha-o-so-27-2023-qh15-luat-kinh-doanh-bat-dong-san-so-29-2023-qh15-va-luat-cac-to-chuc-tin-dung-so-32-2024-qh15-so-43-2024-qh15--170509",
        "ma_van_ban": "31/2024/QH15",
        "ten_van_ban_chuan": "Luật Sửa đổi, bổ sung một số điều của Luật Đất đai số 31/2024/QH15, Luật Nhà ở số 27/2023/QH15, Luật Kinh doanh bất động sản số 29/2023/QH15 và Luật Các tổ chức tín dụng số 32/2024/QH15 số 43/2024/QH15"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-dat-dai-so-31-2024-qh15--177815",
        "ma_van_ban": "31/2024/QH15",
        "ten_van_ban_chuan": "Luật Đất đai số 31/2024/QH15"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-thue-tieu-thu-dac-biet-so-27-2008-qh12--12324",
        "ma_van_ban": "27/2008/QH12",
        "ten_van_ban_chuan": "Luật Thuế tiêu thụ đặc biệt số 27/2008/QH12"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-cong-doan-so-50-2024-qh15--172553",
        "ma_van_ban": "50/2024/QH15",
        "ten_van_ban_chuan": "Luật Công đoàn số 50/2024/QH15"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-85-2021-nd-cp-sua-doi-bo-sung-mot-so-dieu-cua-nghi-dinh-so-52-2013-nd-cp-ngay-16-thang-5-nam-2013-cua-chinh-phu-ve-thuong-mai-dien-tu-nghi-dinh-so-52-2013-nd-cp--150735",
        "ma_van_ban": "85/2021/NĐ-CP",
        "ten_van_ban_chuan": "Nghị định số 85/2021/NĐ-CP sửa đổi, bổ sung một số điều của Nghị định số 52/2013/NĐ-CP ngày 16 tháng 5 năm 2013 của Chính phủ về thương mại điện tử (Nghị định số 52/2013/NĐ-CP)."
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-chung-khoan-so-54-2019-qh14--139885",
        "ma_van_ban": "54/2019/QH14",
        "ten_van_ban_chuan": "Luật Chứng khoán số 54/2019/QH14"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-phi-va-le-phi-so-97-2015-qh13--96119",
        "ma_van_ban": "97/2015/QH13",
        "ten_van_ban_chuan": "Luật Phí và lệ phí số 97/2015/QH13"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-quyet-so-04-2007-qh12-ve-viec-dieu-chinh-chuong-trinh-xay-dung-luat-phap-lenh-nam-2007--25400",
        "ma_van_ban": "04/2007/QH12",
        "ten_van_ban_chuan": "Nghị quyết số 04/2007/QH12 Về việc điều chỉnh chương trình xây dựng luật, pháp lệnh năm 2007"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-an-toan-thuc-pham-so-55-2010-qh12--25495",
        "ma_van_ban": "55/2010/QH12",
        "ten_van_ban_chuan": "Luật An toàn thực phẩm số 55/2010/QH12"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-so-huu-tri-tue-so-50-2005-qh11--16748",
        "ma_van_ban": "50/2005/QH11",
        "ten_van_ban_chuan": "Luật Sở hữu trí tuệ số 50/2005/QH11",
    },
    {
        "url": "http://vbpl.vn/van-ban/chi-tiet/luat-sua-doi-bo-sung-mot-so-dieu-cua-luat-so-huu-tri-tue-so-36-2009-qh12--11716",
        "ma_van_ban": "36/2009/QH12",
        "ten_van_ban_chuan": "Luật Sửa đổi, bổ sung một số điều của Luật Sở hữu trí tuệ số 36/2009/QH12",
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/nghi-quyet-so-36-2009-qh12-ve-ke-hoach-phat-trien-kinh-te-xa-hoi-nam-2010--23823",
        "ma_van_ban": "36/2009/QH12",
        "ten_van_ban_chuan": "Nghị quyết số 36/2009/QH12 Về kế hoạch phát triển kinh tế - xã hội năm 2010"
    },
    {
        "url": "https://vbpl.vn/van-ban/chi-tiet/luat-sua-doi-bo-sung-mot-so-dieu-cua-luat-so-huu-tri-tue-so-07-2022-qh15--157722",
        "ma_van_ban": "07/2022/QH15",
        "ten_van_ban_chuan": "Luật Sửa đổi, bổ sung một số điều của Luật Sở hữu trí tuệ số 07/2022/QH15"
    }
]
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept-Language": "vi,en;q=0.9",
}

# Các class quy định "cấp độ ngữ cảnh" (Phần / Chương / Mục)
PROV_CONTEXT = {
    "prov-part": "phan",
    "prov-chapter": "chuong",
    "prov-section": "muc",
    "prov-subsection": "tieu_muc",
}


# ============================================================
# 2. TIỆN ÍCH
# ============================================================
def clean_text(text: str) -> str:
    """Làm sạch khoảng trắng, &nbsp;, \\r, \\t."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\r\t]", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def get_prov_class(p):
    """Trả về tên class bắt đầu bằng 'prov-' của thẻ <p>, hoặc None."""
    for c in (p.get("class") or []):
        if c.startswith("prov-"):
            return c
    return None


def find_content_div(soup):
    """Tìm khối chứa toàn văn. Ưu tiên div.preview-content."""
    div = soup.select_one("div.preview-content")
    if div:
        return div
    # Dự phòng: panel tab "Toàn văn"
    div = soup.select_one("#rc-tabs-0-panel-toan-van")
    if div:
        return div
    return None


def extract_dieu_label(text: str):
    """Lấy nhãn 'Điều X' (kể cả 'Điều 12a') ở đầu chuỗi."""
    m = re.match(r"^(Điều\s+\d+[a-zA-Z]*)", text)
    return m.group(1) if m else None


# ============================================================
# 3. LẤY HTML — requests trước, Playwright dự phòng tự động
# ============================================================
def _looks_populated(html: str) -> bool:
    """Kiểm tra HTML đã có nội dung điều khoản thật chưa."""
    soup = BeautifulSoup(html, "html.parser")
    cdiv = find_content_div(soup)
    if not cdiv:
        return False
    return cdiv.find("p", class_="prov-article") is not None


def fetch_with_requests(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text
    except Exception as e:
        print(f"    [requests] lỗi: {e}")
        return None


def fetch_with_playwright(url: str):
    """Render trang bằng Chromium (xử lý nội dung load bằng JavaScript)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("    [!] Chưa cài Playwright. Chạy:\n"
              "        pip install playwright && playwright install chromium")
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=HEADERS["User-Agent"],
                                    locale="vi-VN")
            page.goto(url, wait_until="networkidle", timeout=60000)
            # Chờ cho tới khi có ít nhất 1 thẻ Điều xuất hiện
            try:
                page.wait_for_selector("p.prov-article", timeout=20000)
            except Exception:
                page.wait_for_selector("div.preview-content", timeout=10000)
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        print(f"    [playwright] lỗi: {e}")
        return None


def get_html(url: str):
    """Lấy HTML đã có nội dung thật. requests -> nếu rỗng -> Playwright."""
    print("    -> Thử bằng requests...")
    html = fetch_with_requests(url)
    if html and _looks_populated(html):
        print("    -> requests OK (trang có sẵn nội dung).")
        return html

    print("    -> requests không thấy nội dung (trang render JS). "
          "Chuyển sang Playwright...")
    html = fetch_with_playwright(url)
    if html and _looks_populated(html):
        print("    -> Playwright OK.")
        return html

    print("    [!] Vẫn không lấy được nội dung điều khoản.")
    return html  # trả về để debug nếu cần


# ============================================================
# 4. BÓC TÁCH PARAGRAPH -> CHUNK THEO "ĐIỀU"
# ============================================================
def extract_paragraphs(content_div):
    """
    Trả về list (prov_class, text) theo đúng thứ tự xuất hiện.
    Loại bỏ trùng lặp liên tiếp (xử lý cặp <p id=...> và <p id=..._1>
    có nội dung giống hệt nhau do trình render sinh ra).
    """
    out, prev = [], None
    for p in content_div.find_all("p"):
        pc = get_prov_class(p)
        if pc is None:
            continue  # bỏ qua tiêu đề "QUỐC HỘI", "Căn cứ...", &nbsp;...
        text = clean_text(p.get_text(" ", strip=True))
        if not text:
            continue
        if (pc, text) == prev:
            continue
        out.append((pc, text))
        prev = (pc, text)
    return out


def build_chunks(paras, law_info):
    """Gom các đoạn thành chunk theo từng 'Điều', kèm metadata truy hồi."""
    chunks, ctx, cur = [], {}, None

    def snapshot_path():
        parts = [ctx[k] for k in ("phan", "chuong", "muc", "tieu_muc") if ctx.get(k)]
        return " > ".join(parts)

    def flush():
        nonlocal cur
        if cur and cur["_body"]:
            noi_dung = clean_text("\n".join(cur["_body"]))
            chunks.append({
                "dieu": cur["dieu"],
                "noi_dung": noi_dung,
                "metadata_truy_hoi":
                    f"{law_info['ma_van_ban']}|{law_info['ten_van_ban_chuan']}|{cur['dieu']}",
                "vi_tri": cur["vi_tri"],
            })
        cur = None

    for pc, text in paras:
        if pc in PROV_CONTEXT:
            key = PROV_CONTEXT[pc]
            if pc in ("prov-part", "prov-chapter"):
                flush()
            # Gộp "Chương I" + "QUY ĐỊNH CHUNG" thành một dòng nếu liền nhau
            if ctx.get("_last_key") == key and ctx.get(key):
                ctx[key] += " " + text
            else:
                ctx[key] = text
            ctx["_last_key"] = key
        elif pc == "prov-article":
            flush()
            label = extract_dieu_label(text) or text[:40]
            cur = {"dieu": label, "_body": [text], "vi_tri": snapshot_path()}
            ctx["_last_key"] = None
        else:  # prov-clause, prov-point, prov-content...
            ctx["_last_key"] = None
            if cur is None:  # nội dung trước Điều đầu tiên (lời nói đầu)
                cur = {"dieu": "Lời nói đầu", "_body": [], "vi_tri": snapshot_path()}
            cur["_body"].append(text)

    flush()
    return chunks


# ============================================================
# 5. XỬ LÝ 1 VĂN BẢN
# ============================================================
def process_law(law_info):
    print(f"[*] Đang cào: {law_info['ten_van_ban_chuan']}")
    html = get_html(law_info["url"])
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    cdiv = find_content_div(soup)
    if not cdiv:
        print("    [!] Không tìm thấy div.preview-content.")
        return []
    paras = extract_paragraphs(cdiv)
    chunks = build_chunks(paras, law_info)
    print(f"    -> Bóc tách thành công {len(chunks)} Điều.")
    return chunks


# ============================================================
# 6. MAIN
# ============================================================
def main():
    final_dataset = []
    for law in TARGET_LAWS:
        final_dataset.extend(process_law(law))
        time.sleep(2)

    out_file = "corpus_luat_sme.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)

    print(f"\n[+] Hoàn tất! Đã lưu {len(final_dataset)} chunks vào '{out_file}'")
    if final_dataset:
        print("\n--- Ví dụ chunk đầu tiên ---")
        print(json.dumps(final_dataset[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()