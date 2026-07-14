"""Nạp các văn bản BỔ SUNG THỦ CÔNG (.docx) mà nguồn tmquan/vbpl-vn không có/sai.

Lý do: vài VB còn hiệu lực 2026 không có toàn văn trong tmquan (vd 125/2020 tmquan chỉ
có công văn đính chính; 122/2020, 20/2026 không có record). Tải bản .docx chính thống từ
thuvienphapluat → parse cùng parser tmquan → nhập corpus/Qdrant.

File .docx đặt ở data/manual_vbpl/. Mỗi VB khai báo metadata trong MANUAL.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

from ingest.parse import ParsedDoc
from ingest.parse_vbpl import split_dieus

ROOT = Path(__file__).resolve().parents[1]
MANUAL_DIR = ROOT / "data" / "manual_vbpl"

# so_ky_hieu -> metadata. title = TRÍCH YẾU (không gồm loại — _law_name sẽ ghép loại).
MANUAL: dict[str, dict] = {
    "125/2020/NĐ-CP": {
        "loai": "Nghị định",
        "title": "Quy định xử phạt vi phạm hành chính về thuế, hóa đơn",
        "nam": "2020",
        "co_quan": "Chính phủ",
        "file": "125_2020_ND-CP.docx",
    },
    "122/2020/NĐ-CP": {
        "loai": "Nghị định",
        "title": ("Quy định về phối hợp, liên thông thủ tục đăng ký thành lập doanh nghiệp, "
                  "chi nhánh, văn phòng đại diện, khai trình việc sử dụng lao động, cấp mã số "
                  "đơn vị tham gia bảo hiểm xã hội, đăng ký sử dụng hóa đơn của doanh nghiệp"),
        "nam": "2020",
        "co_quan": "Chính phủ",
        "file": "122_2020_ND-CP.docx",
    },
    "20/2026/TT-BTC": {
        "loai": "Thông tư",
        "title": ("Hướng dẫn Luật Thuế thu nhập doanh nghiệp và Nghị định số 320/2025/NĐ-CP "
                  "quy định chi tiết một số điều của Luật Thuế thu nhập doanh nghiệp"),
        "nam": "2026",
        "co_quan": "Bộ Tài chính",
        "file": "20_2026_TT-BTC.docx",
    },
    "135/2020/NĐ-CP": {
        "loai": "Nghị định",
        "title": "Quy định về tuổi nghỉ hưu",
        "nam": "2020",
        "co_quan": "Chính phủ",
        "file": "135_2020_ND-CP.docx",
    },
    "157/2025/NĐ-CP": {
        "loai": "Nghị định",
        "title": ("Quy định chi tiết và biện pháp thi hành một số điều của Luật Bảo hiểm xã hội "
                  "về bảo hiểm xã hội bắt buộc đối với quân nhân, công an nhân dân và người làm "
                  "công tác cơ yếu hưởng lương như đối với quân nhân"),
        "nam": "2025",
        "co_quan": "Chính phủ",
        "file": "157_2025_ND-CP.docx",
    },
    "81/2018/NĐ-CP": {
        "loai": "Nghị định",
        "title": "Quy định chi tiết Luật Thương mại về hoạt động xúc tiến thương mại",
        "nam": "2018",
        "co_quan": "Chính phủ",
        "file": "81_2018_ND-CP.docx",
    },
    "07/2022/NĐ-CP": {
        "loai": "Nghị định",
        "title": ("Sửa đổi, bổ sung một số điều của các nghị định về xử phạt vi phạm hành chính "
                  "trong lĩnh vực sở hữu công nghiệp; tiêu chuẩn, đo lường và chất lượng sản phẩm, "
                  "hàng hóa; hoạt động khoa học và công nghệ, chuyển giao công nghệ; năng lượng nguyên tử"),
        "nam": "2022",
        "co_quan": "Chính phủ",
        "file": "07_2022.docx",
    },
    "07/2022/QH15": {
        "loai": "Luật",
        "title": "Sửa đổi, bổ sung một số điều của Luật Sở hữu trí tuệ",
        "nam": "2022",
        "co_quan": "Quốc hội",
        "file": "07_2022_QH15.docx",
    },
    "68/2026/NĐ-CP": {
        "loai": "Nghị định",
        "title": "Quy định về chính sách thuế và quản lý thuế đối với hộ kinh doanh, cá nhân kinh doanh",
        "nam": "2026",
        "co_quan": "Chính phủ",
        "file": "68_2026_ND-CP.docx",
    },
    "141/2026/NĐ-CP": {
        "loai": "Nghị định",
        "title": ("Sửa đổi, bổ sung một số điều của Nghị định số 68/2026/NĐ-CP quy định về chính "
                  "sách thuế và quản lý thuế đối với hộ kinh doanh, cá nhân kinh doanh"),
        "nam": "2026",
        "co_quan": "Chính phủ",
        "file": "141_2026_ND-CP.docx",
    },
    "132/2026/NĐ-CP": {
        "loai": "Nghị định",
        "title": ("Sửa đổi, bổ sung một số điều của Nghị định số 41/2018/NĐ-CP quy định xử phạt "
                  "vi phạm hành chính trong lĩnh vực kế toán, kiểm toán độc lập"),
        "nam": "2026",
        "co_quan": "Chính phủ",
        "file": "132_2026_ND-CP.docx",
    },
}

_RE_WT = re.compile(r"<w:t[^>]*>(.*?)</w:t>", re.DOTALL)
_RE_TAG = re.compile(r"<[^>]+>")


def read_docx(path: Path) -> str:
    """Trích text thuần từ .docx (mỗi <w:p> = 1 dòng)."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    lines: list[str] = []
    for para in xml.split("</w:p>"):
        texts = _RE_WT.findall(para)
        line = _RE_TAG.sub("", "".join(texts)).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def load_manual_docs() -> list[ParsedDoc]:
    """Đọc tất cả VB thủ công trong MANUAL → list[ParsedDoc] (đã tách Điều)."""
    docs: list[ParsedDoc] = []
    for sk, meta in MANUAL.items():
        path = MANUAL_DIR / meta["file"]
        if not path.exists():
            continue
        text = read_docx(path)
        dieus = split_dieus(text)
        if not dieus:
            continue
        doc = ParsedDoc(
            doc_id=sk,
            so_ky_hieu=sk,
            loai_van_ban=meta["loai"],
            co_quan_ban_hanh=meta["co_quan"],
            ngay_ban_hanh="",
            ngay_hieu_luc="",
            tinh_trang_hieu_luc="Còn hiệu lực",
            linh_vuc="",
            title=meta["title"],
            dieus=dieus,
        )
        doc.source_url = ""
        doc.nguon = "manual (thuvienphapluat .docx)"
        doc.nam = meta["nam"]
        doc.scope = "trung_uong"
        docs.append(doc)
    return docs


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for d in load_manual_docs():
        nums = [x.dieu_so for x in d.dieus]
        print(f"{d.so_ky_hieu}: {len(d.dieus)} điều (max {max(nums)}) — {d.loai_van_ban} {d.title[:40]}")
