"""Parse HTML/text VBPL → cấu trúc Phần/Chương/Mục/Điều/Khoản/Điểm.

Dataset `th1nhng0/vietnamese-legal-documents` có nhiều biến thể trường:
  - `text` / `content` / `noi_dung` (HTML hoặc plain text)
  - Metadata: `so_ky_hieu`, `loai_van_ban`, `co_quan_ban_hanh`,
    `ngay_ban_hanh`, `ngay_hieu_luc`, `tinh_trang_hieu_luc`, `linh_vuc`,
    `title` / `tieu_de`, `doc_id` / `id`.

Hàm `parse_document` chấp nhận một dict raw → trả về cấu trúc đã normalize.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------- Regex parser cho cấu trúc VBPL ----------

# "Điều 36." / "Điều 36 ." / "Điều 36:" / "Điều 36 -"
RE_DIEU = re.compile(
    r"^\s*Điều\s+(\d+)\s*[\.\:\-–]?\s*(.*)$",
    re.MULTILINE,
)

# "1." / "1)" ở đầu dòng → Khoản
RE_KHOAN = re.compile(r"^\s*(\d{1,2})[\.\)]\s+", re.MULTILINE)

# "a)" / "b." → Điểm
RE_DIEM = re.compile(r"^\s*([a-zA-Z])[\.\)]\s+", re.MULTILINE)


# ---------- DTO ----------


@dataclass
class DieuChunk:
    """Một Điều luật — đơn vị semantic chính để chunk + embed."""

    dieu_so: int
    dieu_tieu_de: str
    text: str
    char_len: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_len = len(self.text)


@dataclass
class ParsedDoc:
    """Văn bản đã parse: metadata + list các Điều."""

    doc_id: str
    so_ky_hieu: str
    loai_van_ban: str
    co_quan_ban_hanh: str
    ngay_ban_hanh: str
    ngay_hieu_luc: str
    tinh_trang_hieu_luc: str
    linh_vuc: str
    title: str
    dieus: list[DieuChunk] = field(default_factory=list)
    # Provenance (dùng cho nguồn vbpl-vn; th1nhng0 để rỗng)
    source_url: str = ""
    nguon: str = ""
    nam: str = ""
    scope: str = ""


# ---------- Helpers ----------


def _first_nonempty(d: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return default


def _html_to_text(raw: str) -> str:
    """Chuyển HTML → plain text, giữ line breaks."""
    if not raw:
        return ""
    if "<" not in raw and ">" not in raw:
        return raw
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(["p", "div", "li", "tr"]):
        block.append("\n")
    text = soup.get_text()
    text = re.sub(r" ", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_dieus(body: str) -> list[DieuChunk]:
    """Tách body văn bản thành list Điều dựa trên regex `Điều N.`."""
    matches = list(RE_DIEU.finditer(body))
    if not matches:
        return []

    dieus: list[DieuChunk] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        block = body[start:end].strip()

        dieu_so = int(m.group(1))
        tieu_de = m.group(2).strip().rstrip(".").strip()
        # Loại bỏ dòng "Điều N. tiêu_đề" khỏi nội dung — giữ lại Khoản
        first_nl = block.find("\n")
        text = block[first_nl + 1 :].strip() if first_nl != -1 else block

        dieus.append(
            DieuChunk(
                dieu_so=dieu_so,
                dieu_tieu_de=tieu_de,
                text=text,
            )
        )
    return dieus


# ---------- Entry point ----------


def parse_document(raw: dict[str, Any]) -> ParsedDoc | None:
    """Normalize một record dataset → ParsedDoc. Trả về None nếu rỗng.

    Hỗ trợ cả schema English (`th1nhng0/vietnamese-legal-documents`) và
    biến thể Vietnamese của các dataset khác.
    """
    body_raw = _first_nonempty(raw, "content", "text", "noi_dung", "html")
    body = _html_to_text(body_raw)

    if not body or len(body) < 50:
        return None

    doc = ParsedDoc(
        doc_id=_first_nonempty(raw, "doc_id", "id", "vbpl_id", default=""),
        so_ky_hieu=_first_nonempty(raw, "document_number", "so_ky_hieu", "ky_hieu", "so_hieu"),
        loai_van_ban=_first_nonempty(raw, "legal_type", "loai_van_ban", "loai_vb", "loai"),
        co_quan_ban_hanh=_first_nonempty(raw, "issuing_authority", "co_quan_ban_hanh", "co_quan"),
        ngay_ban_hanh=_first_nonempty(raw, "issuance_date", "ngay_ban_hanh", "ngay_bh"),
        ngay_hieu_luc=_first_nonempty(
            raw, "ngay_co_hieu_luc", "effect_date", "ngay_hieu_luc", "ngay_hl"
        ),
        tinh_trang_hieu_luc=_first_nonempty(
            raw, "effect_status", "tinh_trang_hieu_luc", "tinh_trang", "trang_thai"
        ),
        linh_vuc=_first_nonempty(raw, "legal_sectors", "linh_vuc", "field"),
        title=_first_nonempty(raw, "title", "tieu_de", "ten_van_ban", "ten"),
    )

    if not doc.doc_id:
        # Fallback id từ số ký hiệu (nếu có) hoặc hash
        doc.doc_id = doc.so_ky_hieu or f"hash_{abs(hash(body)) % (10**12)}"

    doc.dieus = _split_into_dieus(body)
    if not doc.dieus:
        # Văn bản ngắn (Quyết định 1 trang) → giữ nguyên làm 1 "Điều 0"
        doc.dieus = [DieuChunk(dieu_so=0, dieu_tieu_de="", text=body)]

    return doc
