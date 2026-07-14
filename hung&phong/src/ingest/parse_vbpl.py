"""Parse văn bản từ dataset tmquan/vbpl-vn (full-text markdown phẳng từ vbpl.vn).

Khác th1nhng0 (HTML): vbpl-vn cho `markdown` là TEXT PHẲNG, các Điều nằm inline:
  "... Điều 1. Phạm vi điều chỉnh <nội dung> Điều 2. Đối tượng áp dụng 1. ..."

Tách Điều = tìm "Điều N." (DẤU CHẤM) + số TĂNG DẦN → loại được tham chiếu lạc số
("Điều 88 của Luật này" không có dấu chấm; "...tại Điều 88." lạc số bị gate chặn).
"""
from __future__ import annotations

import re
from typing import Any

from ingest.parse import DieuChunk, ParsedDoc

# "Điều N." / "Điều Na." theo sau là khoảng trắng → ứng viên đầu một Điều.
# Một số nguồn OCR sai dấu thành "Điêu N.".
# Một số nguồn vbpl/docx mất dấu chấm sau số Điều, ví dụ:
#   "Điều 29 Xử phạt hành vi..."
#   "Điều 38 Chậm đóng bảo hiểm..."
# Vì vậy nhận thêm dạng không dấu chấm, nhưng chặn các tham chiếu phổ biến như
# "Điều 4 Nghị định này", "Điều 51 Luật ...", "Điều 6 và ...".
_NO_PUNCT_STOP = (
    r"(?:của|này|nghị\s+định|luật|thông\s+tư|bộ\s+luật|pháp\s+lệnh|"
    r"quyết\s+định|khoản|điểm|và|hoặc|đến|tại|nêu|trên|số)\b|[,;\)]"
)
_RE_DIEU = re.compile(
    rf"(?:Điều|Điêu)\s+(\d+)([a-zđ]?)\s*(?:[\.\:\-–]\s+|(?!(?:{_NO_PUNCT_STOP}))(?=\w))",
    re.IGNORECASE,
)

# Khoảng nhảy tối đa được chấp nhận giữa hai Điều liên tiếp. Cho phép vượt qua vài
# Điều bị "miss" (format lệch nên regex không bắt) mà KHÔNG nuốt cả phần đuôi văn bản;
# đồng thời đủ nhỏ để loại tham chiếu lạc số nhảy xa ("Điều 169 của Bộ luật...").
_GAP_MAX = 3  # giảm từ 8 → bớt content-bleed (1 Điều hấp thụ tối đa 3 Điều miss, không phải 8). audit 2026-06-20


def split_dieus(text: str) -> list[DieuChunk]:
    """Tách markdown phẳng thành list Điều theo gate số tăng dần (chịu được gap).

    Gate cũ yêu cầu base == expected CỨNG → khi một Điều bị miss (vd Điều 38 format
    lệch), expected kẹt và toàn bộ Điều sau bị bỏ → nuốt vào Điều cuối (mất hàng trăm
    điều, vd Luật BHXH 41/2024 chỉ còn 37/141). Gate mới chấp nhận base tăng dần trong
    khoảng (last_base, last_base + _GAP_MAX] để vượt qua các Điều miss mà vẫn loại tham
    chiếu lạc số (nhảy xa hoặc lùi).
    """
    cands = []
    for m in _RE_DIEU.finditer(text):
        base = int(m.group(1))
        suffix = m.group(2) or ""
        cands.append((m.start(), m.end(), base, suffix))

    bounds: list[tuple[int, int, int, str]] = []
    last_base = 0
    for (s, e, base, suffix) in cands:
        if not bounds:
            if base <= 2:                            # khởi đầu từ Điều 1 hoặc 2
                bounds.append((s, e, base, suffix))
                last_base = base
            continue
        if base == last_base + 1:                    # liên tục — lý tưởng
            bounds.append((s, e, base, suffix))
            last_base = base
        elif suffix and base == last_base:           # 112a ngay sau 112
            bounds.append((s, e, base, suffix))
        elif last_base < base <= last_base + _GAP_MAX:  # vượt vài Điều bị miss
            bounds.append((s, e, base, suffix))
            last_base = base
        # còn lại: tham chiếu lạc số (nhảy xa hoặc lùi) → bỏ

    dieus: list[DieuChunk] = []
    for i, (s, e, base, suffix) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
        body = text[e:end].strip()
        if len(body) < 20:
            continue
        # tiêu đề ~ tới khoản "1." đầu tiên hoặc 100 ký tự đầu
        mt = re.search(r"\s\d+\.\s", body)
        tieu_de = (body[: mt.start()] if mt else body[:100]).strip().rstrip(".")
        dieus.append(DieuChunk(dieu_so=base, dieu_tieu_de=tieu_de[:200], text=body))
    return dieus


def _first(row: dict, key: str) -> str:
    v = row.get(key)
    if isinstance(v, (list, tuple)):
        v = v[0] if v else ""
    return str(v).strip() if v is not None else ""


def parse_vbpl_row(row: dict[str, Any]) -> ParsedDoc | None:
    """Normalize 1 record vbpl-vn → ParsedDoc (kèm provenance source_url)."""
    md = row.get("markdown") or ""
    if not md or len(md) < 200:
        return None

    doc = ParsedDoc(
        doc_id=_first(row, "doc_name") or _first(row, "doc_number"),
        so_ky_hieu=_first(row, "doc_number"),
        loai_van_ban=_first(row, "legal_type"),
        co_quan_ban_hanh=_first(row, "issuing_authority"),
        ngay_ban_hanh=_first(row, "issue_date"),
        ngay_hieu_luc="",
        tinh_trang_hieu_luc="",                       # vbpl-vn không có trường này
        linh_vuc=_first(row, "legal_area"),
        title=_first(row, "title"),
    )
    doc.source_url = _first(row, "source_url")
    doc.nguon = _first(row, "source") or "vbpl.vn"
    doc.nam = _first(row, "year")
    doc.scope = _first(row, "scope")

    doc.dieus = split_dieus(md)
    if not doc.dieus:
        # văn bản không có cấu trúc "Điều N" → giữ TOÀN VĂN (bỏ truncate [:4000] gây mất data).
        # chunk.py sẽ tự cắt theo cỡ; _trim_monster cắt phần đuôi phi-quy-phạm.
        doc.dieus = [DieuChunk(dieu_so=0, dieu_tieu_de="Toàn văn", text=md)]
    return doc
