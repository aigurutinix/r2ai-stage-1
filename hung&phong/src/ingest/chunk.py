"""Chunk văn bản pháp luật theo CẤU TRÚC, không theo đếm token.

Triết lý (RAG pháp lý):
  - Đơn vị semantic tự nhiên của VBPL: Điều → Khoản → Điểm. Chunk phải bám cấu trúc
    này, KHÔNG cắt cứng theo số ký tự.
  - 1 Điều ngắn → 1 chunk (một ý pháp lý trọn vẹn). Giữ header "Điều N. <tiêu đề>".
  - Điều dài → GỘP các Khoản trọn vẹn thành chunk cỡ vừa (mỗi chunk = một/vài khoản
    đầy đủ, không bao giờ cắt ngang một khoản). Prepend header để giữ ngữ cảnh.
  - Khoản đơn lẻ quá dài → tách theo Điểm (a, b, c...). Điểm vẫn quá dài (hiếm) →
    cắt MỀM theo ranh giới câu, không cắt giữa từ/giữa câu.
  - VÙNG ĐÈ: 2 chunk liên tiếp trong cùng một Điều chừa overlap (câu cuối của chunk
    trước lặp sang đầu chunk sau) để không mất ngữ cảnh ở ranh giới.

  Lưu ý: text sau parse đã mất ký tự xuống dòng (99.9% điều là 1 dòng) nên việc nhận
  Khoản/Điểm phải theo CHUỖI SỐ/CHỮ TĂNG DẦN inline, không dựa vào đầu dòng (^).
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass

from ingest.parse import DieuChunk, ParsedDoc
from backend.textnorm import normalize_vn

logger = logging.getLogger(__name__)

# Cỡ chunk theo Ý NGHĨA, không phải giới hạn cứng của model (model chịu 8192 token).
KEEP_WHOLE_CHARS = 4000   # Điều ngắn hơn mức này → giữ nguyên 1 chunk (giảm 2500→4000: bớt tách 14%→ít hơn). audit 2026-06-20
TARGET_CHARS = 1800       # cỡ mong muốn khi gộp các khoản lại
HARD_MAX_CHARS = 3200     # 1 khoản/điểm dài hơn mức này mới buộc cắt mềm theo câu
MIN_CHARS = 30

# Điều > ngưỡng này gần như chắc chắn do parser NUỐT phụ lục/biểu mẫu/văn bản kế tiếp
# vào điều cuối ("Trách nhiệm thi hành", "Tổ chức thực hiện"). Cắt phần đuôi phi quy phạm.
_MONSTER_CHARS = 12_000   # giảm 50k→12k: cắt sớm hơn các điều cuối nuốt phụ lục/văn bản kế. audit 2026-06-20
_RE_APPENDIX = re.compile(
    r"(PHỤ\s*LỤC|Phụ\s*lục\s+(?:số|[IVX0-9])|MẪU\s*(?:SỐ|số)|Mẫu\s+số\b|BIỂU\s*(?:SỐ|MẪU)|Biểu\s+số\b"
    r"|Nơi\s+nhận\s*:|KT\.\s*BỘ\s+TRƯỞNG|TM\.\s*CHÍNH\s+PHỦ|KT\.\s*THỦ\s+TƯỚNG|THỦ\s+TƯỚNG\s+CHÍNH\s+PHỦ)"
)

_RE_KHOAN_CAND = re.compile(r"(\d{1,2})[\.\)]\s+")
# Thứ tự điểm trong VBPL tiếng Việt (bỏ f/j/w; có đ, ư).
_DIEM_ORDER = "a b c d đ e g h i k l m n o p q r s t u ư v x y".split()
_RE_DIEM_CAND = re.compile(r"([a-zđư])\)\s+")
_SENT_BOUNDARY = re.compile(r"[\.;:?!]\s")


@dataclass
class Chunk:
    """Đơn vị payload + text để embed."""

    chunk_id: str
    text: str
    payload: dict


# ───────────────────────── tách theo cấu trúc ─────────────────────────

def _split_by_sequence(text: str, cand_re: re.Pattern, order) -> list[tuple[str | int, str]]:
    """Tách theo chuỗi nhãn TĂNG DẦN (khoản 1,2,3.. hoặc điểm a,b,c..).

    Chỉ nhận ứng viên nối tiếp đúng thứ tự để tránh nhầm số/chữ lẻ trong câu.
    Trả [(nhãn, text)]; phần preamble trước nhãn đầu mang nhãn None.
    """
    cands = [(m.start(), m.group(1)) for m in cand_re.finditer(text)]
    seq: list[tuple[int, str | int]] = []
    idx = 0  # vị trí mong đợi trong `order`
    for pos, lab in cands:
        if idx < len(order) and lab == order[idx]:
            if pos == 0 or text[pos - 1] in " \t\n.;:)":
                seq.append((pos, lab))
                idx += 1
    if len(seq) < 2:
        return [(None, text.strip())]

    out: list[tuple[str | int, str]] = []
    if seq[0][0] > 0:
        pre = text[: seq[0][0]].strip()
        if pre:
            out.append((None, pre))
    for i, (pos, lab) in enumerate(seq):
        end = seq[i + 1][0] if i + 1 < len(seq) else len(text)
        block = text[pos:end].strip()
        if block:
            out.append((lab, block))
    return out


def _split_khoan(text: str):
    return _split_by_sequence(text, _RE_KHOAN_CAND, [str(i) for i in range(1, 100)])


def _split_diem(text: str):
    return _split_by_sequence(text, _RE_DIEM_CAND, _DIEM_ORDER)


def _sentence_split(text: str, max_chars: int) -> list[str]:
    """Cắt MỀM theo ranh giới câu (không cắt giữa từ). Dùng khi 1 điểm quá dài."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if n - i <= max_chars:
            pieces.append(text[i:].strip())
            break
        window = text[i : i + max_chars]
        cut = -1
        for m in _SENT_BOUNDARY.finditer(window):
            cut = m.end()
        if cut < max_chars // 2:
            sp = window.rfind(" ")
            cut = sp + 1 if sp > 0 else max_chars
        seg = text[i : i + cut].strip()
        if seg:
            pieces.append(seg)
        i += cut
    return [p for p in pieces if p]


def _explode_oversized(seg: str) -> list[str]:
    """Khoản đơn > HARD_MAX → tách theo Điểm; điểm vẫn dài → cắt mềm theo câu."""
    diems = _split_diem(seg)
    if len(diems) <= 1:
        return _sentence_split(seg, HARD_MAX_CHARS)
    out: list[str] = []
    for _lab, dtext in diems:
        if len(dtext) <= HARD_MAX_CHARS:
            out.append(dtext)
        else:
            out.extend(_sentence_split(dtext, HARD_MAX_CHARS))
    return out


def _pack(segments: list[tuple[str | int | None, str]]) -> list[tuple[str | int | None, str]]:
    """Gộp các khoản trọn vẹn thành body cỡ ~TARGET. Trả [(nhãn_đầu, body)]."""
    packed: list[tuple[str | int | None, str]] = []
    cur: list[str] = []
    cur_label: str | int | None = None
    cur_len = 0
    for lab, seg in segments:
        if len(seg) > HARD_MAX_CHARS:
            if cur:
                packed.append((cur_label, "\n".join(cur)))
                cur, cur_label, cur_len = [], None, 0
            for sub in _explode_oversized(seg):
                packed.append((lab, sub))
            continue
        if cur and cur_len + len(seg) + 1 > TARGET_CHARS:
            packed.append((cur_label, "\n".join(cur)))
            cur, cur_label, cur_len = [], None, 0
        if not cur:
            cur_label = lab
        cur.append(seg)
        cur_len += len(seg) + 1
    if cur:
        packed.append((cur_label, "\n".join(cur)))
    return packed


# ───────────────────────── build chunk ─────────────────────────

def chunk_document(doc: ParsedDoc) -> list[Chunk]:
    base_payload = {
        "doc_id": doc.doc_id,
        "so_ky_hieu": doc.so_ky_hieu,
        "loai_van_ban": doc.loai_van_ban,
        "co_quan_ban_hanh": doc.co_quan_ban_hanh,
        "ngay_ban_hanh": doc.ngay_ban_hanh,
        "ngay_hieu_luc": doc.ngay_hieu_luc,
        "tinh_trang_hieu_luc": doc.tinh_trang_hieu_luc,
        "linh_vuc": doc.linh_vuc,
        "title": doc.title,
    }
    if doc.source_url:
        base_payload["source_url"] = doc.source_url
    if doc.nguon:
        base_payload["nguon"] = doc.nguon
    if doc.nam:
        base_payload["nam"] = doc.nam

    out: list[Chunk] = []
    for dieu in doc.dieus:
        out.extend(_chunk_dieu(dieu, base_payload))
    return out


def _trim_monster(text: str) -> str:
    if len(text) <= _MONSTER_CHARS:
        return text
    m = _RE_APPENDIX.search(text, 200)
    if m and m.start() > 200:
        text = text[: m.start()].rstrip()
    if len(text) > _MONSTER_CHARS:
        text = text[:_MONSTER_CHARS]
    return text


def _chunk_dieu(dieu: DieuChunk, base: dict) -> list[Chunk]:
    """Chunk 1 Điều. Tách tiêu đề/chapeau khỏi thân DỰA TRÊN cấu trúc text (mốc Khoản
    đầu), KHÔNG dùng dieu_tieu_de của parser — tiêu đề parser hay bị cắt cứng 100 ký tự
    giữa từ ("...và họ"), prepend lại sẽ chèn "\\n" vào giữa từ ("họ⏎at động").

    - Có khoản: header = "Điều N. <tiêu đề + chapeau trước khoản 1>", thân = các khoản.
    - Không khoản (điều 1 đoạn): header = "Điều N.", tiêu đề nằm tự nhiên đầu thân (không
      lặp, không vỡ chữ).
    """
    raw = normalize_vn(_trim_monster(dieu.text).strip())
    if len(raw) < MIN_CHARS:
        return []
    num = dieu.dieu_so
    parts = _split_khoan(raw)  # [(nhãn, text)]; preamble (trước khoản 1) mang nhãn None

    if len(parts) >= 2 and parts[0][0] is None:
        title_region = parts[0][1].strip()
        khoan_parts: list | None = parts[1:]
    elif len(parts) >= 2:
        title_region = ""
        khoan_parts = parts
    else:  # không có khoản
        title_region = ""
        khoan_parts = None

    if num > 0:
        header = f"Điều {num}. {title_region}".strip() if title_region else f"Điều {num}."
    else:
        header = title_region

    # Cả điều đủ ngắn → 1 chunk (một ý pháp lý trọn vẹn)
    if khoan_parts is None:
        full = f"{header}\n{raw}".strip() if header else raw
    else:
        full = (f"{header}\n" + "\n".join(p[1] for p in khoan_parts)).strip()
    if len(full) <= KEEP_WHOLE_CHARS:
        return [_make_chunk(base, dieu, khoan_so=None, text=full)]

    # Điều dài → gộp theo Khoản TRỌN VẸN, cắt đúng ranh giới khoản, KHÔNG vùng đè:
    # BTC chấm theo (số ký hiệu|Điều) → bất kỳ chunk nào của Điều lọt top là đủ tính,
    # overlap chỉ lặp số thứ tự lửng lơ + phình index. Header giữ ngữ cảnh cho mỗi mẩu.
    segments = [(None, raw)] if khoan_parts is None else khoan_parts
    out: list[Chunk] = []
    for lab, pbody in _pack(segments):
        text = f"{header}\n{pbody.strip()}".strip() if header else pbody.strip()
        if len(text) >= MIN_CHARS:
            khoan_so = int(lab) if isinstance(lab, str) and lab.isdigit() else None
            out.append(_make_chunk(base, dieu, khoan_so=khoan_so, text=text))
    return out


def _make_chunk(base: dict, dieu: DieuChunk, khoan_so: int | None, text: str) -> Chunk:
    payload = dict(base)
    payload["dieu_so"] = dieu.dieu_so
    payload["dieu_tieu_de"] = dieu.dieu_tieu_de
    if khoan_so is not None:
        payload["khoan_so"] = khoan_so
    payload["text"] = text
    payload["char_len"] = len(text)

    th = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    key = f"{base.get('doc_id', '')}::dieu={dieu.dieu_so}::khoan={khoan_so}::{th}"
    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, key))
    return Chunk(chunk_id=chunk_id, text=text, payload=payload)
