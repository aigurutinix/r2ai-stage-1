#!/usr/bin/env python3
"""
Parser & chunker cho văn bản pháp luật Việt Nam (vbpl_dataset/full_text/).

Chunk tại cấp Điều/Article. Mỗi chunk mang đầy đủ metadata phân cấp
(Phần → Chương → Mục → Điều) và danh sách khoản/điểm cấu trúc.

Parse types:
  dieu     — Văn bản tiếng Việt có "Điều X" (>32K docs)
  article  — Bản dịch tiếng Anh có "Article X" (~229 docs)
  numbered — Không có Điều/Article, dùng mục số/La Mã (~7.6K docs)
  empty    — File rỗng hoặc quá ngắn
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Paywall detection ────────────────────────────────────────────────────────

_PAYWALL_MARKERS = (
    "Bạn phải đăng nhập hoặc đăng ký",
    "Mọi chi tiết xin liên hệ: ĐT:",
    "You need to login or register",
)

_MAX_TITLE_CHARS = 150   # inline text longer than this is content, not a title


def _is_paywall(line: str) -> bool:
    s = line.strip()
    if s == "...":
        return True
    return any(m in s for m in _PAYWALL_MARKERS)


# ── Regex patterns ───────────────────────────────────────────────────────────

# Article markers
_RE_DIEU = re.compile(r"^Điều\s+(\d+\w*)[\.\:]?\s*(.*)", re.UNICODE)
_RE_ARTICLE = re.compile(r"^Article\s+(\d+\w*)[\.\:\s]*(.*)", re.IGNORECASE)

# Hierarchy markers — Vietnamese
_RE_PHAN = re.compile(r"^PHẦN\s+(.*)", re.UNICODE)
_RE_CHUONG_UPPER = re.compile(r"^CHƯƠNG\s+(.*)", re.UNICODE)
_RE_CHUONG_LOWER = re.compile(r"^Chương\s+(\w+)[:\.]?\s*(.*)", re.UNICODE)
_RE_MUC = re.compile(r"^MỤC\s+(.*)", re.UNICODE)
_RE_TIET = re.compile(r"^TIẾT\s+(.*)", re.UNICODE)

# Hierarchy markers — English
_RE_CHAPTER_EN = re.compile(r"^Chapter\s+(\w+)[:\.]?\s*(.*)", re.IGNORECASE)
_RE_SECTION_EN = re.compile(r"^Section\s+(\w+)[:\.]?\s*(.*)", re.IGNORECASE)

# Khoản / Điểm within a Điều
_RE_KHOAN_DASH = re.compile(r"^(\d+)[-]\s+(.*)", re.UNICODE)
_RE_KHOAN_DOT = re.compile(r"^(\d+)[.]\s+(.*)", re.UNICODE)
_RE_DIEM = re.compile(r"^([a-zđ])\)\s+(.*)", re.UNICODE)

# Roman numeral section (for numbered-type docs)
_RE_ROMAN = re.compile(
    r"^(I{1,3}|IV|VI{0,3}|IX|X{1,3}|XI{0,3}|XIV|XV)[.\:]\s+(.*)", re.UNICODE
)


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class Diem:
    label: str   # "a", "b", "đ"
    content: str


@dataclass
class Khoan:
    so: Optional[int]   # None = unnumbered body
    content: str
    diem: list[Diem] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str            # "{doc_type}/{file_id}#dieu_{N}"
    article_number: str      # "1", "72", "3a"
    article_title: str
    path: dict               # phan/chuong/muc with index
    content: list[str]       # lines of this article (split on \n, empty lines excluded)
    khoan: list[Khoan]
    char_count: int
    citation_keys: list[str]
    prev_article: Optional[str]
    next_article: Optional[str]


@dataclass
class ParseWarning:
    level: str    # "warn" | "error"
    message: str


@dataclass
class ParseResult:
    doc_id: str
    doc_type: str
    file_path: str
    parse_type: str    # "dieu" | "article" | "numbered" | "empty"
    lang: str          # "vi" | "en" | "mixed"
    total_chunks: int
    paywall_lines: int
    warnings: list[ParseWarning]
    chunks: list[Chunk]

    def to_dict(self) -> dict:
        """Serialise (excludes chunk content for summary use)."""
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "file_path": self.file_path,
            "parse_type": self.parse_type,
            "lang": self.lang,
            "total_chunks": self.total_chunks,
            "paywall_lines": self.paywall_lines,
            "warnings": [{"level": w.level, "message": w.message} for w in self.warnings],
        }


# ── Internal helpers ─────────────────────────────────────────────────────────


def _next_nonempty(lines: list[str], start: int) -> tuple[int, str]:
    """Return (index, stripped_line) of next non-empty, non-paywall line."""
    for j in range(start, len(lines)):
        s = lines[j].strip()
        if s and not _is_paywall(s):
            return j, s
    return len(lines), ""


def _parse_khoan(content_lines: list[str]) -> list[Khoan]:
    """Detect khoản/điểm structure within article body lines."""
    khoans: list[Khoan] = []
    current: Optional[Khoan] = None

    for raw in content_lines:
        line = raw.strip()
        if not line:
            continue

        m = _RE_KHOAN_DASH.match(line) or _RE_KHOAN_DOT.match(line)
        if m:
            if current:
                khoans.append(current)
            current = Khoan(so=int(m.group(1)), content=m.group(2).strip())
            continue

        m = _RE_DIEM.match(line)
        if m:
            diem = Diem(label=m.group(1), content=m.group(2).strip())
            if current is None:
                current = Khoan(so=None, content="")
            current.diem.append(diem)
            continue

        # Continuation text
        if current is not None:
            current.content = (current.content + " " + line).strip()

    if current:
        khoans.append(current)
    return khoans


def _make_citation_keys(article_number: str, khoans: list[Khoan]) -> list[str]:
    keys = [f"Điều {article_number}"]
    for k in khoans:
        if k.so is not None:
            keys.append(f"khoản {k.so} Điều {article_number}")
            for d in k.diem:
                keys.append(f"điểm {d.label} khoản {k.so} Điều {article_number}")
    return keys


def _make_path(
    phan: Optional[str] = None, phan_i: int = 0,
    chuong: Optional[str] = None, chuong_i: int = 0,
    muc: Optional[str] = None, muc_i: int = 0,
) -> dict:
    return {
        "phan": phan, "phan_index": phan_i,
        "chuong": chuong, "chuong_index": chuong_i,
        "muc": muc, "muc_index": muc_i,
    }


# ── Main parser ──────────────────────────────────────────────────────────────


class LegalChunker:
    """Parse a single legal document file into structured chunks."""

    def __init__(self, file_path: str | Path):
        self.path = Path(file_path)
        self.doc_type = self.path.parent.name
        self.doc_id = f"{self.doc_type}/{self.path.stem}"

    # ── Public ──────────────────────────────────────────────────────────────

    def parse(self) -> ParseResult:
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return ParseResult(
                doc_id=self.doc_id, doc_type=self.doc_type,
                file_path=str(self.path), parse_type="empty", lang="vi",
                total_chunks=0, paywall_lines=0,
                warnings=[ParseWarning("error", f"Cannot read file: {e}")],
                chunks=[],
            )

        lines = text.splitlines()
        warnings: list[ParseWarning] = []

        paywall_count = sum(1 for l in lines if _is_paywall(l))

        has_dieu = any(_RE_DIEU.match(l.strip()) for l in lines)
        has_article = any(_RE_ARTICLE.match(l.strip()) for l in lines)

        if has_dieu and has_article:
            lang = "mixed"
        elif has_article:
            lang = "en"
        else:
            lang = "vi"

        if has_dieu:
            parse_type = "dieu"
            chunks = self._parse_dieu(lines, warnings)
        elif has_article:
            parse_type = "article"
            chunks = self._parse_article(lines, warnings)
        elif not any(l.strip() for l in lines):
            parse_type = "empty"
            chunks = []
            warnings.append(ParseWarning("warn", "File is empty"))
        else:
            parse_type = "numbered"
            chunks = self._parse_numbered(lines, warnings)

        # Sanity: expected articles but got none
        if parse_type in ("dieu", "article") and len(chunks) == 0:
            warnings.append(ParseWarning("error", "Markers found but no chunks extracted"))

        # Link prev/next
        for i, chunk in enumerate(chunks):
            chunk.prev_article = chunks[i - 1].article_number if i > 0 else None
            chunk.next_article = chunks[i + 1].article_number if i < len(chunks) - 1 else None

        return ParseResult(
            doc_id=self.doc_id, doc_type=self.doc_type,
            file_path=str(self.path), parse_type=parse_type, lang=lang,
            total_chunks=len(chunks), paywall_lines=paywall_count,
            warnings=warnings, chunks=chunks,
        )

    # ── Parse strategies ─────────────────────────────────────────────────────

    def _parse_dieu(self, lines: list[str], warnings: list[ParseWarning]) -> list[Chunk]:
        chunks: list[Chunk] = []
        phan: Optional[str] = None
        phan_i = 0
        chuong: Optional[str] = None
        chuong_i = 0
        muc: Optional[str] = None
        muc_i = 0

        cur_num: Optional[str] = None
        cur_title = ""
        cur_lines: list[str] = []

        i = 0
        while i < len(lines):
            raw = lines[i]
            s = raw.strip()
            i += 1

            if not s or _is_paywall(s):
                if cur_num is not None:
                    cur_lines.append("")
                continue

            # ── PHẦN ──────────────────────────────────────────────────────
            m = _RE_PHAN.match(s)
            if m:
                rest = m.group(1).strip()
                # "PHẦN CHUNG CHƯƠNG I. ..." — inline CHƯƠNG
                if "CHƯƠNG" in rest:
                    parts = rest.split("CHƯƠNG", 1)
                    phan_i += 1
                    phan = parts[0].strip()
                    chuong_i += 1
                    muc = None
                    muc_i = 0
                    chuong = ("CHƯƠNG " + parts[1].strip()).strip()
                else:
                    phan_i += 1
                    phan = rest
                    muc = None
                    muc_i = 0
                continue

            # ── CHƯƠNG (uppercase) ────────────────────────────────────────
            m = _RE_CHUONG_UPPER.match(s)
            if m:
                chuong_i += 1
                muc = None
                muc_i = 0
                title_inline = m.group(1).strip()
                if not title_inline:
                    _, title_inline = _next_nonempty(lines, i)
                chuong = ("CHƯƠNG " + title_inline).strip()
                continue

            # ── Chương (mixed case) ───────────────────────────────────────
            m = _RE_CHUONG_LOWER.match(s)
            if m:
                chuong_i += 1
                muc = None
                muc_i = 0
                num_part = m.group(1)
                title_inline = m.group(2).strip()
                if not title_inline:
                    _, title_inline = _next_nonempty(lines, i)
                chuong = f"Chương {num_part} {title_inline}".strip()
                continue

            # ── MỤC ───────────────────────────────────────────────────────
            m = _RE_MUC.match(s)
            if m:
                muc_i += 1
                muc = ("MỤC " + m.group(1).strip()).strip()
                continue

            # ── TIẾT (between Mục and Điều, rare) ────────────────────────
            m = _RE_TIET.match(s)
            if m:
                # treat as sub-mục; don't increment muc_i
                muc = ("TIẾT " + m.group(1).strip()).strip()
                continue

            # ── Điều ──────────────────────────────────────────────────────
            m = _RE_DIEU.match(s)
            if m:
                if cur_num is not None:
                    chunks.append(self._build_chunk(
                        cur_num, cur_title, cur_lines,
                        phan, phan_i, chuong, chuong_i, muc, muc_i,
                    ))
                cur_num = m.group(1)
                inline = m.group(2).strip()
                cur_lines = []

                if inline:
                    # Short inline → it's the title (and first content line)
                    # Long inline → it's content only (no separate title)
                    cur_title = inline if len(inline) <= _MAX_TITLE_CHARS else ""
                    cur_lines.append(inline)
                else:
                    # Title on next non-empty line (if not another structural marker)
                    j, next_s = _next_nonempty(lines, i)
                    if next_s and not _RE_DIEU.match(next_s) \
                            and not _RE_CHUONG_UPPER.match(next_s) \
                            and not _RE_CHUONG_LOWER.match(next_s) \
                            and not _RE_MUC.match(next_s):
                        cur_title = next_s if len(next_s) <= _MAX_TITLE_CHARS else ""
                        i = j + 1
                    else:
                        cur_title = ""
                continue

            # ── Content ───────────────────────────────────────────────────
            if cur_num is not None:
                cur_lines.append(s)

        if cur_num is not None:
            chunks.append(self._build_chunk(
                cur_num, cur_title, cur_lines,
                phan, phan_i, chuong, chuong_i, muc, muc_i,
            ))

        return chunks

    def _parse_article(self, lines: list[str], warnings: list[ParseWarning]) -> list[Chunk]:
        chunks: list[Chunk] = []
        chuong: Optional[str] = None
        chuong_i = 0
        muc: Optional[str] = None
        muc_i = 0

        cur_num: Optional[str] = None
        cur_title = ""
        cur_lines: list[str] = []

        for raw in lines:
            s = raw.strip()
            if not s or _is_paywall(s):
                continue

            m = _RE_CHAPTER_EN.match(s)
            if m:
                chuong_i += 1
                muc = None
                muc_i = 0
                chuong = f"Chapter {m.group(1)} {m.group(2)}".strip()
                continue

            m = _RE_SECTION_EN.match(s)
            if m:
                muc_i += 1
                muc = f"Section {m.group(1)} {m.group(2)}".strip()
                continue

            m = _RE_ARTICLE.match(s)
            if m:
                if cur_num is not None:
                    chunks.append(self._build_chunk(
                        cur_num, cur_title, cur_lines,
                        None, 0, chuong, chuong_i, muc, muc_i,
                    ))
                cur_num = m.group(1)
                inline = m.group(2).strip()
                cur_title = inline if len(inline) <= _MAX_TITLE_CHARS else ""
                cur_lines = [inline] if inline else []
                continue

            if cur_num is not None:
                cur_lines.append(s)

        if cur_num is not None:
            chunks.append(self._build_chunk(
                cur_num, cur_title, cur_lines,
                None, 0, chuong, chuong_i, muc, muc_i,
            ))

        return chunks

    def _parse_numbered(self, lines: list[str], warnings: list[ParseWarning]) -> list[Chunk]:
        """Fallback for docs without Điều/Article. Split on Roman numeral sections."""
        content_lines = [l.strip() for l in lines if l.strip() and not _is_paywall(l)]

        if not content_lines:
            return []

        # Try Roman-numeral split
        sections: list[tuple[str, list[str]]] = []
        cur_title_: Optional[str] = None
        cur_body: list[str] = []

        for line in content_lines:
            m = _RE_ROMAN.match(line)
            if m:
                if cur_title_ is not None:
                    sections.append((cur_title_, cur_body))
                cur_title_ = line
                cur_body = []
            else:
                if cur_title_ is not None:
                    cur_body.append(line)

        if cur_title_ is not None:
            sections.append((cur_title_, cur_body))

        if sections:
            warnings.append(ParseWarning("warn", f"No Điều — split into {len(sections)} Roman-numeral sections"))
            chunks = []
            for idx, (sec_title, sec_lines) in enumerate(sections):
                body = [l for l in sec_lines if l.strip()]
                chunks.append(Chunk(
                    chunk_id=f"{self.doc_id}#section_{idx + 1}",
                    article_number=str(idx + 1),
                    article_title=sec_title,
                    path=_make_path(),
                    content=body,
                    khoan=[],
                    char_count=sum(len(l) for l in body),
                    citation_keys=[],
                    prev_article=None,
                    next_article=None,
                ))
            return chunks

        # Single whole-doc chunk
        warnings.append(ParseWarning("warn", "No Điều and no Roman sections — single whole-doc chunk"))
        body = [l for l in content_lines if l.strip()]
        return [Chunk(
            chunk_id=f"{self.doc_id}#full",
            article_number="0",
            article_title="",
            path=_make_path(),
            content=body,
            khoan=[],
            char_count=sum(len(l) for l in body),
            citation_keys=[],
            prev_article=None,
            next_article=None,
        )]

    # ── Builder ──────────────────────────────────────────────────────────────

    def _build_chunk(
        self,
        article_number: str,
        article_title: str,
        content_lines: list[str],
        phan: Optional[str], phan_i: int,
        chuong: Optional[str], chuong_i: int,
        muc: Optional[str], muc_i: int,
    ) -> Chunk:
        content = [l for l in content_lines if l.strip()]
        khoans = _parse_khoan(content_lines)
        citation_keys = _make_citation_keys(article_number, khoans)

        return Chunk(
            chunk_id=f"{self.doc_id}#dieu_{article_number}",
            article_number=article_number,
            article_title=article_title,
            path=_make_path(phan, phan_i, chuong, chuong_i, muc, muc_i),
            content=content,
            khoan=khoans,
            char_count=sum(len(l) for l in content),
            citation_keys=citation_keys,
            prev_article=None,
            next_article=None,
        )


# ── Convenience function ─────────────────────────────────────────────────────


def parse_file(path: str | Path) -> ParseResult:
    """Parse a single full-text legal document. Main entry point."""
    return LegalChunker(path).parse()
