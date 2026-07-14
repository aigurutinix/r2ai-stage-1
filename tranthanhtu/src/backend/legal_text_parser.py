"""Shared parsing helpers for Vietnamese legal text (DRY).

Used by build_corpus.py (HF markdown → articles).
Two jobs:
  1. parse_legal_name()      — split a raw law name into code / type / clean name.
  2. split_into_articles()   — slice a full statute body into Điều-level chunks,
                               robust to flat markdown (lost newlines).
"""
import re
from typing import List, Dict, Optional

# Article header e.g. "Điều 12. Quyền của doanh nghiệp".
# Require the period after the number (this alone excludes most inline references
# like "...tại Điều 5 của Luật..." which have no period), but allow ZERO spaces
# after it so flattened markdown ("Điều 1.Phạm vi...") still matches.
_ARTICLE_HEADER = re.compile(r"Điều\s+(\d+)\s*\.\s*")
# Reject a match preceded by a reference word (the remaining inline-ref case).
_REF_PREFIX = re.compile(
    r"(tại|theo|của|và|đến|từ|căn cứ|quy định tại|khoản\s+\d+|điểm\s+[a-zđ])\s*$",
    re.IGNORECASE,
)
_CLAUSE = re.compile(r"^\s*(\d+)\s*\.\s+")          # "1. ..." khoản
_POINT = re.compile(r"^\s*([a-zđ])\s*\)\s+")        # "a) ..." điểm

_LAW_TYPES = ("Bộ luật", "Luật", "Nghị định", "Thông tư", "Nghị quyết",
              "Quyết định", "Pháp lệnh", "Hiến pháp")


def parse_legal_name(law_name: str, doc_number: str = "", legal_type: str = "") -> Dict[str, str]:
    """Return {code, type, clean_name} for a law.

    doc_number / legal_type (from structured HF fields) take priority over regex
    sniffing on the free-text name.
    """
    law_name = re.sub(r"\s+", " ", (law_name or "")).strip()

    code = (doc_number or "").strip()
    if not code:
        m = re.search(r"\b(\d+/\d+/[A-Z0-9\-ĐĐ]+|\d+-[A-Z0-9\-/]+)\b", law_name)
        if m:
            code = m.group(1)

    doc_type = (legal_type or "").strip()
    if not doc_type:
        for t in _LAW_TYPES:
            if law_name.lower().startswith(t.lower()):
                doc_type = t
                break
        if not doc_type:
            doc_type = "Văn bản"

    # Trích yếu = name minus leading type and embedded code.
    subject = law_name
    if doc_type and subject.lower().startswith(doc_type.lower()):
        subject = subject[len(doc_type):].strip()
    if code:
        subject = re.sub(rf"\b(số\s+|số:\s*)?{re.escape(code)}\b", "", subject,
                         flags=re.IGNORECASE).strip()
    subject = re.sub(r"\b(số)\s*$", "", subject, flags=re.IGNORECASE).strip().strip(",. ").strip()

    clean_name = f"{doc_type} {subject}".strip() if subject else doc_type
    return {"code": code, "type": doc_type, "clean_name": clean_name}


def split_into_articles(body: str, max_title_len: int = 160) -> List[Dict[str, str]]:
    """Slice a statute body into article-level chunks.

    Returns list of {article: "Điều N", title, text}. Handles flat markdown by
    splitting on "Điều N." headers wherever a real title (uppercase) follows.
    """
    if not body:
        return []
    body = body.replace("\r", "\n")

    # Collect genuine article-header positions.
    headers = []
    for m in _ARTICLE_HEADER.finditer(body):
        nxt = m.end()
        follow = body[nxt] if nxt < len(body) else ""
        if not follow:
            continue
        # Skip inline references: "...tại Điều 5. ...", "...khoản 2 Điều 36. ..."
        if _REF_PREFIX.search(body[max(0, m.start() - 14):m.start()]):
            continue
        # Accept if a title character follows (letter/digit/opening quote/paren).
        # Period requirement + ref-prefix check already exclude most refs; the
        # min-length filter below drops any stray short match.
        if follow.isalnum() or follow in "\"'(«“‘":
            headers.append((int(m.group(1)), m.start(), m.end()))
    if not headers:
        return []

    articles = []
    for i, (num, start, header_end) in enumerate(headers):
        end = headers[i + 1][1] if i + 1 < len(headers) else len(body)
        chunk = body[start:end].strip()
        if len(chunk) < 20:          # too short → stray reference, skip
            continue
        # Title = text right after header up to first newline / sentence end.
        after = body[header_end:end].strip()
        title_cut = re.split(r"[\n\.]", after, maxsplit=1)[0].strip()
        title = title_cut[:max_title_len]
        articles.append({
            "article": f"Điều {num}",
            "title": f"Điều {num}. {title}",
            "text": chunk,
        })
    return articles


def split_into_clauses(article_text: str) -> List[str]:
    """Optional: split an article into khoản strings (for finer child chunks)."""
    lines = [ln.strip() for ln in article_text.split("\n") if ln.strip()]
    clauses, current = [], ""
    for ln in lines:
        if _CLAUSE.match(ln):
            if current:
                clauses.append(current.strip())
            current = ln
        else:
            current += " " + ln
    if current:
        clauses.append(current.strip())
    return clauses
