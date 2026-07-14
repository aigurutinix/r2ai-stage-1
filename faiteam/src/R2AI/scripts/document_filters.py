"""Document filtering helpers for parquet ingest."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

import pandas as pd

DEFAULT_CUTOFF = date(2026, 3, 1)

TITLE_KEYWORDS = [
    "công ty",
    "thuế",
    "doanh nghiệp",
    "quyền",
    "lao động",
    "xử lý",
    "đăng ký",
    "hợp đồng",
    "hồ sơ",
    "nhân viên",
    "cơ quan",
    "yêu cầu",
    "quy định",
    "nội dung",
    "điều kiện",
    "thời hạn",
    "thông tin",
    "hỗ trợ",
    "trách nhiệm",
    "tổ chức",
]

CATALOG_TITLE_PATTERNS = (
    "công bố danh mục",
    "danh mục văn bản",
    "văn bản quy phạm pháp luật hết hiệu lực",
)

EFFECTIVE_PATTERNS = [
    re.compile(
        r"có hiệu lực thi hành kể từ ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"có hiệu lực kể từ ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"có hiệu lực từ ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        re.IGNORECASE,
    ),
]

EXPIRY_PATTERNS = [
    re.compile(
        r"hết hiệu lực kể từ ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"hết hiệu lực từ ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"ngày hết hiệu lực[:\s]+(\d{1,2})/(\d{1,2})/(\d{4})",
        re.IGNORECASE,
    ),
]

STILL_EFFECTIVE_CODES = {"CHL", "HL", "CON_HL"}
EXPIRED_CODES = {"HHL", "HHLTB", "HHTB", "HHLTP", "NGUNG_HL"}


def load_keywords(keywords_file: str | None = None) -> list[str]:
    if not keywords_file:
        return TITLE_KEYWORDS
    from pathlib import Path

    lines = Path(keywords_file).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def matched_keywords(title: str, keywords: list[str]) -> list[str]:
    title_lower = (title or "").lower()
    return [kw for kw in keywords if kw in title_lower]


def filter_by_title(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    title = df["title"].str.lower().fillna("")
    mask = False
    for kw in keywords:
        mask = mask | title.str.contains(kw, regex=False, na=False)
    return df[mask].copy()


def is_catalog_about_expired_laws(title: str) -> bool:
    title_lower = (title or "").lower()
    return any(pattern in title_lower for pattern in CATALOG_TITLE_PATTERNS)


def parse_vn_date(day: str, month: str, year: str) -> date | None:
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def parse_metadata_date(value: str | None) -> date | None:
    if not value:
        return None
    value = str(value).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _first_match(patterns: list[re.Pattern[str]], text: str) -> date | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            parsed = parse_vn_date(match.group(1), match.group(2), match.group(3))
            if parsed:
                return parsed
    return None


def parse_effective_date_from_content(content: str) -> date | None:
    if not content:
        return None
    tail = content[-12000:]
    if re.search(r"có hiệu lực từ ngày ký ban hành", tail, re.IGNORECASE):
        return None
    return _first_match(EFFECTIVE_PATTERNS, tail)


def parse_expiry_date_from_content(content: str) -> date | None:
    if not content:
        return None
    return _first_match(EXPIRY_PATTERNS, content[:5000] + content[-8000:])


def normalize_eff_code(value: str | None) -> str:
    return (value or "").strip().upper()


def normalize_eff_status(value: str | None) -> str:
    return (value or "").strip().lower()


def is_effective_as_of(
    *,
    cutoff: date = DEFAULT_CUTOFF,
    eff_code: str | None = None,
    eff_status: str | None = None,
    expiry_date: date | None = None,
    effective_date: date | None = None,
    title: str | None = None,
    content: str | None = None,
    issuance_date: date | None = None,
    unknown_policy: str = "include",
) -> bool:
    """Return True if document was applicable on ``cutoff``.

    Rule requested by user:
    - Documents that expired AFTER cutoff remain valid on cutoff.
    - Exclude documents expired on/before cutoff, or not yet effective on cutoff.
    """
    if title and is_catalog_about_expired_laws(title):
        return False

    code = normalize_eff_code(eff_code)
    status = normalize_eff_status(eff_status)

    if code in STILL_EFFECTIVE_CODES or "còn hiệu lực" in status:
        return True

    if code in EXPIRED_CODES or status in {"hết hiệu lực toàn bộ", "hết hiệu lực", "ngưng hiệu lực"}:
        if "một phần" in status:
            return True
        if expiry_date is not None:
            return expiry_date > cutoff
        return False

    if expiry_date is None and content:
        expiry_date = parse_expiry_date_from_content(content)
    if effective_date is None and content:
        effective_date = parse_effective_date_from_content(content)
    if effective_date is None and issuance_date is not None:
        if content and re.search(r"có hiệu lực từ ngày ký ban hành", content[-12000:], re.IGNORECASE):
            effective_date = issuance_date

    if effective_date and effective_date > cutoff:
        return False

    if expiry_date is not None:
        return expiry_date > cutoff

    if unknown_policy == "exclude":
        return False
    return True


def load_effectiveness_table(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    from pathlib import Path

    eff_path = Path(path)
    if not eff_path.is_file():
        return None
    table = pd.read_parquet(eff_path)
    if "id" not in table.columns:
        return None
    return table.set_index("id", drop=False)


def effectiveness_row_for(table: pd.DataFrame | None, doc_id: int) -> dict[str, Any] | None:
    if table is None or doc_id not in table.index:
        return None
    row = table.loc[doc_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row.to_dict()


def filter_metadata_effective(
    df: pd.DataFrame,
    *,
    cutoff: date = DEFAULT_CUTOFF,
    effectiveness_table: pd.DataFrame | None = None,
    unknown_policy: str = "include",
) -> pd.DataFrame:
    kept_rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        doc_id = int(row_dict["id"])
        eff = effectiveness_row_for(effectiveness_table, doc_id)
        ok = is_effective_as_of(
            cutoff=cutoff,
            eff_code=(eff or {}).get("eff_code"),
            eff_status=(eff or {}).get("eff_status"),
            expiry_date=parse_metadata_date((eff or {}).get("expiry_date")),
            effective_date=parse_metadata_date((eff or {}).get("effective_date")),
            title=str(row_dict.get("title") or ""),
            issuance_date=parse_metadata_date(row_dict.get("issuance_date")),
            unknown_policy=unknown_policy,
        )
        if ok:
            kept_rows.append(row_dict)
    return pd.DataFrame(kept_rows)


def filter_document_row(
    row: dict[str, Any],
    *,
    content: str | None = None,
    cutoff: date = DEFAULT_CUTOFF,
    effectiveness_table: pd.DataFrame | None = None,
    unknown_policy: str = "include",
) -> bool:
    doc_id = int(row["id"])
    eff = effectiveness_row_for(effectiveness_table, doc_id)
    return is_effective_as_of(
        cutoff=cutoff,
        eff_code=(eff or {}).get("eff_code"),
        eff_status=(eff or {}).get("eff_status"),
        expiry_date=parse_metadata_date((eff or {}).get("expiry_date")),
        effective_date=parse_metadata_date((eff or {}).get("effective_date")),
        title=str(row.get("title") or ""),
        content=content,
        issuance_date=parse_metadata_date(row.get("issuance_date")),
        unknown_policy=unknown_policy,
    )
