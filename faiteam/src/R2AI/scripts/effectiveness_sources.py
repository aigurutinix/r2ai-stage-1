"""Fetch legal effectiveness metadata: vbpl.vn first, VietLex as fallback."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from vbpl_client import fetch_via_vbpl_crawler, search_by_document_number

BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
TRANSLATION_LOAI = re.compile(r"bản dịch", re.IGNORECASE)


def _normalize_document_number(value: str | None) -> str:
    return re.sub(r"\s+", "", (value or "").strip()).upper()


def _json_request(url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _score_vietlex_candidate(item: dict[str, Any], *, document_number: str, title: str | None) -> int:
    score = 0
    if _normalize_document_number(item.get("soHieu")) == _normalize_document_number(document_number):
        score += 100
    loai = str(item.get("loai") or "")
    if loai and not TRANSLATION_LOAI.search(loai):
        score += 20
    nguon = str(item.get("nguon") or "").lower()
    if "vbpl" in nguon or "moj.gov.vn" in nguon:
        score += 10
    if title:
        title_lower = title.lower()
        item_title = str(item.get("title") or "").lower()
        score += sum(1 for token in title_lower.split()[:8] if len(token) > 2 and token in item_title)
    return score


def fetch_vietlex_by_number(
    document_number: str,
    *,
    title: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any] | None:
    query = urllib.parse.quote(document_number.strip())
    search_url = f"https://vietlex.vn/api/v1/search?q={query}&limit=30"
    payload = _json_request(search_url, timeout=timeout)
    candidates = payload.get("results") or []
    if not candidates:
        return None

    ranked = sorted(
        candidates,
        key=lambda item: _score_vietlex_candidate(item, document_number=document_number, title=title),
        reverse=True,
    )
    best = ranked[0]
    if _score_vietlex_candidate(best, document_number=document_number, title=title) < 100:
        exact = [
            item
            for item in candidates
            if _normalize_document_number(item.get("soHieu")) == _normalize_document_number(document_number)
        ]
        if not exact:
            return None
        best = sorted(
            exact,
            key=lambda item: _score_vietlex_candidate(item, document_number=document_number, title=title),
            reverse=True,
        )[0]

    doc_id = best.get("id")
    if not doc_id:
        return None

    detail_payload = _json_request(f"https://vietlex.vn/api/v1/document/{doc_id}", timeout=timeout)
    detail = detail_payload.get("document") or {}

    effective_date = detail.get("ngayCoHieuLuc") or detail.get("ngayHieuLuc") or detail.get("ngayBanHanh") or ""
    expiry_date = detail.get("ngayHetHieuLuc") or detail.get("ngayHetHan") or ""

    return {
        "eff_code": detail.get("hieuLucCode") or "",
        "eff_status": detail.get("hieuLuc") or "",
        "effective_date": effective_date or "",
        "expiry_date": expiry_date or "",
        "source": "vietlex",
        "source_id": str(doc_id),
    }


def fetch_vbpl_by_number(
    document_number: str,
    *,
    title: str | None = None,
    timeout: float = 30.0,
    verify_ssl: bool = False,
) -> dict[str, Any] | None:
    via_crawler = fetch_via_vbpl_crawler(document_number)
    if via_crawler:
        return via_crawler
    return search_by_document_number(
        document_number,
        title=title,
        timeout=timeout,
        verify_ssl=verify_ssl,
    )


def fetch_effectiveness(
    document_number: str,
    *,
    title: str | None = None,
    timeout: float = 30.0,
    verify_ssl: bool = False,
    use_vbpl: bool = True,
    use_vietlex: bool = True,
) -> dict[str, Any]:
    """Return effectiveness record; vbpl first, VietLex fills gaps."""
    empty = {
        "eff_code": "",
        "eff_status": "",
        "effective_date": "",
        "expiry_date": "",
        "source": "unknown",
        "source_id": "",
        "vbpl_matched": False,
        "vietlex_matched": False,
    }
    if not document_number.strip():
        return empty

    vbpl_data: dict[str, Any] | None = None
    vietlex_data: dict[str, Any] | None = None

    if use_vbpl:
        try:
            vbpl_data = fetch_vbpl_by_number(
                document_number,
                title=title,
                timeout=timeout,
                verify_ssl=verify_ssl,
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError, ValueError):
            vbpl_data = None

    record = dict(empty)
    if vbpl_data and (vbpl_data.get("eff_code") or vbpl_data.get("eff_status")):
        record.update(vbpl_data)
        record["vbpl_matched"] = True
        return record

    if use_vietlex:
        try:
            vietlex_data = fetch_vietlex_by_number(
                document_number,
                title=title,
                timeout=timeout,
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            vietlex_data = None

    if vietlex_data and (vietlex_data.get("eff_code") or vietlex_data.get("eff_status")):
        record.update(vietlex_data)
        record["vietlex_matched"] = True
        return record

    return record
