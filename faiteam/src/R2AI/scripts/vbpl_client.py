"""Minimal client for vbpl.vn SOAP web service (ws.vbpl.vn)."""

from __future__ import annotations

import re
import ssl
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

SOAP_URL = "https://ws.vbpl.vn/vbqppl.asmx"
NS = {"soap": "http://schemas.xmlsoap.org/soap/envelope/", "t": "http://tempuri.org/"}

# Common vbpl effectiveness status codes (fallback when only int id is returned).
TRANG_THAI_HIEU_LUC_MAP: dict[str, tuple[str, str]] = {
    "1": ("CHL", "Còn hiệu lực"),
    "2": ("HHL", "Hết hiệu lực"),
    "3": ("HHLTB", "Hết hiệu lực toàn bộ"),
    "4": ("HHTB", "Hết hiệu lực một phần"),
    "5": ("NGUNG_HL", "Ngưng hiệu lực"),
}

TRANSLATION_LOAI = re.compile(r"bản dịch", re.IGNORECASE)


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(elem: ET.Element | None) -> str:
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def _child_text(parent: ET.Element, name: str) -> str:
    for child in parent:
        if _local_tag(child.tag) == name:
            return _text(child)
    return ""


def _parse_soap_datetime(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip()
    if not value or value.startswith("0001-") or value.startswith("1900-"):
        return ""
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y",
    ):
        try:
            return datetime.strptime(value[:26], fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%d/%m/%Y")
    except ValueError:
        return value


def _normalize_document_number(value: str | None) -> str:
    return re.sub(r"\s+", "", (value or "").strip()).upper()


def _soap_envelope(body_xml: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
        'xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
        f"<soap:Body>{body_xml}</soap:Body></soap:Envelope>"
    )


def soap_call(action: str, body_xml: str, *, timeout: float = 30.0, verify_ssl: bool = False) -> ET.Element:
    payload = _soap_envelope(body_xml).encode("utf-8")
    req = urllib.request.Request(
        SOAP_URL,
        data=payload,
        headers={
            "Content-Type": "text/xml; charset=utf-8",
            "SOAPAction": f'"http://tempuri.org/{action}"',
            "User-Agent": "Mozilla/5.0 (compatible; R2AI-effectiveness/1.0)",
        },
    )
    context = None
    if not verify_ssl:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
        root = ET.fromstring(resp.read())
    body = root.find("soap:Body", NS)
    if body is None:
        raise ValueError("SOAP response missing Body")
    fault = body.find("soap:Fault", NS)
    if fault is not None:
        reason = _text(fault.find("faultstring"))
        raise RuntimeError(reason or "SOAP fault")
    return body


def _extract_status(code_raw: str, status_raw: str) -> tuple[str, str]:
    code = (code_raw or "").strip().upper()
    status = (status_raw or "").strip()
    if not code and status_raw.isdigit():
        mapped = TRANG_THAI_HIEU_LUC_MAP.get(status_raw.strip())
        if mapped:
            return mapped
    if not code and status:
        lowered = status.lower()
        if "còn hiệu lực" in lowered:
            return "CHL", status
        if "hết hiệu lực" in lowered and "một phần" in lowered:
            return "HHTB", status
        if "hết hiệu lực" in lowered:
            return "HHL", status
        if "ngưng hiệu lực" in lowered:
            return "NGUNG_HL", status
    if code and not status:
        for mapped_code, mapped_status in TRANG_THAI_HIEU_LUC_MAP.values():
            if mapped_code == code:
                return code, mapped_status
    return code, status


def parse_van_ban_item(item: ET.Element) -> dict[str, Any]:
    code_raw = _child_text(item, "VBPQTinhTrangHieuLucCode")
    status_raw = _child_text(item, "VBPQTinhTrangHieuLuc")
    if not status_raw:
        status_raw = _child_text(item, "TrangThaiHieuLucText")
    if not code_raw and status_raw.isdigit():
        code_raw = status_raw
        status_raw = ""
    eff_code, eff_status = _extract_status(code_raw, status_raw)

    item_id = _child_text(item, "ID") or _child_text(item, "ItemID")
    return {
        "eff_code": eff_code,
        "eff_status": eff_status,
        "effective_date": _parse_soap_datetime(_child_text(item, "VBPQNgaycohieuluc")),
        "expiry_date": _parse_soap_datetime(
            _child_text(item, "VBPQNgayHetHieuLuc") or _child_text(item, "VBPQNgayHetHieuLuc1phan")
        ),
        "source": "vbpl",
        "source_id": item_id,
        "document_number": _child_text(item, "VBPQSokyhieu"),
        "title": _child_text(item, "Title1") or _child_text(item, "Title"),
        "issuing_authority": _child_text(item, "VBPQCoQuanBanHanh"),
    }


def _collect_van_ban_items(body: ET.Element) -> list[ET.Element]:
    items: list[ET.Element] = []
    for elem in body.iter():
        tag = _local_tag(elem.tag)
        if tag in {"VanBanItem", "VanBanPhapDienItem", "VanBanSearchItem"}:
            items.append(elem)
    return items


def _pick_best_item(
    items: list[dict[str, Any]],
    *,
    document_number: str,
    title: str | None = None,
) -> dict[str, Any] | None:
    if not items:
        return None
    target = _normalize_document_number(document_number)
    exact = [
        item
        for item in items
        if _normalize_document_number(item.get("document_number")) == target
    ]
    pool = exact or items

    if title:
        title_lower = title.lower()
        scored = sorted(
            pool,
            key=lambda item: sum(
                1 for token in title_lower.split()[:6] if token and token in (item.get("title") or "").lower()
            ),
            reverse=True,
        )
        return scored[0]
    return pool[0]


def search_by_document_number(
    document_number: str,
    *,
    title: str | None = None,
    timeout: float = 30.0,
    verify_ssl: bool = False,
) -> dict[str, Any] | None:
    """Lookup effectiveness metadata on vbpl.vn by official document number."""
    if not document_number.strip():
        return None

    body_xml = (
        '<GetListVanBanByListSKH xmlns="http://tempuri.org/">'
        f"<ListSKH>{document_number.strip()}</ListSKH>"
        "</GetListVanBanByListSKH>"
    )
    try:
        body = soap_call("GetListVanBanByListSKH", body_xml, timeout=timeout, verify_ssl=verify_ssl)
    except (urllib.error.URLError, TimeoutError, ET.ParseError, RuntimeError, ValueError):
        body = None

    parsed: list[dict[str, Any]] = []
    if body is not None:
        parsed = [parse_van_ban_item(item) for item in _collect_van_ban_items(body)]

    if not parsed:
        search_xml = (
            '<TimKiemVanBanNew xmlns="http://tempuri.org/">'
            f"<Keyword>{document_number.strip()}</Keyword>"
            "<SearchExact>true</SearchExact>"
            "<SearchInExact>Title1</SearchInExact>"
            "<rowPerPage>20</rowPerPage>"
            "<PageIndex>1</PageIndex>"
            "</TimKiemVanBanNew>"
        )
        try:
            body = soap_call("TimKiemVanBanNew", search_xml, timeout=timeout, verify_ssl=verify_ssl)
            parsed = [parse_van_ban_item(item) for item in _collect_van_ban_items(body)]
        except (urllib.error.URLError, TimeoutError, ET.ParseError, RuntimeError, ValueError):
            parsed = []

    parsed = [item for item in parsed if item.get("eff_code") or item.get("eff_status")]
    best = _pick_best_item(parsed, document_number=document_number, title=title)
    if not best:
        return None
    return {
        "eff_code": best.get("eff_code") or "",
        "eff_status": best.get("eff_status") or "",
        "effective_date": best.get("effective_date") or "",
        "expiry_date": best.get("expiry_date") or "",
        "source": "vbpl",
        "source_id": str(best.get("source_id") or ""),
    }


def fetch_via_vbpl_crawler(document_number: str) -> dict[str, Any] | None:
    """Use local VbplCrawler package when available (doc crawl pipeline)."""
    try:
        from VbplCrawler import VbplCrawler  # type: ignore
    except ImportError:
        return None

    crawler = VbplCrawler(request_delay=0.0)
    search = getattr(crawler, "search_by_number", None) or getattr(crawler, "search", None)
    if not callable(search):
        return None
    try:
        results = search(document_number)
    except Exception:
        return None
    if not results:
        return None
    doc = results[0] if isinstance(results, list) else results
    if callable(doc):
        return None
    status = (doc or {}).get("effStatus") or {}
    code = (status.get("code") or "").strip().upper()
    name = (status.get("name") or "").strip()
    if not code and not name:
        return None
    return {
        "eff_code": code,
        "eff_status": name,
        "effective_date": (doc or {}).get("effectiveDate") or (doc or {}).get("ngayBanHanh") or "",
        "expiry_date": (doc or {}).get("expiryDate") or "",
        "source": "vbpl",
        "source_id": str((doc or {}).get("id") or (doc or {}).get("docId") or ""),
    }
