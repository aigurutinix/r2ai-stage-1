"""Fetch + parse văn bản từ API CHÍNH THỨC vbpl (Bộ Tư pháp) — sạch hơn HF dataset.

Gateway: https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{id}
 - id lấy từ trailing "--{id}" của source_url trong corpus (vbpl.vn/.../slug--100024).
 - documentContent.content = HTML có class prov-article/prov-clause/prov-item → ranh giới
   Điều/Khoản/Điểm đánh dấu SẴN (khỏi regex đoán "Điều X" như legal_text_parser).
 - effStatus.name = trạng thái hiệu lực CHÍNH THỨC ("Còn hiệu lực" / "Hết hiệu lực toàn bộ"...)
   → thay heuristic SUPERSEDED_DOCS bằng sự thật.

Dùng: build lại corpus SME từ nguồn chính thống + lọc hiệu lực thật + phục hồi doc null-markdown.
Chạy fetch số lượng lớn trên Kaggle (mạng nhanh), KHÔNG chạy local (máy user lag).
"""
import json
import re
import time
import urllib.request

API = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{}"
_TAG = re.compile(r"<[^>]+>")                       # strip thẻ HTML
_P = re.compile(r'<p\b[^>]*class="([^"]*)"[^>]*>(.*?)</p>', re.S | re.I)
_ART = re.compile(r"Điều\s+(\d+)\s*\.?\s*(.*)", re.S)
# Chỉ BỎ doc hết hiệu lực TOÀN BỘ. Giữ "còn hiệu lực" + "hết hiệu lực MỘT PHẦN" (vẫn còn điều
# hiệu lực) + trạng thái lạ → recall-safe (F2 nặng recall; ta thừa precision).
EFF_DEAD = "hết hiệu lực toàn bộ"


def extract_id(source_url: str):
    """vbpl.vn/.../slug--100024 hoặc .../doc/100024 → '100024'. None nếu không thấy."""
    m = re.search(r"(?:--|/)(\d{3,})\s*$", (source_url or "").strip())
    return m.group(1) if m else None


def _clean(html: str) -> str:
    t = _TAG.sub("", html or "")
    t = (t.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
         .replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'"))
    return re.sub(r"\s+", " ", t).strip()


def fetch_doc(doc_id, retries=4, timeout=40):
    """GET gateway → dict {docNum, title, effStatus, issueDate, effFrom, effTo, content} | None."""
    req = urllib.request.Request(API.format(doc_id), headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8"))
            x = d.get("data") or {}
            if not d.get("success") or not x:
                return None
            return {
                "id": str(doc_id),
                "docNum": x.get("docNum"),
                "title": x.get("title"),
                "docType": (x.get("docType") or {}).get("name"),
                "effStatus": (x.get("effStatus") or {}).get("name"),
                "issueDate": x.get("issueDate"),
                "effFrom": x.get("effFrom"),
                "effTo": x.get("effTo"),
                "content": (x.get("documentContent") or {}).get("content") or "",
            }
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def parse_articles(content_html: str):
    """HTML prov-* → [{article, title, text}]. Gom Khoản/Điểm/content vào Điều hiện tại."""
    arts, cur = [], None
    for cls, inner in _P.findall(content_html or ""):
        cls = cls.lower()
        txt = _clean(inner)
        if not txt:
            continue
        if "prov-article" in cls:                  # bắt đầu Điều mới
            if cur and cur["text"].strip():
                arts.append(cur)
            m = _ART.match(txt)
            if m:
                cur = {"article": f"Điều {m.group(1)}", "title": m.group(2).strip()[:300], "text": ""}
            else:                                   # header lạ → vẫn mở điều, không số
                cur = {"article": "", "title": txt[:300], "text": ""}
        elif "prov-chapter" in cls or "prov-section" in cls:
            continue                                # cấu trúc, bỏ khỏi text
        elif cur is not None:                       # khoản/điểm/content → thân điều hiện tại
            cur["text"] += (" " if cur["text"] else "") + txt
    if cur and cur["text"].strip():
        arts.append(cur)
    return [a for a in arts if a["article"] and a["text"]]


def is_in_force(eff_status: str) -> bool:
    return EFF_DEAD not in (eff_status or "").strip().lower()   # bỏ chỉ khi hết hiệu lực toàn bộ


if __name__ == "__main__":   # test nhanh
    for did in ("139877", "100024"):
        d = fetch_doc(did)
        if not d:
            print(did, "FETCH FAIL"); continue
        arts = parse_articles(d["content"])
        print(f"\nid {did} | {d['docNum']} | {d['effStatus']} | in_force={is_in_force(d['effStatus'])}")
        print(f"  parsed {len(arts)} điều")
        for a in arts[:2]:
            print(f"    [{a['article']}] {a['title'][:40]} | text {len(a['text'])} chars: {a['text'][:90]}...")
