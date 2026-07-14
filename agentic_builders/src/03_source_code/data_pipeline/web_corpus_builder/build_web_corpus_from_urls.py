import argparse
import hashlib
import html
import json
import re
import time
import unicodedata
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


DEFAULT_URLS_PATH = Path(__file__).with_name("manual_source_urls.txt")
DEFAULT_OUT_PATH = Path(__file__).with_name("web_corpus_articles.jsonl")
DEFAULT_REPORT_PATH = Path(__file__).with_name("web_corpus_report.json")
DEFAULT_RAW_DIR = Path(__file__).with_name("raw_pages")


BLOCK_TAGS = {
    "article", "aside", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "main", "p", "section", "table", "td", "th", "tr", "ul", "ol",
}


ARTICLE_RE = re.compile(
    r"(?im)^\s*(Điều\s+(\d+[a-zA-Z]?)\s*[.:]\s*[^\n]*)\n"
)
DOC_CODE_PATTERNS = [
    re.compile(r"\b\d{1,4}/\d{4}/QH\d+\b", re.IGNORECASE),
    re.compile(r"\b\d{1,4}/\d{4}/NĐ-CP\b", re.IGNORECASE),
    re.compile(r"\b\d{1,4}/\d{4}/ND-CP\b", re.IGNORECASE),
    re.compile(r"\b\d{1,4}/\d{4}/TT-[A-ZĐ]+\b", re.IGNORECASE),
    re.compile(r"\b\d{1,4}/\d{4}/TT-B[A-ZĐ]+\b", re.IGNORECASE),
    re.compile(r"\b\d{1,4}/\d{4}/QĐ-[A-ZĐ]+\b", re.IGNORECASE),
    re.compile(r"\b\d{1,4}/\d{4}/QD-[A-ZĐ]+\b", re.IGNORECASE),
]


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title += (" " + text)
        self.parts.append(text)
        self.parts.append(" ")

    def get_text(self):
        text = " ".join(self.parts)
        text = html.unescape(text)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def strip_accents(text):
    decomposed = unicodedata.normalize("NFD", str(text or ""))
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").replace("đ", "d").replace("Đ", "D")


def stable_id(*parts):
    raw = "||".join(str(p or "") for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def fetch_url(url, timeout=30):
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; legal-corpus-builder/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        body = resp.read()
    return body.decode(charset, errors="replace")


def html_to_text(raw_html):
    parser = TextExtractor()
    parser.feed(raw_html)
    return parser.title.strip(), parser.get_text()


def legacy_doc_to_text(path):
    raw = path.read_bytes()
    text = raw.decode("utf-16le", errors="ignore")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", "\n", text)
    text = re.sub(r"[\uF000-\uFFFF]+", "\n", text)
    text = re.sub(r"[ \t\r]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    start_match = re.search(r"(BỘ|CHÍNH PHỦ|QUỐC HỘI|THÔNG TƯ|NGHỊ ĐỊNH|LUẬT)", text)
    if start_match:
        text = text[start_match.start():]
    return text.strip()


def docx_to_text(path):
    paragraphs = []
    with zipfile.ZipFile(path) as zf:
        xml_names = [
            name for name in zf.namelist()
            if name.endswith(".xml")
            and (name == "word/document.xml" or name.startswith("word/header") or name.startswith("word/footer"))
        ]
        if "word/document.xml" not in xml_names and "word/document.xml" in zf.namelist():
            xml_names.insert(0, "word/document.xml")
        for name in xml_names:
            root = ET.fromstring(zf.read(name))
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            for para in root.findall(".//w:p", ns):
                text = "".join(node.text or "" for node in para.findall(".//w:t", ns)).strip()
                if text:
                    paragraphs.append(text)
    text = "\n".join(paragraphs)
    text = re.sub(r"[ \t\r]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_doc_code(text):
    for pattern in DOC_CODE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).replace("ND-CP", "NĐ-CP").replace("QD-", "QĐ-")
    law_number = re.search(r"Luật\s+số\s*[: ]\s*(\d{1,4}/\d{4}/QH\d+)", text, re.IGNORECASE)
    if law_number:
        return law_number.group(1)
    return ""


def extract_doc_code_from_filename(path):
    stem = path.stem
    patterns = [
        (re.compile(r"(\d{1,4})_(\d{4})_(QH\d+)", re.IGNORECASE), "group3"),
        (re.compile(r"(\d{1,4})_(\d{4})_NĐ-CP", re.IGNORECASE), "NĐ-CP"),
        (re.compile(r"(\d{1,4})_(\d{4})_ND-CP", re.IGNORECASE), "NĐ-CP"),
        (re.compile(r"(\d{1,4})_(\d{4})_TT-B([A-ZĐ]+)", re.IGNORECASE), "TT-B"),
        (re.compile(r"(\d{1,4})_(\d{4})_TT-([A-ZĐ]+)", re.IGNORECASE), "TT-"),
        (re.compile(r"(\d{1,4})_(\d{4})_QĐ-([A-ZĐ]+)", re.IGNORECASE), "QĐ-"),
        (re.compile(r"(\d{1,4})_QĐ-TTg", re.IGNORECASE), "QĐ-TTg"),
    ]
    matches = []
    for pattern, kind in patterns:
        for match in pattern.finditer(stem):
            matches.append((match.start(), match, kind))
    for _, match, kind in sorted(matches, key=lambda item: item[0]):
        if kind == "TT-B":
            return f"{match.group(1)}/{match.group(2)}/TT-B{match.group(3).upper()}"
        if kind == "TT-":
            return f"{match.group(1)}/{match.group(2)}/TT-{match.group(3).upper()}"
        if kind == "QĐ-":
            return f"{match.group(1)}/{match.group(2)}/QĐ-{match.group(3).upper()}"
        if kind == "group3":
            return f"{match.group(1)}/{match.group(2)}/{match.group(3).upper()}"
        if kind == "QĐ-TTg":
            return f"{match.group(1)}/QĐ-TTg"
        return f"{match.group(1)}/{match.group(2)}/{kind}"
    return ""


def extract_doc_title(page_title, text):
    candidates = []
    for line in text.splitlines():
        line = line.strip()
        if 8 <= len(line) <= 180:
            candidates.append(line)

    for line in candidates:
        if re.match(r"^Luật\s+số\s*:", line, re.IGNORECASE):
            continue
        if re.search(r"^(LUẬT|NGHỊ ĐỊNH|THÔNG TƯ|QUYẾT ĐỊNH)\b", line, re.IGNORECASE):
            return line.title() if line.isupper() else line

    title = re.sub(r"\s+", " ", page_title or "").strip()
    title = re.sub(r"\s*-\s*THƯ VIỆN PHÁP LUẬT.*$", "", title, flags=re.IGNORECASE)
    return title[:180]


def extract_dates(text):
    issued = ""
    effective = ""
    issued_match = re.search(r"Ngày ban hành\s*[: ]\s*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
    effective_match = re.search(r"Ngày hiệu lực\s*[: ]\s*([^\n]{1,40})", text, re.IGNORECASE)
    if issued_match:
        issued = issued_match.group(1)
    if effective_match:
        effective = effective_match.group(1).strip()
    return issued, effective


def infer_topic(doc_title, text):
    probe = strip_accents(f"{doc_title} {text[:2000]}").lower()
    if "dau thau" in probe or "mua sam" in probe or "cptpp" in probe or "evfta" in probe or "ukvfta" in probe:
        return "Đấu thầu, mua sắm theo hiệp định"
    if "doanh nghiep nho va vua" in probe or "sme" in probe:
        return "Doanh nghiệp nhỏ và vừa"
    if "thue" in probe or "hoa don" in probe:
        return "Thuế"
    if "lao dong" in probe or "bao hiem xa hoi" in probe:
        return "Lao động, bảo hiểm xã hội"
    if "ke toan" in probe:
        return "Kế toán"
    if "dau thau" in probe:
        return "Đấu thầu"
    if "so huu tri tue" in probe:
        return "Sở hữu trí tuệ"
    return "Nguồn web bổ sung"


def split_articles(text):
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return []

    articles = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_title = match.group(1).strip()
        article_no = match.group(2).strip()
        content_block = text[start:end].strip()
        content = content_block[len(article_title):].strip(" \n.:")
        content = re.sub(r"\n{2,}", "\n", content).strip()
        if len(content) < 20:
            continue
        articles.append(
            {
                "article": f"Điều {article_no}",
                "article_title": article_title,
                "content": content,
            }
        )
    return articles


def build_bm25_search(row):
    parts = [
        f"{row.get('doc_code', '')} {row.get('article', '')} {row.get('article_title', '')} {row.get('subject_title', '')}",
        f"Mã văn bản: {row.get('doc_code', '')}",
        f"Tên văn bản: {row.get('subject_title', '')}",
        f"Điều thật: {row.get('article', '')}",
        f"Tiêu đề điều: {row.get('article_title', '')}",
        f"Lĩnh vực: {row.get('topic_title', '')}",
        f"Chủ đề: {row.get('topic_title', '')}",
        f"Nguồn văn bản: {row.get('source_note', '')}",
        f"Nội dung điều luật: {row.get('content', '')}",
    ]
    return "\n".join(p for p in parts if p and p.strip())


def convert_url_to_rows(url):
    raw_html = fetch_url(url)
    return convert_document_text(url, raw_html)


def convert_document_text(url, raw_html_or_text, doc_code_hint=""):
    page_title, text = html_to_text(raw_html_or_text)
    if not text:
        text = raw_html_or_text
    doc_code = doc_code_hint or extract_doc_code(text)
    doc_title = extract_doc_title(page_title, text)
    issued_date, effective_date = extract_dates(text)
    topic_title = infer_topic(doc_title, text)
    source_host = urlparse(url).netloc
    articles = split_articles(text)

    rows = []
    doc_id = stable_id(url, doc_code, doc_title)
    for item in articles:
        article_no = item["article"]
        row = {
            "doc_id": f"web_{doc_id}_{stable_id(article_no)[:10]}",
            "subject_id": f"web_{stable_id(doc_code or doc_title)[:16]}",
            "topic_id": f"web_{stable_id(topic_title)[:16]}",
            "topic_title": topic_title,
            "subject_title": doc_title,
            "chapter_title": "",
            "article": article_no,
            "article_title": item["article_title"],
            "content": item["content"],
            "source_note": f"{article_no} {doc_title}" + (f" số {doc_code}" if doc_code else ""),
            "related_note": "",
            "source_links": [url],
            "source": source_host,
            "doc_code": doc_code,
            "issued_date": issued_date,
            "effective_date": effective_date,
        }
        row["bm25_search"] = build_bm25_search(row)
        rows.append(row)

    return rows, {
        "url": url,
        "source": source_host,
        "doc_code": doc_code,
        "doc_title": doc_title,
        "topic_title": topic_title,
        "issued_date": issued_date,
        "effective_date": effective_date,
        "article_count": len(rows),
    }


def convert_raw_file_to_rows(path):
    if path.suffix.lower() == ".doc":
        raw = legacy_doc_to_text(path)
        url = path.as_uri()
        rows, item_report = convert_document_text(url, raw, doc_code_hint=extract_doc_code_from_filename(path))
    elif path.suffix.lower() == ".docx":
        raw = docx_to_text(path)
        url = path.as_uri()
        rows, item_report = convert_document_text(url, raw, doc_code_hint=extract_doc_code_from_filename(path))
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")
        first_line = raw.splitlines()[0].strip() if raw.splitlines() else ""
        url = first_line[5:].strip() if first_line.lower().startswith("url: ") else path.as_uri()
        rows, item_report = convert_document_text(url, raw, doc_code_hint=extract_doc_code_from_filename(path))
    item_report["raw_file"] = str(path)
    return rows, item_report


def read_urls(path):
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", type=Path, default=DEFAULT_URLS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    urls = read_urls(args.urls)
    all_rows = []
    report = {
        "urls_path": str(args.urls),
        "output_path": str(args.out),
        "items": [],
        "errors": [],
    }

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Fetching {url}")
        try:
            rows, item_report = convert_url_to_rows(url)
            all_rows.extend(rows)
            report["items"].append(item_report)
            print(f"  -> {item_report['doc_code'] or '(no code)'} | {item_report['article_count']} articles")
        except Exception as exc:
            report["errors"].append({"url": url, "error": repr(exc)})
            print(f"  !! error: {exc!r}")
        if i < len(urls) and args.sleep > 0:
            time.sleep(args.sleep)

    if args.raw_dir.exists():
        raw_files = [
            p for p in sorted(args.raw_dir.iterdir())
            if p.is_file() and p.suffix.lower() in {".html", ".htm", ".txt", ".doc", ".docx"}
        ]
        for i, path in enumerate(raw_files, 1):
            print(f"[raw {i}/{len(raw_files)}] Parsing {path}")
            try:
                rows, item_report = convert_raw_file_to_rows(path)
                all_rows.extend(rows)
                report["items"].append(item_report)
                print(f"  -> {item_report['doc_code'] or '(no code)'} | {item_report['article_count']} articles")
            except Exception as exc:
                report["errors"].append({"raw_file": str(path), "error": repr(exc)})
                print(f"  !! error: {exc!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report["total_urls"] = len(urls)
    report["total_rows"] = len(all_rows)
    report["unique_doc_codes"] = sorted({r.get("doc_code", "") for r in all_rows if r.get("doc_code")})
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote rows: {len(all_rows)} -> {args.out}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
