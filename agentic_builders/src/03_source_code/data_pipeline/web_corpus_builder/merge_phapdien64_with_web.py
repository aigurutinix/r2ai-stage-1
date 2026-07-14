import json
import re
from pathlib import Path


BASE_PATH = Path("articles.jsonl")
WEB_PATH = Path("web_corpus_builder/web_corpus_articles_dedup.jsonl")
OUT_PATH = Path("articles_phapdien64_web_bm25.jsonl")
REPORT_PATH = Path("web_corpus_builder/merge_phapdien64_with_web_report.json")

DOC_CODE_RE = re.compile(
    r"\b\d{1,4}/\d{4}/(?:QH\d+|NĐ-CP|ND-CP|TT-[A-ZĐ]+|TT-B[A-ZĐ]+|TTLT-[A-ZĐ-]+|QĐ-[A-ZĐ]+|QD-[A-ZĐ]+|VBHN-[A-ZĐ]+)\b",
    re.IGNORECASE,
)
ARTICLE_RE = re.compile(r"(?:Điều|Dieu)\s+(\d+[a-zA-Z]?)", re.IGNORECASE)


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalize_code(code):
    return (
        str(code or "")
        .upper()
        .replace("ND-CP", "NĐ-CP")
        .replace("QD-", "QĐ-")
        .replace("QĐ-TTG", "QĐ-TTg")
    )


def extract_doc_code(row):
    if row.get("doc_code"):
        return normalize_code(row["doc_code"])
    text = "\n".join(str(row.get(k, "")) for k in ("source_note", "source", "article_title", "subject_title"))
    match = DOC_CODE_RE.search(text)
    return normalize_code(match.group(0)) if match else ""


def extract_real_article(row):
    if str(row.get("article", "")).strip().lower().startswith("điều "):
        return row["article"].strip()
    text = " ".join(str(row.get(k, "")) for k in ("source_note", "article_title"))
    match = ARTICLE_RE.search(text)
    return f"Điều {match.group(1)}" if match else ""


def build_bm25_search(row):
    doc_code = row.get("doc_code") or extract_doc_code(row)
    real_article = row.get("real_article") or extract_real_article(row)
    subject_title = row.get("subject_title", "")
    topic_title = row.get("topic_title", "")
    chapter_title = row.get("chapter_title", "")
    article = row.get("article", "")
    article_title = row.get("article_title", "")
    source_note = row.get("source_note", "")
    content = row.get("content", "")
    source = row.get("source", "")

    parts = [
        f"{doc_code} {real_article} {article_title} {subject_title} {topic_title}",
        f"Mã văn bản: {doc_code}",
        f"Điều thật: {real_article}",
        f"Tên văn bản: {subject_title}",
        f"Tiêu đề điều: {article_title}",
        f"Lĩnh vực: {subject_title}",
        f"Chủ đề: {topic_title}",
        f"Chương: {chapter_title}",
        f"Nguồn văn bản: {source_note}",
        f"Nguồn dữ liệu: {source}",
        f"Nội dung điều luật: {content}",
    ]
    return "\n".join(p for p in parts if p and p.strip())


def normalize_row(row, source_override=None):
    row = dict(row)
    row["doc_code"] = extract_doc_code(row)
    row["real_article"] = extract_real_article(row)
    if source_override:
        row["source"] = source_override
    row["source"] = row.get("source") or "phapdien"
    row["topic_title"] = row.get("topic_title") or ""
    row["subject_title"] = row.get("subject_title") or ""
    row["chapter_title"] = row.get("chapter_title") or ""
    row["related_note"] = row.get("related_note") or ""
    row["source_links"] = row.get("source_links") or []
    row["bm25_search"] = build_bm25_search(row)
    return row


def main():
    base_count = 0
    web_count = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for row in iter_jsonl(BASE_PATH):
            out.write(json.dumps(normalize_row(row), ensure_ascii=False) + "\n")
            base_count += 1
        for row in iter_jsonl(WEB_PATH):
            out.write(json.dumps(normalize_row(row, source_override="web_manual"), ensure_ascii=False) + "\n")
            web_count += 1

    report = {
        "base_path": str(BASE_PATH),
        "web_path": str(WEB_PATH),
        "out_path": str(OUT_PATH),
        "base_rows": base_count,
        "web_rows": web_count,
        "total_rows": base_count + web_count,
    }
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
