import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("articles_phapdien64_web_bm25.jsonl")
DEFAULT_OUTPUT = Path("articles_phapdien64_web_canonical.jsonl")
DEFAULT_REGISTRY = Path("web_corpus_builder/legal_doc_registry.json")
DEFAULT_REPORT = Path("audit/canonical_corpus_report.json")


DOC_CODE_RE = re.compile(
    r"\b\d{1,4}/\d{4}/(?:QH\d+|PL-UBTVQH\d+|NQ-UBTVQH\d+|NĐ-CP|ND-CP|NQ-CP|TT-[A-ZĐ]+|TT-B[A-ZĐ]+|TTLT-[A-ZĐ-]+|QĐ-[A-ZĐ]+|QD-[A-ZĐ]+|VBHN-[A-ZĐ]+)\b",
    re.IGNORECASE,
)
ARTICLE_RE = re.compile(r"(?:Điều|Dieu)\s+(\d+[a-zA-ZđĐ]?)", re.IGNORECASE)
BAD_TITLE_RE = re.compile(
    r"\s+(?:áp dụng(?:\s+năm)?\s+20\d{2}|áp dụng\s+20\d{2}|mới nhất|năm\s+20\d{2}\s+mới nhất)\b.*$",
    re.IGNORECASE,
)


def normalize_code(code: str) -> str:
    code = unicodedata.normalize("NFC", str(code or "").strip())
    code = re.sub(r"\s+", "", code)
    code = code.upper()
    code = code.replace("ND-CP", "NĐ-CP")
    code = code.replace("QD-", "QĐ-")
    code = code.replace("TT-BKHDT", "TT-BKHĐT")
    code = code.replace("QĐ-TTG", "QĐ-TTg")
    return code


def clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def clean_title(title: str, code: str = "") -> str:
    title = clean_ws(title)
    if not title:
        return ""
    title = BAD_TITLE_RE.sub("", title).strip()
    if code:
        title = re.sub(rf"\s+số\s+{re.escape(code)}\b", "", title, flags=re.IGNORECASE).strip()
    title = re.sub(r"\s*-\s*THƯ VIỆN PHÁP LUẬT.*$", "", title, flags=re.IGNORECASE).strip()
    return clean_ws(title)


def load_registry(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    docs = {}
    for item in raw.get("docs", []):
        code = normalize_code(item.get("doc_code"))
        if not code:
            continue
        docs[code] = {
            "doc_code": code,
            "canonical_title": clean_ws(item.get("canonical_title")),
            "domain": item.get("domain", ""),
            "priority": item.get("priority", ""),
            "aliases": [clean_ws(x) for x in item.get("aliases", []) if clean_ws(x)],
        }
    return docs


def extract_doc_code(row: dict) -> str:
    for key in ("doc_code", "code"):
        code = normalize_code(row.get(key))
        if code:
            return code

    text = "\n".join(
        str(row.get(k, ""))
        for k in ("source_note", "source", "source_links", "article_title", "subject_title", "doc_name", "bm25_search")
    )
    match = DOC_CODE_RE.search(text)
    return normalize_code(match.group(0)) if match else ""


def extract_article_ref(row: dict) -> str:
    for key in ("article_ref", "real_article", "article"):
        value = clean_ws(row.get(key))
        match = ARTICLE_RE.search(value)
        if match:
            return f"Điều {match.group(1)}"

    text = " ".join(str(row.get(k, "")) for k in ("source_note", "article_title", "bm25_search"))
    match = ARTICLE_RE.search(text)
    return f"Điều {match.group(1)}" if match else ""


def choose_title(row: dict, doc_code: str, registry: dict):
    if doc_code in registry:
        return registry[doc_code]["canonical_title"], "registry"

    for key in ("doc_name", "doc_title", "subject_title", "title"):
        title = clean_title(row.get(key), doc_code)
        if title:
            return title, "cleaned"
    return "", "missing"


def build_bm25_search(row: dict) -> str:
    doc_code = row.get("doc_code", "")
    doc_name = row.get("doc_name") or row.get("subject_title", "")
    real_article = row.get("real_article") or row.get("article_ref") or ""
    article_title = row.get("article_title", "")
    topic_title = row.get("topic_title", "")
    chapter_title = row.get("chapter_title", "")
    legal_domain = row.get("legal_domain", "")
    aliases = ", ".join(row.get("doc_aliases", [])[:5])
    source_note = row.get("source_note", "")
    content = row.get("content", "")
    source = row.get("source", "")

    parts = [
        f"{doc_code} {real_article} {article_title} {doc_name} {topic_title} {legal_domain}",
        f"Mã văn bản: {doc_code}",
        f"Tên văn bản chuẩn: {doc_name}",
        f"Tên gọi khác: {aliases}" if aliases else "",
        f"Điều thật: {real_article}",
        f"Tiêu đề điều: {article_title}",
        f"Lĩnh vực chuẩn: {legal_domain}" if legal_domain else "",
        f"Lĩnh vực: {row.get('subject_title', '')}",
        f"Chủ đề: {topic_title}",
        f"Chương: {chapter_title}",
        f"Nguồn văn bản: {source_note}",
        f"Nguồn dữ liệu: {source}",
        f"Nội dung điều luật: {content}",
    ]
    return "\n".join(p for p in parts if clean_ws(p))


def canonicalize_row(row: dict, registry: dict):
    out = dict(row)
    doc_code = extract_doc_code(out)
    article_ref = extract_article_ref(out)
    canonical_title, title_source = choose_title(out, doc_code, registry)
    reg = registry.get(doc_code, {})

    original_subject = clean_ws(out.get("subject_title"))
    original_doc_name = clean_ws(out.get("doc_name"))

    out["doc_code"] = doc_code
    out["real_article"] = article_ref
    out["article_ref"] = article_ref
    out["doc_name"] = canonical_title
    out["doc_title"] = canonical_title
    out["subject_title"] = canonical_title or original_subject
    out["canonical_doc_key"] = f"{doc_code}|{canonical_title}" if doc_code and canonical_title else ""
    out["doc_title_source"] = title_source
    out["legal_domain"] = reg.get("domain", out.get("legal_domain", ""))
    out["doc_priority"] = reg.get("priority", out.get("doc_priority", ""))
    out["doc_aliases"] = reg.get("aliases", out.get("doc_aliases", []))

    if original_subject and original_subject != out["subject_title"]:
        out["raw_subject_title"] = original_subject
    if original_doc_name and original_doc_name != out["doc_name"]:
        out["raw_doc_name"] = original_doc_name

    out["bm25_search"] = build_bm25_search(out)
    return out


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    code_counts = Counter()
    registry_code_counts = Counter()
    title_sources = Counter()
    title_variants_before = defaultdict(Counter)
    title_variants_after = defaultdict(Counter)
    bad_title_before_rows = 0

    with args.output.open("w", encoding="utf-8") as out_f:
        for row in iter_jsonl(args.input):
            stats["rows"] += 1
            before_code = extract_doc_code(row)
            before_title = clean_ws(row.get("doc_name") or row.get("doc_title") or row.get("subject_title"))
            if before_code:
                title_variants_before[before_code][before_title] += 1
            if before_title and BAD_TITLE_RE.search(before_title):
                bad_title_before_rows += 1

            out = canonicalize_row(row, registry)
            code = out.get("doc_code", "")
            title = out.get("doc_name", "")
            title_sources[out.get("doc_title_source", "")] += 1
            if code:
                code_counts[code] += 1
                title_variants_after[code][title] += 1
            if code in registry:
                stats["registry_rows"] += 1
                registry_code_counts[code] += 1
            if out.get("article_ref"):
                stats["rows_with_article_ref"] += 1
            if out.get("canonical_doc_key"):
                stats["rows_with_canonical_doc_key"] += 1
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")

    target_doc_rows = []
    for code, meta in registry.items():
        target_doc_rows.append(
            {
                "doc_code": code,
                "canonical_title": meta["canonical_title"],
                "domain": meta.get("domain", ""),
                "priority": meta.get("priority", ""),
                "rows": code_counts.get(code, 0),
                "title_variants_before": title_variants_before.get(code, Counter()).most_common(8),
                "title_variants_after": title_variants_after.get(code, Counter()).most_common(8),
                "status": "present" if code_counts.get(code, 0) else "missing_or_unparsed",
            }
        )

    target_doc_rows.sort(key=lambda x: (x["priority"] or 99, x["domain"], x["doc_code"]))
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "registry": str(args.registry),
        "total_rows": stats["rows"],
        "registry_rows": stats["registry_rows"],
        "rows_with_article_ref": stats["rows_with_article_ref"],
        "rows_with_canonical_doc_key": stats["rows_with_canonical_doc_key"],
        "bad_title_before_rows": bad_title_before_rows,
        "title_sources": dict(title_sources),
        "missing_registry_docs": [row for row in target_doc_rows if row["status"] != "present"],
        "weak_registry_docs_under_10_rows": [
            row for row in target_doc_rows if 0 < row["rows"] < 10
        ],
        "target_docs": target_doc_rows,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Wrote canonical corpus:", args.output)
    print("Wrote report:", args.report)
    print("Rows:", stats["rows"])
    print("Registry rows:", stats["registry_rows"])
    print("Bad title rows before:", bad_title_before_rows)
    print("Missing registry docs:", len(report["missing_registry_docs"]))
    print("Weak registry docs <10 rows:", len(report["weak_registry_docs_under_10_rows"]))


if __name__ == "__main__":
    main()
