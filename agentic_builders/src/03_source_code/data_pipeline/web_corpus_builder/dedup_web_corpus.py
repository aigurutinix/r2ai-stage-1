import json
from collections import Counter
from pathlib import Path


IN_PATH = Path(__file__).with_name("web_corpus_articles.jsonl")
OUT_PATH = Path(__file__).with_name("web_corpus_articles_dedup.jsonl")
REPORT_PATH = Path(__file__).with_name("web_corpus_dedup_report.json")


def load_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_quality(row):
    return (
        len(row.get("content", "")),
        len(row.get("article_title", "")),
        len(row.get("bm25_search", "")),
    )


def main():
    rows = load_rows(IN_PATH)
    best_by_key = {}
    duplicates = Counter()

    for row in rows:
        key = (row.get("doc_code", ""), row.get("article", ""))
        if not key[0] or not key[1]:
            key = (row.get("doc_id", ""), row.get("article_title", ""))
        if key in best_by_key:
            duplicates[key] += 1
            if row_quality(row) > row_quality(best_by_key[key]):
                best_by_key[key] = row
        else:
            best_by_key[key] = row

    deduped = list(best_by_key.values())
    deduped.sort(key=lambda r: (r.get("doc_code", ""), r.get("article", ""), r.get("article_title", "")))

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_doc = Counter(row.get("doc_code", "") for row in deduped)
    report = {
        "input_rows": len(rows),
        "output_rows": len(deduped),
        "removed_duplicate_rows": len(rows) - len(deduped),
        "doc_counts": dict(by_doc.most_common()),
        "duplicate_keys": [
            {"doc_code": key[0], "article": key[1], "extra_rows": count}
            for key, count in duplicates.most_common(50)
        ],
    }
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
