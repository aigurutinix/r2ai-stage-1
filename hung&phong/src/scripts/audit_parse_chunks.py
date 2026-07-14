from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from backend.config import get_settings
from ingest.manual_docs import MANUAL, MANUAL_DIR, read_docx
from ingest.parse_vbpl import parse_vbpl_row, split_dieus

sys.stdout.reconfigure(encoding="utf-8")

_RE_HEADER = re.compile(
    r"(?<![\w/])Điều\s+(\d+)([a-zđ]?)\s*[\.\:\-–]\s+",
    re.IGNORECASE,
)


@dataclass
class RawDoc:
    so_ky_hieu: str
    title: str
    source: str
    text: str
    row: dict | None = None


def _first(row: dict, key: str) -> str:
    value = row.get(key)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value or "").strip()


def _header_numbers(text: str) -> list[int]:
    nums: list[int] = []
    for match in _RE_HEADER.finditer(text or ""):
        try:
            nums.append(int(match.group(1)))
        except ValueError:
            pass
    return nums


def _context(text: str, number: int) -> str:
    match = re.search(
        rf"Điều\s+{number}\s*[\.\:\-–]\s+",
        text or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    start = max(0, match.start() - 120)
    end = min(len(text), match.start() + 520)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def _longest_dieus(parsed_nums: list[int], dieus) -> list[tuple[int, int]]:
    pairs = [(d.dieu_so, len(d.text or "")) for d in dieus if d.dieu_so in parsed_nums]
    return sorted(pairs, key=lambda x: x[1], reverse=True)[:5]


def iter_hf_docs(limit: int | None = None) -> Iterable[RawDoc]:
    settings = get_settings()
    ds = load_dataset(
        "tmquan/vbpl-vn",
        "documents",
        split="train",
        cache_dir=settings.hf_cache_dir,
    )
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        text = row.get("markdown") or ""
        sk = _first(row, "doc_number")
        if not sk or not text:
            continue
        yield RawDoc(
            so_ky_hieu=sk,
            title=_first(row, "title"),
            source=_first(row, "source_url") or "tmquan/vbpl-vn",
            text=text,
            row=dict(row),
        )


def iter_manual_docs() -> Iterable[RawDoc]:
    for sk, meta in MANUAL.items():
        path = MANUAL_DIR / meta["file"]
        if not path.exists():
            continue
        yield RawDoc(
            so_ky_hieu=sk,
            title=meta["title"],
            source=str(path),
            text=read_docx(path),
        )


def parse_doc(raw: RawDoc):
    if raw.row is not None:
        return parse_vbpl_row(raw.row)
    return type("Parsed", (), {"dieus": split_dieus(raw.text)})()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--min-missing", type=int, default=1)
    parser.add_argument("--focus", nargs="*", default=[])
    parser.add_argument("--manual-only", action="store_true")
    args = parser.parse_args()

    docs = list(iter_manual_docs()) if args.manual_only else list(iter_hf_docs(args.limit)) + list(iter_manual_docs())
    focus = set(args.focus)
    rows = []
    total = 0
    no_parse = 0
    for raw in docs:
        if focus and raw.so_ky_hieu not in focus:
            continue
        total += 1
        header_nums = set(_header_numbers(raw.text))
        if not header_nums:
            continue
        parsed = parse_doc(raw)
        dieus = parsed.dieus if parsed else []
        parsed_nums = {d.dieu_so for d in dieus if d.dieu_so > 0}
        if not parsed_nums:
            no_parse += 1
        missing = sorted(header_nums - parsed_nums)
        extra = sorted(parsed_nums - header_nums)
        if len(missing) >= args.min_missing:
            rows.append(
                {
                    "so_ky_hieu": raw.so_ky_hieu,
                    "title": raw.title,
                    "source": raw.source,
                    "header_count": len(header_nums),
                    "parsed_count": len(parsed_nums),
                    "missing_count": len(missing),
                    "missing": missing,
                    "extra": extra[:15],
                    "longest": _longest_dieus(list(parsed_nums), dieus),
                    "context": _context(raw.text, missing[0]) if missing else "",
                }
            )

    print(f"docs_checked={total} docs_with_no_parsed_article={no_parse} docs_with_missing={len(rows)}")
    bucket = Counter(min(r["missing_count"], 20) for r in rows)
    print("missing_count_bucket", dict(sorted(bucket.items())))
    for row in sorted(rows, key=lambda r: (r["missing_count"], r["header_count"]), reverse=True)[: args.top]:
        print("\n---")
        print(
            f"{row['so_ky_hieu']} | missing={row['missing_count']} "
            f"| headers={row['header_count']} parsed={row['parsed_count']}"
        )
        print(row["title"][:220])
        print("missing:", row["missing"][:80])
        if row["extra"]:
            print("extra:", row["extra"])
        print("longest:", row["longest"])
        print("context:", row["context"][:650])
        print("source:", row["source"])


if __name__ == "__main__":
    main()
