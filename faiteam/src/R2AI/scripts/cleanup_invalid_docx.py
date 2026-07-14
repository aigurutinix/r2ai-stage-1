#!/usr/bin/env python3
"""Remove docx files missing BOTH law code and title from crawl output folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "FisReader" / "src"))

from FisReader.document_factory import DocumentFactory

DEFAULT_OUTPUT_ROOT = ROOT / "VbplCrawler" / "output"


def has_valid_title(title: str) -> bool:
    if not title:
        return False
    stripped = title.strip()
    if not stripped or all(c in "-_ " for c in stripped):
        return False
    lowered = stripped.lower()
    return not (lowered.startswith("căn cứ") or lowered.startswith("theo "))


def has_valid_code(code: str) -> bool:
    return bool((code or "").strip())


def should_delete(doc) -> bool:
    return not has_valid_code(doc.code) and not has_valid_title(doc.title)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete docx files without both law code and title"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent folder containing *-docx directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print files that would be deleted",
    )
    args = parser.parse_args()

    factory = DocumentFactory()
    docx_dirs = sorted(args.output_root.glob("*-docx"))
    if not docx_dirs:
        raise SystemExit(f"No *-docx folders under {args.output_root}")

    total = removed = errors = 0
    for docx_dir in docx_dirs:
        files = sorted(docx_dir.glob("*.docx"))
        dir_removed = 0
        print(f"\n=== {docx_dir.name}: {len(files)} files ===")
        for fpath in files:
            total += 1
            try:
                doc = factory.read(fpath, doc_id=0, for_llm=False)
                if should_delete(doc):
                    print(f"  {'DRY-RUN' if args.dry_run else 'XÓA'}: {fpath.name}")
                    if not args.dry_run:
                        fpath.unlink()
                    removed += 1
                    dir_removed += 1
            except Exception as exc:
                errors += 1
                print(f"  LỖI: {fpath.name}: {exc}")
        print(f"  => {'Sẽ xóa' if args.dry_run else 'Xóa'} {dir_removed} file")

    print(f"\nTổng kết: {total} file, {'sẽ xóa' if args.dry_run else 'xóa'} {removed}, lỗi {errors}")


if __name__ == "__main__":
    main()
