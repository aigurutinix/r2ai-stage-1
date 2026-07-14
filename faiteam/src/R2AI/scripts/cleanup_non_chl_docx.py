#!/usr/bin/env python3
"""Remove crawled docx files whose legal effect status is not 'Còn hiệu lực'."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "VbplCrawler" / "src"))

from VbplCrawler import VbplCrawler

DEFAULT_OUTPUT_ROOT = ROOT / "VbplCrawler" / "output"
DOC_ID_SUFFIX = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|\d+)\.docx$",
    re.IGNORECASE,
)
CRAWL_LINE = re.compile(r"\]\s+(\S+)\s+-\s+(.+)")


def _normalize_title(title: str) -> str:
    if not title:
        return ""
    try:
        return title.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return title


def is_still_effective(metadata: dict) -> bool:
    status = metadata.get("effStatus") or {}
    code = (status.get("code") or "").upper()
    name = _normalize_title(status.get("name") or "").lower()
    return code == "CHL" or name == "còn hiệu lực"


def build_doc_id_index(log_dir: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    index: dict[str, str] = {}
    by_doc_num: list[tuple[str, str]] = []

    for log_file in sorted(log_dir.glob("*.log")):
        for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
            match = CRAWL_LINE.search(line)
            if not match:
                continue
            doc_id, title = match.group(1), match.group(2)
            index[doc_id] = doc_id
            if "-" in doc_id:
                for part in doc_id.split("-"):
                    if len(part) >= 8:
                        index.setdefault(part[:13], doc_id)
            else:
                index.setdefault(doc_id, doc_id)

            normalized = _normalize_title(title)
            num_match = re.search(
                r"(\d{1,3}/\d{4}/(?:QH\d+|NĐ-CP|TT-[A-ZĐ]+|QĐ-[A-Z]+|NQ-HĐND|PL-UBTVQH\d+))",
                normalized,
                re.IGNORECASE,
            )
            if num_match:
                doc_num = num_match.group(1)
                by_doc_num.append((doc_num.replace("/", "_"), doc_id))
                by_doc_num.append((doc_num, doc_id))

    return index, by_doc_num


def resolve_doc_id(
    filename: str,
    index: dict[str, str],
    by_doc_num: list[tuple[str, str]],
) -> str | None:
    suffix_match = DOC_ID_SUFFIX.search(filename)
    if suffix_match:
        return suffix_match.group(1)

    if filename in index:
        return index[filename]
    suffix = filename.rsplit("_", 1)[-1].removesuffix(".docx")
    if suffix in index:
        return index[suffix]
    if re.fullmatch(r"\d+", suffix):
        return suffix
    if "-" in suffix:
        for key, doc_id in index.items():
            if isinstance(key, str) and doc_id.startswith(suffix):
                return doc_id

    stem = filename.removesuffix(".docx")
    for doc_num_key, doc_id in by_doc_num:
        if doc_num_key and doc_num_key in stem:
            return doc_id
    head = re.match(r"^([\d]+[_-][\d]{4}[_-][A-Za-z0-9ĐƯƠĂÂÊ-]+)", stem)
    if head:
        key = head.group(1).replace("-", "_")
        for doc_num_key, doc_id in by_doc_num:
            if doc_num_key.replace("-", "_") == key:
                return doc_id
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete crawled docx files that are not 'Còn hiệu lực'"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent folder containing *-docx directories",
    )
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report files that would be deleted",
    )
    args = parser.parse_args()

    docx_dirs = sorted(args.output_root.glob("*-docx"))
    if not docx_dirs:
        raise SystemExit(f"No *-docx folders under {args.output_root}")

    index, by_doc_num = build_doc_id_index(args.output_root)
    crawler = VbplCrawler(request_delay=args.delay)
    status_cache: dict[str, tuple[bool, str, str]] = {}

    total = kept = removed = unresolved = errors = 0
    status_counts: dict[tuple[str, str], int] = {}

    for docx_dir in docx_dirs:
        files = sorted(docx_dir.glob("*.docx"))
        dir_removed = 0
        print(f"\n=== {docx_dir.name}: {len(files)} files ===")

        for fpath in files:
            total += 1
            doc_id = resolve_doc_id(fpath.name, index, by_doc_num)
            if not doc_id:
                unresolved += 1
                print(f"  KHÔNG RESOLVE doc_id: {fpath.name}")
                continue

            if doc_id not in status_cache:
                try:
                    metadata = crawler.get_document(doc_id)
                    status = metadata.get("effStatus") or {}
                    code = status.get("code") or ""
                    name = _normalize_title(status.get("name") or "")
                    ok = is_still_effective(metadata)
                    status_cache[doc_id] = (ok, code, name)
                    status_counts[(code, name)] = status_counts.get((code, name), 0) + 1
                    time.sleep(args.delay)
                except Exception as exc:
                    errors += 1
                    print(f"  LỖI API {doc_id} ({fpath.name[:60]}): {exc}")
                    continue

            ok, code, name = status_cache[doc_id]
            if ok:
                kept += 1
                continue

            removed += 1
            dir_removed += 1
            action = "DRY-RUN" if args.dry_run else "XÓA"
            print(f"  {action} [{code or '?'} / {name or '?'}]: {fpath.name}")

            if not args.dry_run:
                fpath.unlink()

        print(
            f"  => {'Sẽ xóa' if args.dry_run else 'Xóa'} {dir_removed} file "
            f"trong {docx_dir.name}"
        )

    print("\n=== Trạng thái hiệu lực (theo API) ===")
    for (code, name), count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}  {code or '?'} / {name or '?'}")

    print(
        f"\nTổng kết: {total} file, giữ {kept}, "
        f"{'sẽ xóa' if args.dry_run else 'xóa'} {removed}, "
        f"không resolve doc_id {unresolved}, lỗi API {errors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
