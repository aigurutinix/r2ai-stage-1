#!/usr/bin/env python3
"""Re-download docx files that were wrongly deleted from vbpl.vn crawl output."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "VbplCrawler" / "src"))

from VbplCrawler import VbplCrawler
from VbplCrawler.html_to_docx import html_to_docx, safe_filename

DELETE_LOG = (
    Path.home()
    / ".cursor/projects/home-buncha-Document-ROAD2AI/terminals/228075.txt"
)
OUTPUT_ROOT = ROOT / "VbplCrawler" / "output"


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


def should_have_been_deleted(code: str, title: str) -> bool:
    return not has_valid_code(code) and not has_valid_title(title)


def parse_wrong_deletions(log_path: Path) -> list[tuple[str, Path]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    current_dir: Path | None = None
    wrong: list[tuple[str, Path]] = []

    for line in text.splitlines():
        section = re.match(r"^=== (.+-docx): \d+ files ===$", line)
        if section:
            current_dir = OUTPUT_ROOT / section.group(1)
            continue
        delete = re.match(r"^  XÓA: (.+\.docx)$", line)
        if delete and current_dir is not None:
            wrong.append((delete.group(1), current_dir))
            continue
        meta = re.match(r"^       code=(.+), title=(.+)$", line)
        if meta and wrong:
            name, folder = wrong[-1]
            code = eval(meta.group(1))
            title = eval(meta.group(2))
            if should_have_been_deleted(code, title):
                wrong.pop()
    return wrong


def build_doc_id_index(log_dir: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Map suffix/filename -> doc_id, plus (doc_num_key, doc_id) for fuzzy lookup."""
    index: dict[str, str] = {}
    by_doc_num: list[tuple[str, str]] = []
    pattern = re.compile(r"\]\s+(\S+)\s+-\s+(.+)")

    for log_file in log_dir.glob("*.log"):
        for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.search(line)
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

            num_match = re.search(
                r"(\d{1,3}/\d{4}/(?:QH\d+|NĐ-CP|TT-[A-ZĐ]+|QĐ-[A-Z]+|NQ-HĐND|PL-UBTVQH\d+))",
                title,
                re.IGNORECASE,
            )
            if num_match:
                doc_num = num_match.group(1)
                by_doc_num.append((doc_num.replace("/", "_"), doc_id))
                by_doc_num.append((doc_num, doc_id))

    ingest_log = log_dir / "ingest-doanh-nghiep.log"
    if ingest_log.exists():
        for line in ingest_log.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"\[\d+/\d+\]\s+(.+\.docx)", line)
            if not match:
                continue
            filename = match.group(1)
            suffix = filename.rsplit("_", 1)[-1].removesuffix(".docx")
            if suffix in index:
                index[filename] = index[suffix]
            else:
                index.setdefault(suffix, suffix)
    return index, by_doc_num


def resolve_doc_id(
    filename: str,
    index: dict[str, str],
    by_doc_num: list[tuple[str, str]],
) -> str | None:
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


def restore_one(
    crawler: VbplCrawler,
    *,
    filename: str,
    output_dir: Path,
    doc_id: str,
    delay: float,
) -> bool:
    out_path = output_dir / filename
    if out_path.exists():
        return True

    time.sleep(delay)
    metadata = crawler.get_document(doc_id)
    html = (metadata.get("documentContent") or {}).get("content")
    if not html:
        raise RuntimeError("Không có nội dung HTML")

    title = metadata.get("title") or ""
    doc_num = metadata.get("docNum") or ""
    expected_name = f"{safe_filename(f'{doc_num}_{title}_{doc_id}')}.docx"
    target = output_dir / expected_name
    html_to_docx(html, target, title=title)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete-log", type=Path, default=DELETE_LOG)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    wrong = parse_wrong_deletions(args.delete_log)
    index, by_doc_num = build_doc_id_index(OUTPUT_ROOT)
    crawler = VbplCrawler(request_delay=args.delay)

    restored = skipped = failed = unresolved = 0
    for i, (filename, folder) in enumerate(wrong, start=1):
        if args.limit is not None and i > args.limit:
            break
        out_path = folder / filename
        if out_path.exists():
            skipped += 1
            continue
        doc_id = resolve_doc_id(filename, index, by_doc_num)
        if not doc_id:
            unresolved += 1
            print(f"[{i}/{len(wrong)}] KHÔNG TÌM THẤY doc_id: {filename}")
            continue
        try:
            restore_one(
                crawler,
                filename=filename,
                output_dir=folder,
                doc_id=doc_id,
                delay=args.delay,
            )
            restored += 1
            print(f"[{i}/{len(wrong)}] OK {doc_id} -> {folder.name}/{filename[:70]}")
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(wrong)}] LỖI {doc_id} ({filename[:60]}): {exc}")

    print(
        f"\nHoàn tất: khôi phục {restored}, bỏ qua {skipped}, "
        f"không resolve doc_id {unresolved}, lỗi {failed}"
    )


if __name__ == "__main__":
    main()
