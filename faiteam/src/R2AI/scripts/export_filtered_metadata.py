#!/usr/bin/env python3
"""Export metadata filtered by title keywords and effectiveness cutoff."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from document_filters import (
    DEFAULT_CUTOFF,
    filter_by_title,
    filter_metadata_effective,
    load_effectiveness_table,
    load_keywords,
    parse_metadata_date,
)
from ingest_parquet_to_qdrant import resolve_data_dir

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "filtered_by_keywords_effective.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export filtered legal document metadata")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--effectiveness-path",
        type=Path,
        default=DEFAULT_DATA_DIR / "effectiveness.parquet",
    )
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF.isoformat())
    parser.add_argument("--keywords-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--unknown-policy",
        choices=("include", "exclude"),
        default="include",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cutoff = parse_metadata_date(args.cutoff) or DEFAULT_CUTOFF
    keywords = load_keywords(str(args.keywords_file) if args.keywords_file else None)
    data_dir = resolve_data_dir(args.data_dir)

    print(f"Data dir: {data_dir}")
    print(f"Cutoff date: {cutoff.isoformat()}")

    meta = pd.read_parquet(data_dir / "metadata.parquet")
    print(f"Tổng metadata: {len(meta):,}")

    by_title = filter_by_title(meta, keywords)
    print(f"Sau lọc title: {len(by_title):,}")

    eff_table = load_effectiveness_table(str(args.effectiveness_path))
    if eff_table is not None:
        print(f"Effectiveness sidecar: {len(eff_table):,} rows")
    else:
        print("Effectiveness sidecar: không có — dùng heuristic metadata/title")

    filtered = filter_metadata_effective(
        by_title,
        cutoff=cutoff,
        effectiveness_table=eff_table,
        unknown_policy=args.unknown_policy,
    )
    print(f"Sau lọc hiệu lực: {len(filtered):,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(args.output, index=False)
    print(f"Đã lưu: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
