#!/usr/bin/env python3
"""Build optional effectiveness sidecar for TVPL parquet documents."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from effectiveness_sources import fetch_effectiveness
from ingest_parquet_to_qdrant import resolve_data_dir

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "effectiveness.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/effectiveness.parquet sidecar (vbpl first, VietLex fallback)",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--input-metadata", type=Path, default=None, help="Pre-filtered metadata parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between API calls")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP/SOAP timeout per request")
    parser.add_argument(
        "--skip-vbpl",
        action="store_true",
        help="Skip vbpl.vn lookup (VietLex only)",
    )
    parser.add_argument(
        "--skip-vietlex",
        action="store_true",
        help="Skip VietLex fallback",
    )
    parser.add_argument(
        "--vbpl-verify-ssl",
        action="store_true",
        help="Verify TLS certificate for ws.vbpl.vn (default: disabled due to cert mismatch)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)

    if args.input_metadata and args.input_metadata.is_file():
        meta = pd.read_parquet(args.input_metadata)
    else:
        meta = pd.read_parquet(data_dir / "metadata.parquet")

    if args.limit:
        meta = meta.head(args.limit)

    rows: list[dict] = []
    total = len(meta)
    print(f"Fetching effectiveness for {total:,} documents (vbpl → VietLex)...")

    vbpl_hits = vietlex_hits = unknown = errors = 0

    for idx, row in enumerate(meta.itertuples(index=False), start=1):
        row_dict = row._asdict()
        doc_id = int(row_dict["id"])
        document_number = str(row_dict.get("document_number") or "").strip()
        title = str(row_dict.get("title") or "").strip() or None

        record = {
            "id": doc_id,
            "document_number": document_number,
            "eff_code": "",
            "eff_status": "",
            "effective_date": "",
            "expiry_date": "",
            "source": "unknown",
            "source_id": "",
        }

        if document_number:
            try:
                fetched = fetch_effectiveness(
                    document_number,
                    title=title,
                    timeout=args.timeout,
                    verify_ssl=args.vbpl_verify_ssl,
                    use_vbpl=not args.skip_vbpl,
                    use_vietlex=not args.skip_vietlex,
                )
                record.update(
                    {
                        "eff_code": fetched.get("eff_code") or "",
                        "eff_status": fetched.get("eff_status") or "",
                        "effective_date": fetched.get("effective_date") or "",
                        "expiry_date": fetched.get("expiry_date") or "",
                        "source": fetched.get("source") or "unknown",
                        "source_id": fetched.get("source_id") or "",
                    }
                )
                if fetched.get("vbpl_matched"):
                    vbpl_hits += 1
                elif fetched.get("vietlex_matched"):
                    vietlex_hits += 1
                elif record["source"] == "unknown":
                    unknown += 1
            except Exception as exc:
                errors += 1
                print(f"  [{idx}/{total}] id={doc_id} lỗi: {exc}")

        rows.append(record)
        if idx % 50 == 0 or idx == total:
            print(f"  [{idx}/{total}] vbpl={vbpl_hits:,} vietlex={vietlex_hits:,} unknown={unknown:,}")
        time.sleep(args.sleep)

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(
        f"Đã lưu {len(out):,} rows → {args.output}\n"
        f"  vbpl: {vbpl_hits:,} | VietLex fallback: {vietlex_hits:,} | unknown: {unknown:,} | errors: {errors:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
