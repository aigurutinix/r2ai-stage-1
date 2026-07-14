#!/usr/bin/env python3
"""Collect unique law_code from Qdrant chunks and list expired ones via vbpl → VietLex."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from document_filters import EXPIRED_CODES, STILL_EFFECTIVE_CODES
from effectiveness_sources import fetch_effectiveness
from qdrant_config import (
    get_collection_name,
    init_qdrant_from_args,
    load_env,
    make_qdrant_client,
    normalize_chunk_payload,
)


def is_expired(eff_code: str, eff_status: str) -> bool:
    code = (eff_code or "").strip().upper()
    status = (eff_status or "").strip().lower()
    if code in STILL_EFFECTIVE_CODES:
        return False
    if "còn hiệu lực" in status and "một phần" not in status:
        return False
    if code in EXPIRED_CODES:
        return True
    return "hết hiệu lực" in status or "ngưng hiệu lực" in status


def collect_law_codes(client, collection: str, *, scroll_limit: int = 1000) -> dict[str, str]:
    """Return {law_code: law_title} from all chunk payloads."""
    codes: dict[str, str] = {}
    offset = None
    total = 0
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=scroll_limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = normalize_chunk_payload(point.payload or {})
            code = (payload.get("document_number") or payload.get("law_code") or "").strip()
            if not code:
                continue
            if code not in codes:
                title = str(payload.get("document_title") or payload.get("law_title") or "").strip()
                codes[code] = title
        total += len(points)
        if total % 10000 == 0 or offset is None:
            print(f"  scrolled {total:,} points, unique law_codes={len(codes):,}")
        if offset is None:
            break
    return codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=None, help="Dotenv file (e.g. .env.cloud)")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-api-key", default=None)
    parser.add_argument("--scroll-limit", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay after each completed lookup")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=16, help="Parallel effectiveness lookups")
    parser.add_argument("--limit", type=int, default=None, help="Limit law_codes to check (debug)")
    parser.add_argument(
        "--codes-cache",
        type=Path,
        default=SCRIPT_DIR.parent / "test" / "law_codes_from_chunks.json",
        help="Cache unique law_codes from Qdrant scroll",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR.parent / "test" / "expired_law_codes.json",
    )
    parser.add_argument(
        "--progress-cache",
        type=Path,
        default=SCRIPT_DIR.parent / "test" / "law_code_effectiveness_progress.json",
    )
    parser.add_argument("--skip-scroll", action="store_true", help="Use cached law_codes only")
    parser.add_argument("--skip-vbpl", action="store_true")
    parser.add_argument("--skip-vietlex", action="store_true")
    parser.add_argument("--vbpl-verify-ssl", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.env_file:
        load_env(args.env_file)
    init_qdrant_from_args(args)

    if args.skip_scroll and args.codes_cache.is_file():
        codes = json.loads(args.codes_cache.read_text(encoding="utf-8"))
        print(f"Loaded {len(codes):,} law_codes from cache {args.codes_cache}")
    else:
        client = make_qdrant_client(args.qdrant_url, args.qdrant_api_key)
        collection = get_collection_name(args.collection)
        info = client.get_collection(collection)
        print(f"Scrolling collection '{collection}' ({info.points_count:,} points)...")
        codes = collect_law_codes(client, collection, scroll_limit=args.scroll_limit)
        args.codes_cache.parent.mkdir(parents=True, exist_ok=True)
        args.codes_cache.write_text(json.dumps(codes, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(codes):,} unique law_codes → {args.codes_cache}")

    items = sorted(codes.items())
    if args.limit:
        items = items[: args.limit]

    progress: dict[str, dict] = {}
    if args.progress_cache.is_file():
        progress = json.loads(args.progress_cache.read_text(encoding="utf-8"))
        print(f"Resuming from progress cache ({len(progress):,} done)")

    pending = [(code, title) for code, title in items if code not in progress]
    total = len(items)
    print(
        f"Checking effectiveness for {len(pending):,} pending / {total:,} law_codes "
        f"(workers={args.workers}, vbpl → VietLex)..."
    )

    def lookup(item: tuple[str, str]) -> tuple[str, dict]:
        code, title = item
        result = fetch_effectiveness(
            code,
            title=title or None,
            timeout=args.timeout,
            verify_ssl=args.vbpl_verify_ssl,
            use_vbpl=not args.skip_vbpl,
            use_vietlex=not args.skip_vietlex,
        )
        return code, {
            "law_title": title,
            "eff_code": (result.get("eff_code") or "").strip(),
            "eff_status": (result.get("eff_status") or "").strip(),
            "effective_date": result.get("effective_date") or "",
            "expiry_date": result.get("expiry_date") or "",
            "source": result.get("source") or "",
        }

    done = len(progress)
    expired: list[dict] = []
    skipped_not_found = 0
    skipped_effective = 0

    def recompute_stats() -> None:
        nonlocal skipped_not_found, skipped_effective, expired
        skipped_not_found = skipped_effective = 0
        expired = []
        for code, record in progress.items():
            eff_code = record.get("eff_code") or ""
            eff_status = record.get("eff_status") or ""
            if not eff_code and not eff_status:
                skipped_not_found += 1
                continue
            if not is_expired(eff_code, eff_status):
                skipped_effective += 1
                continue
            expired.append(
                {
                    "law_code": code,
                    "law_title": record.get("law_title") or codes.get(code, ""),
                    "eff_code": eff_code,
                    "eff_status": eff_status,
                    "effective_date": record.get("effective_date") or "",
                    "expiry_date": record.get("expiry_date") or "",
                    "source": record.get("source") or "",
                }
            )

    recompute_stats()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(lookup, item): item[0] for item in pending}
        for future in as_completed(futures):
            code = futures[future]
            try:
                looked_up_code, record = future.result()
                progress[looked_up_code] = record
            except Exception as exc:
                progress[code] = {
                    "law_title": codes.get(code, ""),
                    "eff_code": "",
                    "eff_status": "",
                    "effective_date": "",
                    "expiry_date": "",
                    "source": "error",
                    "error": str(exc),
                }
            done += 1
            if done % 100 == 0 or done == total:
                recompute_stats()
                args.progress_cache.parent.mkdir(parents=True, exist_ok=True)
                args.progress_cache.write_text(
                    json.dumps(progress, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(expired, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"  [{done}/{total}] not_found={skipped_not_found:,} "
                    f"effective={skipped_effective:,} expired={len(expired):,}"
                )
            if args.sleep > 0:
                time.sleep(args.sleep)

    recompute_stats()
    expired.sort(key=lambda row: row["law_code"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(expired, ensure_ascii=False, indent=2), encoding="utf-8")
    args.progress_cache.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\nDone. Expired: {len(expired):,} / {total:,} checked\n"
        f"  skipped (not found): {skipped_not_found:,}\n"
        f"  skipped (still effective): {skipped_effective:,}\n"
        f"  output → {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
