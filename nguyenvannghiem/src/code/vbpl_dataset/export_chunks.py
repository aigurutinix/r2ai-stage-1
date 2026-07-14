#!/usr/bin/env python3
"""
Export toàn bộ vbpl_dataset/full_text/ thành JSON chunks.

Output layout:
  chunks/
  ├── bo_luat/
  │   └── 2559.json          ← per-doc: full ParseResult + chunks
  ├── luat/
  │   └── ...
  ├── bo_luat.jsonl           ← per-type flat: 1 line = 1 chunk
  ├── nghi_dinh.jsonl
  ├── ...
  └── _stats.json             ← tổng hợp

Usage:
  python export_chunks.py                        # export all
  python export_chunks.py --skip-existing        # resume (skip done files)
  python export_chunks.py --type nghi_dinh       # one doc type only
  python export_chunks.py --workers 16           # parallelism
  python export_chunks.py --out-dir /some/path   # custom output dir
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).parent / "full_text"
DEFAULT_OUT = Path(__file__).parent / "chunks"
sys.path.insert(0, str(Path(__file__).parent))

from legal_chunker import parse_file, ParseResult


# ── Serialisation ─────────────────────────────────────────────────────────────


def result_to_dict(r: ParseResult) -> dict:
    """Full ParseResult → JSON-serialisable dict (per-doc JSON)."""
    return {
        "doc_id": r.doc_id,
        "doc_type": r.doc_type,
        "file_path": r.file_path,
        "parse_type": r.parse_type,
        "lang": r.lang,
        "total_chunks": r.total_chunks,
        "paywall_lines": r.paywall_lines,
        "warnings": [{"level": w.level, "message": w.message} for w in r.warnings],
        "chunks": [asdict(c) for c in r.chunks],
    }


def chunk_to_jsonl_record(r: ParseResult, chunk_dict: dict) -> dict:
    """Flat chunk record for JSONL (adds doc-level fields)."""
    return {
        "chunk_id": chunk_dict["chunk_id"],
        "doc_id": r.doc_id,
        "doc_type": r.doc_type,
        "parse_type": r.parse_type,
        "lang": r.lang,
        "paywall_lines": r.paywall_lines,
        **{k: chunk_dict[k] for k in (
            "article_number", "article_title", "path",
            "content", "khoan", "char_count",
            "citation_keys", "prev_article", "next_article",
        )},
    }


# ── Phase 1: parse + write per-doc JSON ──────────────────────────────────────


def _process_one(txt_path: Path, out_dir: Path, skip_existing: bool) -> tuple[str, dict | None]:
    """
    Parse one file and write per-doc JSON.
    Returns (doc_type, summary_dict_or_None_if_skipped).
    """
    doc_type = txt_path.parent.name
    out_path = out_dir / doc_type / f"{txt_path.stem}.json"

    if skip_existing and out_path.exists():
        # Return minimal summary from existing file instead of re-parsing
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            return doc_type, {
                "doc_id": existing["doc_id"],
                "parse_type": existing["parse_type"],
                "lang": existing["lang"],
                "total_chunks": existing["total_chunks"],
                "paywall_lines": existing["paywall_lines"],
                "has_error": any(w["level"] == "error" for w in existing.get("warnings", [])),
                "skipped": True,
            }
        except Exception:
            pass  # re-parse if existing file is corrupt

    result = parse_file(txt_path)
    data = result_to_dict(result)

    # Atomic write: temp → rename
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)

    return doc_type, {
        "doc_id": result.doc_id,
        "parse_type": result.parse_type,
        "lang": result.lang,
        "total_chunks": result.total_chunks,
        "paywall_lines": result.paywall_lines,
        "has_error": any(w.level == "error" for w in result.warnings),
        "skipped": False,
    }


def phase1_parse(
    paths: list[Path],
    out_dir: Path,
    workers: int,
    skip_existing: bool,
) -> list[tuple[str, dict]]:
    """Parse all files and write per-doc JSONs in parallel."""
    summaries: list[tuple[str, dict]] = []
    done = skipped = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, p, out_dir, skip_existing): p for p in paths}
        for fut in as_completed(futures):
            try:
                doc_type, summary = fut.result()
                summaries.append((doc_type, summary))
                if summary.get("skipped"):
                    skipped += 1
                else:
                    done += 1
            except Exception as e:
                print(f"  ERROR {futures[fut]}: {e}", file=sys.stderr)

            total_done = done + skipped
            if total_done % 2000 == 0:
                elapsed = time.time() - t0
                rate = total_done / elapsed if elapsed else 0
                print(f"  {total_done:,}/{len(paths):,}  ({rate:.0f} files/s)"
                      + (f"  [{skipped} skipped]" if skipped else ""))

    elapsed = time.time() - t0
    print(f"  Phase 1 done: {done:,} parsed, {skipped:,} skipped — {elapsed:.1f}s")
    return summaries


# ── Phase 2: write per-type JSONL ────────────────────────────────────────────


def phase2_jsonl(doc_types: list[str], out_dir: Path) -> dict[str, int]:
    """Read per-doc JSONs and write flat JSONL per doc_type."""
    type_chunk_counts: dict[str, int] = {}

    for doc_type in doc_types:
        type_dir = out_dir / doc_type
        json_files = sorted(type_dir.glob("*.json"))
        if not json_files:
            continue

        jsonl_path = out_dir / f"{doc_type}.jsonl"
        count = 0
        with jsonl_path.open("w", encoding="utf-8") as fout:
            for jf in json_files:
                try:
                    data = json.loads(jf.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"  JSONL skip {jf}: {e}", file=sys.stderr)
                    continue

                # Reconstruct a minimal ParseResult-like object for chunk_to_jsonl_record
                class _R:
                    pass
                r = _R()
                r.doc_id = data["doc_id"]
                r.doc_type = data["doc_type"]
                r.parse_type = data["parse_type"]
                r.lang = data["lang"]
                r.paywall_lines = data["paywall_lines"]

                for chunk_dict in data.get("chunks", []):
                    record = chunk_to_jsonl_record(r, chunk_dict)
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

        type_chunk_counts[doc_type] = count
        print(f"  {doc_type:<28} → {count:>8,} chunks  ({jsonl_path.name})")

    return type_chunk_counts


# ── Stats ─────────────────────────────────────────────────────────────────────


def write_stats(summaries: list[tuple[str, dict]], type_chunk_counts: dict, out_dir: Path) -> None:
    from collections import Counter

    all_summaries = [s for _, s in summaries]
    parse_types = Counter(s["parse_type"] for s in all_summaries)
    langs = Counter(s["lang"] for s in all_summaries)

    by_type: dict[str, dict] = defaultdict(lambda: {"files": 0, "chunks": 0, "errors": 0})
    for doc_type, s in summaries:
        d = by_type[doc_type]
        d["files"] += 1
        d["chunks"] += s["total_chunks"]
        if s["has_error"]:
            d["errors"] += 1

    stats = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_files": len(all_summaries),
        "total_chunks": sum(s["total_chunks"] for s in all_summaries),
        "parse_type": dict(parse_types),
        "lang": dict(langs),
        "files_with_errors": sum(1 for s in all_summaries if s["has_error"]),
        "files_with_paywall": sum(1 for s in all_summaries if s["paywall_lines"] > 0),
        "by_doc_type": {
            k: dict(v) for k, v in sorted(by_type.items())
        },
        "jsonl_chunk_counts": type_chunk_counts,
    }

    out = out_dir / "_stats.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Stats saved → {out}")
    print(f"  Total files : {stats['total_files']:,}")
    print(f"  Total chunks: {stats['total_chunks']:,}")
    print(f"  Errors      : {stats['files_with_errors']}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", dest="doc_type", default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files where per-doc JSON already exists (resume)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # Collect input paths
    if args.doc_type:
        paths = sorted((ROOT / args.doc_type).glob("*.txt"))
        doc_types = [args.doc_type]
    else:
        paths = sorted(ROOT.rglob("*.txt"))
        doc_types = sorted(d.name for d in ROOT.iterdir() if d.is_dir())

    # Create output subdirectories
    for dt in doc_types:
        (out_dir / dt).mkdir(parents=True, exist_ok=True)

    print(f"Input : {len(paths):,} files from {ROOT}")
    print(f"Output: {out_dir}")
    print(f"Workers: {args.workers}  skip-existing: {args.skip_existing}")
    print()

    # Phase 1: parse + per-doc JSON
    print("── Phase 1: Parse & write per-doc JSON ─────────────────────────")
    summaries = phase1_parse(paths, out_dir, args.workers, args.skip_existing)
    print()

    # Phase 2: flat JSONL per type
    print("── Phase 2: Write per-type JSONL ───────────────────────────────")
    type_chunk_counts = phase2_jsonl(doc_types, out_dir)
    print()

    # Stats
    print("── Stats ────────────────────────────────────────────────────────")
    write_stats(summaries, type_chunk_counts, out_dir)


if __name__ == "__main__":
    main()
