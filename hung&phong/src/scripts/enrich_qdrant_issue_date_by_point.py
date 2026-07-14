"""Enrich a Qdrant collection with issue_date using direct point-id updates.

This is intended for a collection copied from `vbpl_aiteam`. It does not filter
by `source_url` inside Qdrant, because that causes slow full scans when the field
is not indexed. Instead it scrolls points once, builds batches of point ids, and
updates payload by ids.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from qdrant_client import QdrantClient


ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = ROOT / "data" / "hf_cache" / "tmquan___vbpl-vn" / "documents" / "0.0.0"


def find_arrow_dir() -> Path:
    dirs = [p for p in HF_CACHE.glob("*") if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No HF Arrow cache directory under {HF_CACHE}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def load_issue_by_url() -> dict[str, str]:
    out: dict[str, str] = {}
    arrow_dir = find_arrow_dir()
    for fp in sorted(arrow_dir.glob("*.arrow")):
        with pa.memory_map(str(fp), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                table = pa.Table.from_batches([batch]).select(["source_url", "issue_date"])
                data = table.to_pydict()
                for url, issue_date in zip(data["source_url"], data["issue_date"]):
                    url = str(url or "").strip()
                    issue_date = str(issue_date or "").strip()
                    if url and issue_date:
                        out.setdefault(url, issue_date)
    return out


def chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--collection", default="vbpl_aiteam_meta_20260628")
    ap.add_argument("--scroll", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--limit-points", type=int, default=0)
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()

    client = QdrantClient(url=args.url, timeout=120)
    info = client.get_collection(args.collection)
    print(f"collection={args.collection} points={info.points_count:,} status={info.status}", flush=True)

    issue_by_url = load_issue_by_url()
    print(f"issue_date map loaded: {len(issue_by_url):,} source_url rows", flush=True)

    ids_by_date: dict[str, list] = defaultdict(list)
    scanned = matched = already = 0
    offset = None
    t0 = time.time()
    while True:
        points, offset = client.scroll(
            collection_name=args.collection,
            limit=args.scroll,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            scanned += 1
            payload = p.payload or {}
            source_url = str(payload.get("source_url") or "").strip()
            issue_date = issue_by_url.get(source_url)
            if not issue_date:
                continue
            matched += 1
            if payload.get("ngay_ban_hanh") == issue_date and payload.get("metadata_enriched_issue_date"):
                already += 1
                continue
            ids_by_date[issue_date].append(p.id)
            if args.limit_points and matched >= args.limit_points:
                offset = None
                break
        if scanned % 50000 == 0:
            print(
                f"  scanned={scanned:,} matched={matched:,} pending_ids={sum(len(v) for v in ids_by_date.values()):,}",
                flush=True,
            )
        if offset is None:
            break

    pending = sum(len(v) for v in ids_by_date.values())
    print(
        f"scan done in {time.time()-t0:.0f}s: scanned={scanned:,}, matched={matched:,}, "
        f"already={already:,}, pending={pending:,}, dates={len(ids_by_date):,}",
        flush=True,
    )

    updated = 0
    t1 = time.time()
    for issue_date, ids in sorted(ids_by_date.items()):
        payload = {
            "ngay_ban_hanh": issue_date,
            "metadata_enriched_issue_date": "tmquan/vbpl-vn.issue_date",
        }
        for batch_ids in chunks(ids, args.batch):
            client.set_payload(
                collection_name=args.collection,
                payload=payload,
                points=batch_ids,
                wait=args.wait,
            )
            updated += len(batch_ids)
            if updated % 50000 == 0:
                rate = updated / max(time.time() - t1, 1)
                print(f"  updated={updated:,}/{pending:,} ({rate:.0f}/s)", flush=True)

    print(f"done: updated={updated:,}/{pending:,} in {time.time()-t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
