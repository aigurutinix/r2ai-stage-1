"""Copy a Qdrant collection point-for-point.

Used to keep the original collection as backup before payload enrichment.
Vectors, payloads and point ids are preserved.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
sys.stdout.reconfigure(encoding="utf-8")

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--src", default="vbpl_aiteam")
    ap.add_argument("--dst", default="vbpl_aiteam_issue_date")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--resume-existing", action="store_true")
    args = ap.parse_args()

    client = QdrantClient(url=args.url)
    if not client.collection_exists(args.src):
        raise SystemExit(f"Source collection not found: {args.src}")
    if client.collection_exists(args.dst):
        if args.recreate:
            client.delete_collection(args.dst)
        elif not args.resume_existing:
            raise SystemExit(
                f"Destination exists: {args.dst}. Use --resume-existing to upsert into it "
                "or --recreate to replace it."
            )

    src_info = client.get_collection(args.src)
    params = src_info.config.params
    print(
        f"Copy {args.src} -> {args.dst}: points={src_info.points_count:,}, "
        f"vectors={params.vectors}",
        flush=True,
    )

    if not client.collection_exists(args.dst):
        client.create_collection(
            collection_name=args.dst,
            vectors_config=params.vectors,
            shard_number=params.shard_number,
            replication_factor=params.replication_factor,
            write_consistency_factor=params.write_consistency_factor,
            on_disk_payload=params.on_disk_payload,
        )

        for field, schema in (src_info.payload_schema or {}).items():
            try:
                client.create_payload_index(
                    collection_name=args.dst,
                    field_name=field,
                    field_schema=schema.data_type,
                )
            except Exception as exc:
                print(f"  index skip {field}: {exc}", flush=True)
    else:
        dst_count = client.count(args.dst, exact=True).count
        print(f"Destination exists; upserting into it. current dst_count={dst_count:,}", flush=True)

    offset = None
    total = 0
    t0 = time.time()
    while True:
        points, offset = client.scroll(
            collection_name=args.src,
            limit=args.batch,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if points:
            batch = [
                qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload or {})
                for p in points
            ]
            client.upsert(collection_name=args.dst, points=batch, wait=False)
            total += len(batch)
            if total % (args.batch * 20) == 0:
                rate = total / max(time.time() - t0, 1)
                print(f"  copied {total:,}/{src_info.points_count:,} ({rate:.0f}/s)", flush=True)
        if offset is None:
            break

    # Ensure async upserts are flushed enough for count visibility.
    dst_count = client.count(args.dst, exact=True).count
    print(f"Done: copied={total:,}; dst_count={dst_count:,}", flush=True)


if __name__ == "__main__":
    main()
