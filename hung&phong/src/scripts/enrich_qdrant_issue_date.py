"""Enrich Qdrant payload with issue_date from the original tmquan/vbpl-vn cache.

The current exported corpus/manifest lost `issue_date`, while the HuggingFace
Arrow cache still has it. This script updates Qdrant payload field
`ngay_ban_hanh` using exact `source_url` matches.

It intentionally does not fill `ngay_hieu_luc` or `tinh_trang_hieu_luc` because
the tmquan/vbpl-vn source does not provide those fields.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib import request

import pyarrow as pa
import pyarrow.ipc as ipc


ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = ROOT / "data" / "hf_cache" / "tmquan___vbpl-vn" / "documents" / "0.0.0"
MANIFEST = ROOT / "data" / "corpus_vbpl_v2" / "manifest.json"


def find_arrow_dir() -> Path:
    dirs = [p for p in HF_CACHE.glob("*") if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No HF Arrow cache directory under {HF_CACHE}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def load_target_urls() -> set[str]:
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {str(r.get("source_url") or "").strip() for r in rows if r.get("source_url")}


def load_issue_dates(target_urls: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    arrow_dir = find_arrow_dir()
    cols = ["source_url", "issue_date"]
    for fp in sorted(arrow_dir.glob("*.arrow")):
        with pa.memory_map(str(fp), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                table = pa.Table.from_batches([batch]).select(cols)
                data = table.to_pydict()
                for url, issue_date in zip(data["source_url"], data["issue_date"]):
                    url = str(url or "").strip()
                    issue_date = str(issue_date or "").strip()
                    if url in target_urls and issue_date:
                        out.setdefault(url, issue_date)
    return out


def qdrant_post(base_url: str, path: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def collection_info(base_url: str, collection: str) -> dict:
    with request.urlopen(f"{base_url.rstrip('/')}/collections/{collection}", timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))["result"]


def set_payload_by_source_url(base_url: str, collection: str, source_url: str, issue_date: str, wait: bool) -> None:
    payload = {
        "payload": {
            "ngay_ban_hanh": issue_date,
            "metadata_enriched_issue_date": "tmquan/vbpl-vn.issue_date",
        },
        "filter": {"must": [{"key": "source_url", "match": {"value": source_url}}]},
    }
    qdrant_post(base_url, f"/collections/{collection}/points/payload?wait={str(wait).lower()}", payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", default="vbpl_aiteam")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()

    info = collection_info(args.qdrant_url, args.collection)
    print(
        f"Collection {args.collection}: points={info.get('points_count')} "
        f"status={info.get('status')}",
        flush=True,
    )

    target_urls = load_target_urls()
    issue_by_url = load_issue_dates(target_urls)
    items = sorted(issue_by_url.items())
    if args.limit:
        items = items[: args.limit]
    print(f"Target URLs={len(target_urls):,}; issue_date found={len(issue_by_url):,}; updating={len(items):,}", flush=True)

    t0 = time.time()
    for i, (url, issue_date) in enumerate(items, 1):
        set_payload_by_source_url(args.qdrant_url, args.collection, url, issue_date, args.wait)
        if i % 500 == 0 or i == len(items):
            print(f"  updated {i:,}/{len(items):,} ({i / max(time.time() - t0, 1):.1f}/s)", flush=True)

    print("Done. Rebuild BM25 after this if lexical side should carry updated payload metadata.", flush=True)


if __name__ == "__main__":
    main()
