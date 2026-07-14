"""Build document-level metadata graph from the active Qdrant collection.

The retrieval selector needs document facts that are not available from a
single chunk score:

- issue date and benchmark cutoff eligibility;
- whether a document is an amendment/consolidation-style document;
- which document numbers are referenced in the title;
- rough successor edges inferred from amendment titles.

This script does not mutate Qdrant.  It creates a JSON artifact that pipeline
code can load deterministically.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower().replace("đ", "d")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text).strip()


def prefix(so_ky_hieu: str) -> str:
    sk = unicodedata.normalize("NFKC", so_ky_hieu or "").strip()
    m = re.match(r"(\d+/\d{4})", sk)
    return m.group(1) if m else sk


def referenced_prefixes(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text or "")
    refs = sorted(set(re.findall(r"\b\d{1,4}/(?:19|20)\d{2}\b", text)))
    return refs


def amendment_kind(title: str, loai: str) -> str:
    t = norm(title)
    if "van ban hop nhat" in norm(loai) or "hop nhat" in t:
        return "consolidated"
    if any(s in t for s in ["sua doi", "bo sung", "bai bo", "thay the"]):
        return "amendment"
    return "base"


def scroll_docs(url: str, collection: str) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    offset = None
    while True:
        body = {"limit": 1000, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(f"{url}/collections/{collection}/points/scroll", json=body, timeout=30)
        r.raise_for_status()
        result = r.json()["result"]
        for point in result["points"]:
            p = point.get("payload") or {}
            sk = str(p.get("so_ky_hieu") or "")
            if not sk:
                continue
            px = prefix(sk)
            doc = docs.setdefault(
                px,
                {
                    "prefix": px,
                    "so_ky_hieu_examples": set(),
                    "title": str(p.get("title") or ""),
                    "loai_van_ban": str(p.get("loai_van_ban") or ""),
                    "linh_vuc": str(p.get("linh_vuc") or ""),
                    "co_quan_ban_hanh": str(p.get("co_quan_ban_hanh") or ""),
                    "ngay_ban_hanh": str(p.get("ngay_ban_hanh") or ""),
                    "nam": str(p.get("nam") or ""),
                    "source_url": str(p.get("source_url") or ""),
                    "article_count": 0,
                    "chunk_count": 0,
                    "_articles": set(),
                },
            )
            doc["so_ky_hieu_examples"].add(sk)
            doc["chunk_count"] += 1
            if p.get("dieu_so") not in (None, ""):
                doc["_articles"].add(str(p.get("dieu_so")))
            for k in ["title", "loai_van_ban", "linh_vuc", "co_quan_ban_hanh", "ngay_ban_hanh", "nam", "source_url"]:
                if not doc.get(k) and p.get(k):
                    doc[k] = str(p.get(k))
        offset = result.get("next_page_offset")
        if offset is None:
            break
    return docs


def build(args: argparse.Namespace) -> None:
    docs = scroll_docs(args.qdrant_url, args.collection)
    successors: dict[str, list[str]] = defaultdict(list)
    for px, doc in docs.items():
        refs = referenced_prefixes(doc["title"])
        doc["referenced_prefixes_in_title"] = refs
        doc["amendment_kind"] = amendment_kind(doc["title"], doc["loai_van_ban"])
        doc["after_eval_cutoff"] = bool(doc["ngay_ban_hanh"] and doc["ngay_ban_hanh"] > args.cutoff)
        doc["article_count"] = len(doc.pop("_articles"))
        doc["so_ky_hieu_examples"] = sorted(doc["so_ky_hieu_examples"])
        if doc["amendment_kind"] in {"amendment", "consolidated"}:
            for ref in refs:
                if ref != px:
                    successors[ref].append(px)

    for old, news in successors.items():
        successors[old] = sorted(set(news), key=lambda p: (docs.get(p, {}).get("ngay_ban_hanh", ""), p))

    out = {
        "collection": args.collection,
        "qdrant_url": args.qdrant_url,
        "eval_cutoff": args.cutoff,
        "doc_count": len(docs),
        "docs": dict(sorted(docs.items())),
        "successors_from_title_refs": dict(sorted(successors.items())),
        "notes": [
            "Edges are title-derived candidates, not authoritative legal-effectiveness facts.",
            "Use as ranking evidence, not as a hard replacement graph.",
            "ngay_hieu_luc/tinh_trang_hieu_luc are not available in the current collection.",
        ],
    }
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out_path),
        "doc_count": len(docs),
        "successor_sources": len(successors),
        "after_eval_cutoff_docs": sum(1 for d in docs.values() if d["after_eval_cutoff"]),
        "amendment_docs": sum(1 for d in docs.values() if d["amendment_kind"] == "amendment"),
        "consolidated_docs": sum(1 for d in docs.values() if d["amendment_kind"] == "consolidated"),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", default="vbpl_aiteam_meta_parsefix_20260628")
    ap.add_argument("--cutoff", default="2026-03-31")
    ap.add_argument("--out", default="data/doc_metadata_graph_20260630.json")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
