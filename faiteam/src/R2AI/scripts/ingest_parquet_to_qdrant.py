#!/usr/bin/env python3
"""Ingest filtered parquet legal documents into Qdrant via tree chunking + Vietnamese_Embedding_v2."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ChunkExtractor.chunk_extractor import ChunkExtractor
from document_filters import (
    DEFAULT_CUTOFF,
    filter_by_title,
    filter_document_row,
    filter_metadata_effective,
    load_effectiveness_table,
    load_keywords,
    matched_keywords,
    parse_metadata_date,
)
from qdrant_config import (
    ROOT,
    add_qdrant_args,
    get_collection_name,
    get_embed_model_path,
    get_qdrant_url,
    init_qdrant_from_args,
    make_qdrant_client,
)
from qdrant_ingest import ensure_collection, get_ingested_doc_ids, make_point_id, resolve_device
from text_document_builder import TextDocumentBuilder

DEFAULT_PARQUET = ROOT.parent / "data" / "full.parquet"
DEFAULT_CHECKPOINT = ROOT / "output" / "ingest_parquet_checkpoint.json"
DEFAULT_EFFECTIVENESS = ROOT.parent / "data" / "effectiveness.parquet"


def resolve_data_dir(path: Path) -> Path:
    if path.is_file():
        return path.parent
    return path


def load_filtered_metadata(
    data_dir: Path,
    keywords: list[str],
    metadata_filter: Path | None = None,
) -> pd.DataFrame:
    if metadata_filter is not None and metadata_filter.is_file():
        return pd.read_parquet(metadata_filter)
    meta_path = data_dir / "metadata.parquet"
    if meta_path.is_file():
        meta = pd.read_parquet(meta_path)
    elif (data_dir / "full.parquet").is_file():
        meta = pd.read_parquet(data_dir / "full.parquet", columns=[
            "id", "document_number", "title", "url", "legal_type",
            "legal_sectors", "issuing_authority", "issuance_date", "signers",
        ])
    else:
        raise FileNotFoundError(f"Không tìm thấy metadata trong {data_dir}")
    return filter_by_title(meta, keywords)


def load_documents_batch(
    data_dir: Path,
    meta_batch: pd.DataFrame,
) -> pd.DataFrame:
    """Load content for a metadata batch without loading the full content parquet."""
    ids = meta_batch["id"].tolist()
    content_path = data_dir / "content.parquet"
    if content_path.is_file():
        content = pd.read_parquet(content_path, filters=[("id", "in", ids)])
        return meta_batch.merge(content, on="id")
    full_path = data_dir / "full.parquet"
    if full_path.is_file():
        full = pd.read_parquet(full_path, filters=[("id", "in", ids)])
        return full
    raise FileNotFoundError(f"Không tìm thấy content parquet trong {data_dir}")


def iter_document_batches(
    data_dir: Path,
    filtered_meta: pd.DataFrame,
    *,
    batch_size: int = 32,
):
    """Yield (batch_start, dataframe, total) slices of filtered documents."""
    total = len(filtered_meta)
    for start in range(0, total, batch_size):
        meta_batch = filtered_meta.iloc[start : start + batch_size]
        docs = load_documents_batch(data_dir, meta_batch)
        yield start, docs, total


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"processed_ids": [], "last_offset": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_with_retry(client, collection: str, points: list[PointStruct], retries: int = 5) -> None:
    delay = 1.0
    for attempt in range(retries):
        try:
            client.upsert(collection_name=collection, points=points)
            return
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"  Upsert retry {attempt + 1}/{retries}: {exc}")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def build_chunks_for_row(
    row: dict[str, Any],
    doc_id: str,
    builder: TextDocumentBuilder,
    chunk_extractor: ChunkExtractor,
) -> list[dict[str, Any]]:
    document = builder.build(row)
    file_name = f"parquet_{doc_id}.txt"
    original_file_path = str(row.get("url") or f"parquet://{doc_id}")

    chunks = chunk_extractor.get_chunks_in_tree(
        document=document,
        doc_id=doc_id,
        original_file_path=original_file_path,
        file_name=file_name,
    )
    if not chunks:
        chunks = chunk_extractor.get_chunks_no_article(
            document=document,
            doc_id=doc_id,
            original_file_path=original_file_path,
            file_name=file_name,
        )
    return chunks


def chunk_to_payload(row: dict[str, Any], chunk: dict[str, Any], keywords_hit: list[str]) -> dict[str, Any]:
    return {
        "doc_id": chunk["doc_id"],
        "chunk_id": chunk["chunk_id"],
        "file_name": chunk["file_name"],
        "original_file_path": chunk["original_file_path"],
        "law_title": row.get("title") or chunk.get("law_title") or "",
        "law_type": row.get("legal_type") or chunk.get("law_type") or "",
        "law_code": row.get("document_number") or chunk.get("law_code") or "",
        "article_number": chunk.get("article_number") or "",
        "text": chunk["text"],
        "url": row.get("url") or "",
        "source": "parquet",
        "matched_keywords": keywords_hit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest filtered parquet legal documents into Qdrant",
    )
    add_qdrant_args(parser)
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=DEFAULT_PARQUET,
        help="Path to full.parquet (or metadata+content parent dir)",
    )
    parser.add_argument(
        "--metadata-filter",
        type=Path,
        default=None,
        help="Pre-filtered metadata parquet (skip title keyword filter)",
    )
    parser.add_argument(
        "--keywords-file",
        type=Path,
        default=None,
        help="Optional file with one title keyword per line",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to Vietnamese_Embedding_v2 model (default: EMBED_MODEL_PATH or ROAD2AI/models/)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument(
        "--upsert-batch",
        type=int,
        default=128,
        help="Points per Qdrant upsert call",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max documents to process")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N filtered rows")
    parser.add_argument(
        "--skip-ingested",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip doc_id already present in Qdrant (default: on)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"Checkpoint JSON path (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint writes",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate collection before ingest",
    )
    parser.add_argument(
        "--effectiveness-path",
        type=Path,
        default=DEFAULT_EFFECTIVENESS,
        help="Optional effectiveness sidecar parquet (id, eff_code, eff_status, expiry_date)",
    )
    parser.add_argument(
        "--cutoff-date",
        default=DEFAULT_CUTOFF.isoformat(),
        help="Keep documents valid on this date (default: 2026-03-01)",
    )
    parser.add_argument(
        "--unknown-policy",
        choices=("include", "exclude"),
        default="include",
        help="Policy when effectiveness is unknown",
    )
    parser.add_argument(
        "--skip-effectiveness-filter",
        action="store_true",
        help="Disable effectiveness filter (title keywords only)",
    )
    parser.add_argument(
        "--read-batch",
        type=int,
        default=32,
        help="Documents to load from parquet per I/O batch",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Device for embedding model: "cuda", "cpu", or "auto"',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    init_qdrant_from_args(args)
    checkpoint_path = None if args.no_checkpoint else (args.checkpoint or DEFAULT_CHECKPOINT)
    data_dir = resolve_data_dir(args.parquet_path)

    if not data_dir.is_dir() or not (
        (data_dir / "metadata.parquet").is_file()
        or (data_dir / "full.parquet").is_file()
    ):
        print(f"Parquet không tồn tại: {data_dir}", file=sys.stderr)
        return 1

    model_path = get_embed_model_path(args.model_path)
    if not model_path.is_dir():
        print(f"Model không tồn tại: {model_path}", file=sys.stderr)
        return 1
    args.model_path = model_path

    keywords = load_keywords(str(args.keywords_file) if args.keywords_file else None)
    cutoff = parse_metadata_date(args.cutoff_date) or DEFAULT_CUTOFF
    print(f"Data dir: {data_dir}")
    print(f"Mốc hiệu lực: {cutoff.isoformat()}")

    filtered_meta = load_filtered_metadata(data_dir, keywords, args.metadata_filter)
    if args.metadata_filter:
        print(f"Metadata filter: {args.metadata_filter} → {len(filtered_meta):,} rows")
    else:
        print(f"Sau lọc title ({len(keywords)} keywords): {len(filtered_meta):,}")

    eff_table = None
    if not args.skip_effectiveness_filter:
        eff_table = load_effectiveness_table(str(args.effectiveness_path))
        if eff_table is not None:
            print(f"Effectiveness sidecar: {len(eff_table):,} rows")
        else:
            print("Effectiveness sidecar: không có — lọc metadata + parse content khi ingest")

        filtered_meta = filter_metadata_effective(
            filtered_meta,
            cutoff=cutoff,
            effectiveness_table=eff_table,
            unknown_policy=args.unknown_policy,
        )
        print(f"Sau lọc hiệu lực (mốc {cutoff.isoformat()}): {len(filtered_meta):,}")

    if args.offset:
        filtered_meta = filtered_meta.iloc[args.offset :].reset_index(drop=True)
        print(f"Sau offset {args.offset}: {len(filtered_meta):,}")

    if args.limit:
        filtered_meta = filtered_meta.head(args.limit)
        print(f"Giới hạn --limit {args.limit}: {len(filtered_meta):,}")

    if filtered_meta.empty:
        print("Không có văn bản để ingest.")
        return 0

    collection = get_collection_name(args.collection)
    client = make_qdrant_client(args.qdrant_url, args.qdrant_api_key)
    ensure_collection(client, collection, args.recreate)

    ingested: set[str] = set()
    if args.skip_ingested:
        ingested = get_ingested_doc_ids(client, collection)
        print(f"Đã ingest trước đó: {len(ingested):,} doc_id")

    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path else {"processed_ids": []}
    processed_ids = set(checkpoint.get("processed_ids", []))

    device = resolve_device(args.device)
    print(f"Model: {args.model_path}")
    print(f"Device: {device or 'default'} | embed_batch={args.batch_size} | upsert_batch={args.upsert_batch}")
    print(f"Qdrant: {get_qdrant_url(args.qdrant_url)} / collection={collection}")

    model_kwargs: dict[str, Any] = {}
    if device:
        model_kwargs["device"] = device
    model = SentenceTransformer(str(args.model_path), **model_kwargs)
    model.max_seq_length = 2048

    builder = TextDocumentBuilder()
    chunk_extractor = ChunkExtractor()

    total_chunks = 0
    processed_docs = 0
    failed: list[tuple[str, str]] = []
    pending_points: list[PointStruct] = []

    def flush_points() -> None:
        nonlocal pending_points
        if not pending_points:
            return
        for i in range(0, len(pending_points), args.upsert_batch):
            batch = pending_points[i : i + args.upsert_batch]
            upsert_with_retry(client, collection, batch)
        pending_points = []

    total_docs = len(filtered_meta)
    skipped_effect = 0
    for batch_start, filtered, _ in iter_document_batches(
        data_dir,
        filtered_meta,
        batch_size=args.read_batch,
    ):
        for idx_in_batch, row in enumerate(filtered.itertuples(index=False), start=1):
            row_dict = row._asdict()
            doc_id = str(row_dict["id"])
            title = str(row_dict.get("title") or "")
            global_idx = batch_start + idx_in_batch

            if doc_id in ingested or doc_id in processed_ids:
                continue

            content = str(row_dict.get("content") or "")
            if not args.skip_effectiveness_filter and not filter_document_row(
                row_dict,
                content=content,
                cutoff=cutoff,
                effectiveness_table=eff_table,
                unknown_policy=args.unknown_policy,
            ):
                skipped_effect += 1
                continue

            print(f"[{global_idx}/{total_docs}] id={doc_id} | {title[:80]}")

            try:
                keywords_hit = matched_keywords(title, keywords)
                chunks = build_chunks_for_row(row_dict, doc_id, builder, chunk_extractor)
            except Exception as exc:
                msg = str(exc)
                print(f"  LỖI: {msg}")
                failed.append((doc_id, msg))
                continue

            if not chunks:
                print("  Bỏ qua: không có chunk")
                continue

            texts = [c["text"] for c in chunks]

            for i in range(0, len(texts), args.batch_size):
                batch_chunks = chunks[i : i + args.batch_size]
                batch_texts = texts[i : i + args.batch_size]
                embeddings = model.encode(
                    batch_texts,
                    batch_size=args.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                for chunk, vector in zip(batch_chunks, embeddings):
                    payload = chunk_to_payload(row_dict, chunk, keywords_hit)
                    pending_points.append(
                        PointStruct(
                            id=make_point_id(payload["doc_id"], payload["chunk_id"]),
                            vector=vector.tolist(),
                            payload=payload,
                        )
                    )

            if len(pending_points) >= args.upsert_batch:
                flush_points()

            total_chunks += len(chunks)
            processed_docs += 1
            processed_ids.add(doc_id)
            print(f"  → {len(chunks)} chunks")

            if checkpoint_path:
                save_checkpoint(
                    checkpoint_path,
                    {
                        "processed_ids": sorted(processed_ids, key=lambda x: int(x) if x.isdigit() else x),
                        "last_doc_id": doc_id,
                        "total_chunks": total_chunks,
                    },
                )

    flush_points()

    info = client.get_collection(collection)
    print()
    print(f"Hoàn tất: {total_chunks} chunks từ {processed_docs}/{total_docs} văn bản")
    if skipped_effect:
        print(f"Bỏ qua thêm {skipped_effect} văn bản sau kiểm tra hiệu lực theo nội dung")
    print(f"Qdrant collection '{collection}': {info.points_count} points")
    if failed:
        print(f"Lỗi ({len(failed)} văn bản):")
        for doc_id, err in failed[:10]:
            print(f"  - {doc_id}: {err}")
        if len(failed) > 10:
            print(f"  ... và {len(failed) - 10} văn bản khác")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
