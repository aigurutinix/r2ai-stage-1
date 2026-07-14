#!/usr/bin/env python3
"""Ingest DOCX legal documents into Qdrant via ChunkExtractor + Vietnamese_Embedding_v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ChunkExtractor.chunk_extractor import ChunkExtractor
from FisReader.document_factory import DocumentFactory
from qdrant_config import (
    ROOT,
    add_qdrant_args,
    get_collection_name,
    get_qdrant_url,
    make_qdrant_client,
)
from qdrant_ingest import ensure_collection, get_ingested_doc_ids, make_point_id, resolve_device

DEFAULT_DOCX_DIR = ROOT / "VbplCrawler" / "output" / "doanh-nghiep-docx"
DEFAULT_MODEL_PATH = ROOT / "models" / "Vietnamese_Embedding_v2"


def extract_doc_id(file_path: Path) -> str:
    stem = file_path.stem
    if "_" in stem:
        return stem.split("_")[-1]
    return stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest DOCX files into Qdrant")
    add_qdrant_args(parser)
    parser.add_argument(
        "--docx-dir",
        type=Path,
        default=DEFAULT_DOCX_DIR,
        help="Directory containing .docx files",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to Vietnamese_Embedding_v2 model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of docx files to process (for testing)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Device for embedding model: "cuda", "cpu", or "auto" (default: cuda)',
    )
    parser.add_argument(
        "--skip-ingested",
        action="store_true",
        help="Skip docx files whose doc_id already exists in Qdrant",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate collection before ingest",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.docx_dir.is_dir():
        print(f"Thư mục không tồn tại: {args.docx_dir}", file=sys.stderr)
        return 1
    if not args.model_path.is_dir():
        print(f"Model không tồn tại: {args.model_path}", file=sys.stderr)
        return 1

    docx_files = sorted(args.docx_dir.glob("*.docx"))
    if args.limit:
        docx_files = docx_files[: args.limit]
    if not docx_files:
        print(f"Không tìm thấy file .docx trong {args.docx_dir}", file=sys.stderr)
        return 1

    collection = get_collection_name(args.collection)
    client = make_qdrant_client(args.qdrant_url, args.qdrant_api_key)
    ensure_collection(client, collection, args.recreate)

    if args.skip_ingested:
        ingested = get_ingested_doc_ids(client, collection)
        before = len(docx_files)
        docx_files = [f for f in docx_files if extract_doc_id(f) not in ingested]
        print(f"Bỏ qua {before - len(docx_files)} file đã ingest, còn {len(docx_files)} file")

    if not docx_files:
        print("Tất cả file đã được ingest.")
        return 0

    device = resolve_device(args.device)
    print(f"Tìm thấy {len(docx_files)} file docx")
    print(f"Model: {args.model_path}")
    print(f"Device: {device or 'default'} | batch_size={args.batch_size}")
    print(f"Qdrant: {get_qdrant_url(args.qdrant_url)} / collection={collection}")

    model_kwargs = {}
    if device:
        model_kwargs["device"] = device
    model = SentenceTransformer(str(args.model_path), **model_kwargs)
    model.max_seq_length = 2048

    factory = DocumentFactory()
    chunk_extractor = ChunkExtractor()

    total_chunks = 0
    failed: list[tuple[str, str]] = []

    for idx, file_path in enumerate(docx_files, start=1):
        doc_id = extract_doc_id(file_path)
        print(f"[{idx}/{len(docx_files)}] {file_path.name}")

        try:
            document = factory.read(file_path, doc_id=doc_id, chunk_type="legal")
            chunks = chunk_extractor.get_chunks_in_tree(
                document=document,
                doc_id=doc_id,
                original_file_path=str(file_path),
                file_name=file_path.name,
            )
        except Exception as exc:
            msg = str(exc)
            print(f"  LỖI: {msg}")
            failed.append((file_path.name, msg))
            continue

        if not chunks:
            print("  Bỏ qua: không có chunk")
            continue

        texts = [c["text"] for c in chunks]
        points: list[PointStruct] = []

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
                points.append(
                    PointStruct(
                        id=make_point_id(chunk["doc_id"], chunk["chunk_id"]),
                        vector=vector.tolist(),
                        payload={
                            "doc_id": chunk["doc_id"],
                            "chunk_id": chunk["chunk_id"],
                            "file_name": chunk["file_name"],
                            "original_file_path": chunk["original_file_path"],
                            "law_title": chunk["law_title"],
                            "law_type": chunk["law_type"],
                            "law_code": chunk["law_code"],
                            "article_number": chunk["article_number"],
                            "text": chunk["text"],
                            "source": "docx",
                        },
                    )
                )

        client.upsert(collection_name=collection, points=points)
        total_chunks += len(chunks)
        print(f"  → {len(chunks)} chunks upserted")

    info = client.get_collection(collection)
    print()
    print(f"Hoàn tất: {total_chunks} chunks từ {len(docx_files) - len(failed)}/{len(docx_files)} file")
    print(f"Qdrant collection '{collection}': {info.points_count} points")
    if failed:
        print(f"Lỗi ({len(failed)} file):")
        for name, err in failed[:10]:
            print(f"  - {name}: {err}")
        if len(failed) > 10:
            print(f"  ... và {len(failed) - 10} file khác")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
