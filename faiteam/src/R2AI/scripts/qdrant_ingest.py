"""Shared helpers for Qdrant ingest scripts."""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from qdrant_config import VECTOR_SIZE


def make_point_id(doc_id: str, chunk_id: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:{chunk_id}"))


def get_ingested_doc_ids(client: QdrantClient, collection: str) -> set[str]:
    if not client.collection_exists(collection):
        return set()
    doc_ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=["doc_id"],
            with_vectors=False,
        )
        for point in points:
            if point.payload and point.payload.get("doc_id"):
                doc_ids.add(str(point.payload["doc_id"]))
        if offset is None:
            break
    return doc_ids


def ensure_collection(
    client: QdrantClient,
    collection: str,
    recreate: bool,
) -> None:
    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)
        print(f"Đã xóa collection: {collection}")

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Đã tạo collection: {collection}")
    else:
        print(f"Collection '{collection}' đã tồn tại — upsert thêm dữ liệu")


def resolve_device(device: str) -> str | None:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device if device else None
