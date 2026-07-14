"""Embed chunks qua LLM provider rồi upsert vào Qdrant."""
from __future__ import annotations

import logging
from typing import Iterable, Iterator

from qdrant_client.http import models as qm
from tqdm import tqdm

from backend.config import get_settings
from backend.embed import EmbeddingClient
from backend.qdrant_store import QdrantStore
from ingest.chunk import Chunk

logger = logging.getLogger(__name__)


def _batched(iterable: Iterable, batch_size: int) -> Iterator[list]:
    batch: list = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def embed_and_upsert(
    chunks: Iterable[Chunk],
    embedder: EmbeddingClient | None = None,
    store: QdrantStore | None = None,
    show_progress: bool = True,
) -> int:
    """Embed batch & upsert. Trả về tổng số chunk đã ghi."""
    s = get_settings()
    embedder = embedder or EmbeddingClient()
    store = store or QdrantStore()
    store.ensure_collection(recreate=False)

    total = 0
    chunks_list = list(chunks) if not isinstance(chunks, list) else chunks
    iterator: Iterable[list[Chunk]] = _batched(chunks_list, s.embed_batch_size)
    if show_progress:
        n_batches = (len(chunks_list) + s.embed_batch_size - 1) // s.embed_batch_size
        iterator = tqdm(iterator, total=n_batches, desc="embed+upsert", unit="batch")

    for batch in iterator:
        texts = [c.text for c in batch]
        vectors = embedder.embed_batch(texts)
        points = [
            qm.PointStruct(id=c.chunk_id, vector=vec, payload=c.payload)
            for c, vec in zip(batch, vectors)
        ]
        store.upsert_batch(points)
        total += len(points)

    logger.info("Upserted %d chunks to collection %s", total, store.collection)
    return total
