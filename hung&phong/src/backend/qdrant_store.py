"""Qdrant wrapper — create collection, upsert, search.

Hỗ trợ 3 mode qua biến `QDRANT_URL`:
  - `http://...` / `https://...` → kết nối server (Docker / cloud).
  - `:memory:` → embedded in-process, mất khi tắt.
  - `file://path` hoặc path thường (vd `./qdrant_local`) → embedded
    persistent, lưu vào ổ đĩa.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from backend.config import get_settings

logger = logging.getLogger(__name__)


def _build_client(url: str) -> tuple[QdrantClient, bool]:
    """Tạo QdrantClient phù hợp với mode. Trả về (client, is_embedded)."""
    if url.startswith(("http://", "https://")):
        return QdrantClient(url=url, timeout=60), False
    if url == ":memory:":
        logger.info("Qdrant in-memory mode (volatile)")
        return QdrantClient(location=":memory:"), True
    if url.startswith("file://"):
        parsed = urlparse(url)
        # Windows: file:///D:/foo → path = /D:/foo → strip leading /
        raw = parsed.path
        if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
            raw = raw[1:]
        path = Path(raw)
    else:
        path = Path(url)
    path.mkdir(parents=True, exist_ok=True)
    logger.info("Qdrant embedded persistent mode at %s", path.resolve())
    return QdrantClient(path=str(path)), True


class QdrantStore:
    def __init__(self) -> None:
        s = get_settings()
        self.client, self.is_embedded = _build_client(s.qdrant_url)
        self.collection = s.qdrant_collection
        self.dim = s.embed_dim

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            # Embedded mode (local) không support optimizers_config & on_disk.
            if self.is_embedded:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=qm.VectorParams(
                        size=self.dim,
                        distance=qm.Distance.COSINE,
                    ),
                )
            else:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=qm.VectorParams(
                        size=self.dim,
                        distance=qm.Distance.COSINE,
                        on_disk=True,
                    ),
                    optimizers_config=qm.OptimizersConfigDiff(
                        default_segment_number=4,
                        indexing_threshold=20000,
                    ),
                )
            self._create_payload_indexes()
            logger.info("Created Qdrant collection: %s", self.collection)

    def _create_payload_indexes(self) -> None:
        """Index payload fields for fast metadata filtering."""
        fields = [
            ("doc_id", qm.PayloadSchemaType.KEYWORD),
            ("so_ky_hieu", qm.PayloadSchemaType.KEYWORD),
            ("loai_van_ban", qm.PayloadSchemaType.KEYWORD),
            ("co_quan_ban_hanh", qm.PayloadSchemaType.KEYWORD),
            ("tinh_trang_hieu_luc", qm.PayloadSchemaType.KEYWORD),
            ("linh_vuc", qm.PayloadSchemaType.KEYWORD),
            ("dieu_so", qm.PayloadSchemaType.INTEGER),
            ("ngay_ban_hanh", qm.PayloadSchemaType.KEYWORD),
        ]
        for name, schema in fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=name,
                    field_schema=schema,
                )
            except Exception as e:
                logger.debug("Index %s may already exist: %s", name, e)

    def upsert_batch(self, points: Iterable[qm.PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=list(points), wait=False)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_must: dict[str, Any] | None = None,
    ) -> list[dict]:
        qfilter = None
        if filter_must:
            conditions = [
                qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
                for k, v in filter_must.items()
            ]
            qfilter = qm.Filter(must=conditions)

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        )

        return [
            {"score": h.score, "payload": dict(h.payload or {}), "id": h.id}
            for h in hits
        ]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count
