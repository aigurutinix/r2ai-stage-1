"""Nạp legal records từ JSON vào PostgreSQL bằng upsert theo id."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Khi chạy trực tiếp ``python scripts/load_postgres.py``, Python chỉ thêm thư
# mục ``scripts`` vào import path. Thêm backend root để import package ``src``.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.config import get_settings

DEFAULT_DATASET = BACKEND_ROOT / "data" / "base_data.json"
REQUIRED_FIELDS = {
    "id",
    "law_id",
    "law_name",
    "doc_type",
    "article",
    "article_title",
    "content",
    "author",
}


def parse_args() -> argparse.Namespace:
    """Đọc tham số CLI cho lệnh import."""

    parser = argparse.ArgumentParser(description="Nạp legal dataset JSON vào PostgreSQL")
    parser.add_argument("--file", type=Path, default=DEFAULT_DATASET, help="Đường dẫn JSON array")
    parser.add_argument("--database-url", help="Ghi đè PostgreSQL URL trong config.yaml")
    parser.add_argument("--batch-size", type=int, default=500, help="Số record mỗi batch")
    parser.add_argument("--truncate", action="store_true", help="Xóa dữ liệu cũ trước khi nạp")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    """Đọc và validate cấu trúc tối thiểu của dataset."""

    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Dataset phải là một JSON array")

    seen_ids: set[int] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} không phải JSON object")
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"Record {index} thiếu field: {sorted(missing)}")
        record_id = int(record["id"])
        if record_id in seen_ids:
            raise ValueError(f"ID bị trùng: {record_id}")
        seen_ids.add(record_id)
    return records


def to_row(record: dict[str, Any]) -> tuple[Any, ...]:
    """Chuẩn hóa một JSON record thành bộ tham số SQL."""

    extra = record.get("extra") or []
    return (
        int(record["id"]),
        str(record["law_id"]),
        str(record["law_name"]),
        str(record["doc_type"]),
        record.get("chapter"),
        str(record["article"]),
        str(record["article_title"]),
        str(record["content"]),
        str(record["author"]),
        json.dumps(extra, ensure_ascii=False),
    )


async def ensure_schema(conn: Any) -> None:
    """Tạo/migrate bảng PostgreSQL sang schema không còn category."""

    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS legal_knowledge_records (
            id BIGINT PRIMARY KEY,
            law_id TEXT NOT NULL,
            law_name TEXT NOT NULL,
            doc_type TEXT NOT NULL,
            chapter TEXT,
            article TEXT NOT NULL,
            article_title TEXT NOT NULL,
            content TEXT NOT NULL,
            author TEXT NOT NULL,
            extra JSONB NOT NULL DEFAULT '[]'::jsonb
        )
        """
    )
    await conn.execute("DROP INDEX IF EXISTS idx_legal_knowledge_category")
    await conn.execute("ALTER TABLE legal_knowledge_records DROP COLUMN IF EXISTS category")
    await conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_legal_knowledge_reference
        ON legal_knowledge_records (law_id, article)
        """
    )


async def import_records(args: argparse.Namespace) -> None:
    """Kết nối PostgreSQL và upsert dataset theo batch."""

    import asyncpg

    if args.batch_size < 1:
        raise ValueError("--batch-size phải lớn hơn 0")

    records = load_records(args.file.resolve())
    database_url = args.database_url or get_settings().legal_assistant.postgres.database_url
    conn = await asyncpg.connect(database_url)
    try:
        await ensure_schema(conn)
        if args.truncate:
            await conn.execute("TRUNCATE TABLE legal_knowledge_records")

        sql = """
            INSERT INTO legal_knowledge_records (
                id, law_id, law_name, doc_type, chapter, article,
                article_title, content, author, extra
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                law_id = EXCLUDED.law_id,
                law_name = EXCLUDED.law_name,
                doc_type = EXCLUDED.doc_type,
                chapter = EXCLUDED.chapter,
                article = EXCLUDED.article,
                article_title = EXCLUDED.article_title,
                content = EXCLUDED.content,
                author = EXCLUDED.author,
                extra = EXCLUDED.extra
        """
        for start in range(0, len(records), args.batch_size):
            batch = [to_row(record) for record in records[start : start + args.batch_size]]
            await conn.executemany(sql, batch)
            print(f"Đã nạp {min(start + args.batch_size, len(records))}/{len(records)} record")

        total = await conn.fetchval("SELECT COUNT(*) FROM legal_knowledge_records")
        print(f"Hoàn tất: PostgreSQL có {total} record")
    finally:
        await conn.close()

    persist_directory = get_settings().legal_assistant.vector_store.persist_directory
    for filename in ("legal_index_manifest.json", "legal_bm25_cache.json"):
        (persist_directory / filename).unlink(missing_ok=True)
    print("Đã invalid Chroma/BM25 cache; backend sẽ rebuild index ở lần khởi động tiếp theo")


if __name__ == "__main__":
    asyncio.run(import_records(parse_args()))
