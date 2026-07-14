from __future__ import annotations

import os
import sqlite3
import threading
from typing import Any

from Agents.config import (
    CHECKPOINT_POSTGRES_DSN,
    CHECKPOINT_POSTGRES_POOL_SIZE,
    CHECKPOINT_SQLITE_PATH,
    CHECKPOINTER_BACKEND,
)

SUPPORTED_BACKENDS = {"postgres", "sqlite", "none"}

_checkpointer_lock = threading.Lock()
_checkpointer: Any | None = None
_sqlite_conn: sqlite3.Connection | None = None
_postgres_pool: Any | None = None
_postgres_context_manager: Any | None = None


def get_checkpointer_backend() -> str:
    """Trả về backend checkpoint đã normalize."""
    backend = (CHECKPOINTER_BACKEND or "postgres").strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"CHECKPOINTER_BACKEND={backend!r} không hợp lệ. "
            f"Chỉ hỗ trợ: {', '.join(sorted(SUPPORTED_BACKENDS))}."
        )
    return backend


def get_sqlite_path() -> str:
    return CHECKPOINT_SQLITE_PATH


def get_postgres_dsn() -> str:
    return CHECKPOINT_POSTGRES_DSN


def _setup_saver(saver: Any) -> Any:
    if hasattr(saver, "setup"):
        saver.setup()
    return saver


def _make_sqlite_checkpointer() -> Any:
    global _sqlite_conn

    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = get_sqlite_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(
        db_path,
        check_same_thread=False,
        timeout=30.0,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")

    _sqlite_conn = conn
    return _setup_saver(SqliteSaver(conn))


def _missing_postgres_dependency_error(exc: ImportError) -> RuntimeError:
    return RuntimeError(
        "CHECKPOINTER_BACKEND=postgres cần cài dependency: "
        "`langgraph-checkpoint-postgres` và `psycopg[binary,pool]`."
    )


def _make_postgres_checkpointer() -> Any:
    global _postgres_pool, _postgres_context_manager

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:
        raise _missing_postgres_dependency_error(exc) from exc

    dsn = get_postgres_dsn()

    try:
        from psycopg_pool import ConnectionPool
    except ImportError:
        ConnectionPool = None

    if ConnectionPool is not None:
        try:
            pool_size = max(1, CHECKPOINT_POSTGRES_POOL_SIZE)
            _postgres_pool = ConnectionPool(
                conninfo=dsn,
                min_size=1,
                max_size=pool_size,
                kwargs={"autocommit": True, "prepare_threshold": 0},
            )
            return _setup_saver(PostgresSaver(_postgres_pool))
        except Exception as exc:
            raise RuntimeError(
                "Không thể khởi tạo Postgres checkpointer. "
                "Hãy kiểm tra Postgres local đã chạy, database `legal_assistant` đã tồn tại, "
                "và CHECKPOINT_POSTGRES_DSN đúng."
            ) from exc

    try:
        _postgres_context_manager = PostgresSaver.from_conn_string(dsn)
        saver = _postgres_context_manager.__enter__()
        return _setup_saver(saver)
    except Exception as exc:
        raise RuntimeError(
            "Không thể khởi tạo Postgres checkpointer. "
            "Hãy kiểm tra Postgres local đã chạy, database `legal_assistant` đã tồn tại, "
            "và CHECKPOINT_POSTGRES_DSN đúng."
        ) from exc


def get_checkpointer() -> Any | None:
    """
    Tạo checkpointer dùng chung cho process.

    - postgres: khuyến nghị khi phục vụ nhiều request đồng thời.
    - sqlite: fallback/dev mode với WAL và busy_timeout.
    - none: không lưu checkpoint, dùng cho batch evaluation độc lập.
    """
    global _checkpointer

    backend = get_checkpointer_backend()
    if backend == "none":
        return None

    if _checkpointer is None:
        with _checkpointer_lock:
            if _checkpointer is None:
                if backend == "postgres":
                    _checkpointer = _make_postgres_checkpointer()
                elif backend == "sqlite":
                    _checkpointer = _make_sqlite_checkpointer()

    return _checkpointer
