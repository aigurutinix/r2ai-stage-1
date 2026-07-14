"""Khởi động Uvicorn bằng host/port trong backend/config.yaml."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.config import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy legal assistant backend")
    parser.add_argument("--reload", action="store_true", help="Tự reload khi source thay đổi")
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=args.reload,
        workers=1,
    )


if __name__ == "__main__":
    main()
