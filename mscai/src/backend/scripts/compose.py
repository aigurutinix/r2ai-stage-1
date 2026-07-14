"""Chạy Docker Compose bằng các giá trị lấy từ backend/config.yaml."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.config import get_settings


def main() -> None:
    settings = get_settings()
    postgres = urlparse(settings.legal_assistant.postgres.database_url)
    if postgres.scheme not in {"postgres", "postgresql"}:
        raise ValueError("legal_assistant.postgres.database_url phải là PostgreSQL URL")

    env = os.environ.copy()
    env.update(
        {
            "POSTGRES_USER": unquote(postgres.username or "postgres"),
            "POSTGRES_PASSWORD": unquote(postgres.password or "postgres"),
            "POSTGRES_DB": postgres.path.lstrip("/") or "legal_assistant",
            "POSTGRES_PORT": str(postgres.port or 5432),
            "UI_PORT": str(settings.ui.port),
        }
    )
    command = ["docker", "compose", "-f", str(PROJECT_ROOT / "docker-compose.yml"), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command, cwd=PROJECT_ROOT, env=env))


if __name__ == "__main__":
    main()
