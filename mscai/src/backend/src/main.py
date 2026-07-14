"""Entry point FastAPI duy nhất của runtime agent.

File này đăng ký router hỏi đáp và lifecycle startup. Pipeline index vẫn nằm
trong module vector_store riêng; main chỉ gọi nó một lần trước khi nhận request.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.routers import health_router, legal_router
from src.services.vector_store.index_builder import initialize_legal_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build/khôi phục retrieval index một lần trước khi nhận request."""

    await initialize_legal_index(app.state.settings)
    yield


def create_app() -> FastAPI:
    """Tạo FastAPI app cho agent service."""

    settings = get_settings()
    app = FastAPI(title="Vietnamese Legal Assistant Agent", version="0.1.0", lifespan=lifespan)
    # UI nằm ở folder ``ui`` riêng và gọi API backend qua browser, nên bật CORS
    # cho môi trường dev local. Production có thể siết lại domain thật sau.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://localhost:{settings.ui.port}",
            f"http://127.0.0.1:{settings.ui.port}",
            f"http://localhost:{settings.app.port}",
            f"http://127.0.0.1:{settings.app.port}",
            "null",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Lưu settings vào app.state để các middleware/tooling sau này có thể đọc
    # mà không cần parse lại YAML.
    app.state.settings = settings
    app.include_router(health_router)
    app.include_router(legal_router)
    return app


app = create_app()
