"""Export router được dùng trong FastAPI app factory."""
from src.routers.health import router as health_router
from src.routers.legal import router as legal_router

__all__ = ["health_router", "legal_router"]
