"""Endpoint health check tối giản cho agent service."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Trả tín hiệu service còn sống cho người dùng, Docker hoặc load balancer."""

    return {"status": "ok"}
