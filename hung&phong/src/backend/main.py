"""FastAPI app — /chat (streaming SSE), /chat_sync, /health."""
from __future__ import annotations

import json
import logging
from typing import Iterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.config import get_settings
from backend.openai_adapter import router as openai_router
from backend.rag import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot VBPL VN", version="0.1.0")
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI-compatible endpoints for open-webui / other UIs.
app.include_router(openai_router)

pipeline: RAGPipeline | None = None


@app.on_event("startup")
def _startup() -> None:
    global pipeline
    pipeline = RAGPipeline()
    logger.info(
        "Backend ready · model=%s · embed=%s · collection=%s",
        settings.llm_model,
        settings.embed_model,
        settings.qdrant_collection,
    )


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


@app.get("/health")
def health() -> dict:
    count = pipeline.store.count() if pipeline else 0
    return {
        "status": "ok",
        "collection": settings.qdrant_collection,
        "doc_chunks": count,
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
    }


@app.post("/chat_sync")
def chat_sync(req: ChatRequest) -> dict:
    assert pipeline is not None
    return pipeline.answer(req.query, top_k=req.top_k)


@app.post("/chat")
def chat_stream(req: ChatRequest) -> EventSourceResponse:
    assert pipeline is not None

    def event_gen() -> Iterator[dict]:
        try:
            for evt in pipeline.stream_answer(req.query, top_k=req.top_k):
                yield {"event": evt["type"], "data": json.dumps(evt["data"], ensure_ascii=False)}
        except Exception as e:
            logger.exception("stream error")
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    return EventSourceResponse(event_gen())


if __name__ == "__main__":
    # reload=True làm 2 process tranh lock với embedded Qdrant.
    # Bật lại khi chuyển sang QDRANT_URL=http://...
    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=False,
    )
