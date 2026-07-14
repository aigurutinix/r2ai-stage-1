"""OpenAI Chat Completions-compatible adapter cho open-webui / generic clients.

Wraps `RAGPipeline` behind `/v1/chat/completions` + `/v1/models` để bất kỳ
OpenAI-compatible UI (open-webui, Continue.dev, ChatBox, ...) đều chat được
với backend mà không cần biết SSE custom format. Sources hiển thị bằng cách
append markdown footer vào cuối câu trả lời.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.config import get_settings

router = APIRouter(prefix="/v1", tags=["openai"])


class _Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[_Message]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


def _extract_query(messages: list[_Message]) -> str:
    """Lấy nội dung user message cuối cùng (RAG không cần lịch sử)."""
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    lines = ["\n\n---", "**Nguồn tham khảo:**"]
    for i, s in enumerate(sources[:5], 1):
        title = s.get("title") or "?"
        skh = s.get("so_ky_hieu") or "?"
        loai = s.get("loai_van_ban") or ""
        dieu = s.get("dieu_so")
        dieu_str = f" · Điều {dieu}" if dieu is not None else ""
        ttl = s.get("tinh_trang_hieu_luc")
        ttl_str = f" _(⚠️ {ttl})_" if ttl and "hết" in ttl.lower() else ""
        lines.append(f"{i}. {loai} {skh}{dieu_str} — {title}{ttl_str}")
    return "\n".join(lines)


def _make_chunk(content: str, model: str, completion_id: str, role: str | None = None) -> dict:
    delta: dict = {"content": content}
    if role:
        delta["role"] = role
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


@router.get("/models")
def list_models() -> dict:
    s = get_settings()
    return {
        "object": "list",
        "data": [
            {
                "id": "chatbot-vbpl-vn",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "root": "chatbot-vbpl-vn",
                "meta": {
                    "description": "Chatbot văn bản pháp luật VN (RAG)",
                    "llm_backend": s.llm_model,
                },
            }
        ],
    }


@router.post("/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    # Import muộn để tránh circular import với main.py.
    from backend.main import pipeline

    assert pipeline is not None
    query = _extract_query(req.messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    model_id = req.model or "chatbot-vbpl-vn"

    if not req.stream:
        result = pipeline.answer(query)
        content = result["answer"] + _format_sources(result["sources"])
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def sse_gen() -> Iterator[str]:
        sources: list[dict] = []
        first = True
        for evt in pipeline.stream_answer(query):
            if evt["type"] == "sources":
                sources = evt["data"]
                continue
            if evt["type"] == "token":
                role = "assistant" if first else None
                first = False
                chunk = _make_chunk(evt["data"], model_id, completion_id, role=role)
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            elif evt["type"] == "done":
                footer = _format_sources(sources)
                if footer:
                    chunk = _make_chunk(footer, model_id, completion_id)
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                final = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

    return StreamingResponse(sse_gen(), media_type="text/event-stream")
