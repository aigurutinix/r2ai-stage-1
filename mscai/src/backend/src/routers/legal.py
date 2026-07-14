"""Các HTTP endpoint phục vụ hỏi đáp pháp lý."""
from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from time import perf_counter

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.config import get_settings
from src.dependencies import get_legal_assistant_agent
from src.schemas.api.chat import (
    ChatRequest,
    ChatResponse,
    ChatStreamDoneEvent,
    ChatStreamErrorEvent,
    ChatStreamMessagePayload,
    ChatStreamResultEvent,
    ChatStreamStatusEvent,
    ChatStreamTokenEvent,
    ChatStreamTokenPayload,
    LegalRuntimeConfig,
    CompetitionBatchRequest,
    CompetitionBatchResponse,
    CompetitionRecord,
)
from src.schemas.legal import LegalAnswerRequest, LegalAnswerResponse
from src.services.competition.output import (
    append_report,
    compact_results,
    completed_count,
    new_output_paths,
    output_status,
    persist_final,
    persist_progress,
)
from src.services.agents.legal_assistant import LegalAssistantAgent

router = APIRouter(prefix="/api/v1/legal", tags=["legal-assistant"])


@router.get("/config", response_model=LegalRuntimeConfig)
async def runtime_config() -> LegalRuntimeConfig:
    """Trả config runtime để UI biết có dùng chat streaming hay không."""

    settings = get_settings().legal_assistant
    return LegalRuntimeConfig(
        chat_streaming=settings.chat.streaming,
        token_streaming=settings.chat.token_streaming,
        competition_enabled=settings.competition.enabled,
    )


@router.post("/answer", response_model=LegalAnswerResponse)
async def answer_question(
    request: LegalAnswerRequest,
    agent: LegalAssistantAgent = Depends(get_legal_assistant_agent),
) -> LegalAnswerResponse:
    """Trả lời một câu hỏi pháp lý đơn lẻ, không tự ghi short-memory."""

    return await agent.answer(request)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent: LegalAssistantAgent = Depends(get_legal_assistant_agent),
) -> ChatResponse:
    """Chat wrapper dùng ``session_id`` làm LangGraph thread_id."""

    answer = await agent.answer(
        LegalAnswerRequest(
            session_id=request.session_id,
            question=request.message,
            competition_mode=request.competition_mode,
            top_k=request.top_k,
            include_debug=True,
        )
    )
    return ChatResponse(
        session_id=request.session_id,
        message=request.message,
        answer=answer,
        tool_calls=answer.debug.get("tool_calls", []),
    )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    agent: LegalAssistantAgent = Depends(get_legal_assistant_agent),
) -> StreamingResponse:
    """Stream từng stage, heartbeat và kết quả chat cuối cùng bằng SSE."""

    async def events():
        def pack(stream_event) -> str:
            event = stream_event.event
            data = stream_event.data.model_dump(mode="json")
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        started = perf_counter()
        stage_started = started
        current_stage = "request"
        current_message = "Đang chuẩn bị request"
        current_status = "started"
        progress_queue: asyncio.Queue[dict] = asyncio.Queue()

        async def on_progress(payload: dict) -> None:
            await progress_queue.put(payload)
            # Nhường event loop để StreamingResponse gửi event trước khi node
            # tiếp tục một tác vụ nặng.
            await asyncio.sleep(0)

        answer_request = LegalAnswerRequest(
            session_id=request.session_id,
            question=request.message,
            competition_mode=request.competition_mode,
            top_k=request.top_k,
            include_debug=True,
        )
        task = asyncio.create_task(agent.answer_with_progress(answer_request, on_progress))
        try:
            yield pack(
                ChatStreamStatusEvent(
                    data=ChatStreamMessagePayload(
                        message="Đã nhận request chat",
                        stage="request",
                        status="started",
                        elapsed_ms=0,
                        detail=f"Session: {request.session_id or 'tạm thời'}; {len(request.message)} ký tự.",
                    )
                )
            )

            while not task.done() or not progress_queue.empty():
                try:
                    payload = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if current_status in {"completed", "warning", "error"}:
                        current_stage = "response"
                        current_message = "Đang hoàn thiện response"
                        current_status = "running"
                        stage_started = perf_counter()
                    elapsed_ms = round((perf_counter() - stage_started) * 1000)
                    yield pack(
                        ChatStreamStatusEvent(
                            data=ChatStreamMessagePayload(
                                message=f"{current_message} ({elapsed_ms / 1000:.1f}s)",
                                stage=current_stage,
                                status="running",
                                elapsed_ms=elapsed_ms,
                                detail="Backend vẫn đang xử lý stage này; kết nối SSE còn hoạt động.",
                            )
                        )
                    )
                    continue

                if payload.get("event") == "token":
                    yield pack(
                        ChatStreamTokenEvent(
                            data=ChatStreamTokenPayload(
                                token=str(payload.get("token") or ""),
                                stage=str(payload.get("stage") or "answer"),
                            )
                        )
                    )
                    continue

                current_stage = str(payload.get("stage") or current_stage)
                current_message = str(payload.get("message") or current_message)
                current_status = str(payload.get("status") or current_status)
                if payload.get("status") == "started":
                    stage_started = perf_counter()
                elif payload.get("elapsed_ms") is None:
                    payload["elapsed_ms"] = round((perf_counter() - stage_started) * 1000)
                yield pack(ChatStreamStatusEvent(data=ChatStreamMessagePayload.model_validate(payload)))

            answer = await task
            response = ChatResponse(
                session_id=request.session_id,
                message=request.message,
                answer=answer,
                tool_calls=answer.debug.get("tool_calls", []),
            )
            yield pack(ChatStreamResultEvent(data=response))
            yield pack(
                ChatStreamDoneEvent(
                    data=ChatStreamMessagePayload(
                        message="Hoàn tất request",
                        stage="request",
                        status="completed",
                        elapsed_ms=round((perf_counter() - started) * 1000),
                    )
                )
            )
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception as exc:  # pragma: no cover - trả lỗi runtime cho UI
            yield pack(
                ChatStreamErrorEvent(
                    data=ChatStreamMessagePayload(
                        message=f"Lỗi tại stage {current_stage}: {exc}",
                        stage=current_stage,
                        status="error",
                        elapsed_ms=round((perf_counter() - stage_started) * 1000),
                        detail=exc.__class__.__name__,
                        metadata={"request_elapsed_ms": round((perf_counter() - started) * 1000)},
                    )
                )
            )
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _competition_request(item: LegalAnswerRequest) -> LegalAnswerRequest:
    """Chuẩn hóa một item tập test thành request legal RAG không dùng memory."""

    return item.model_copy(update={"competition_mode": True, "session_id": None, "include_debug": False})


def _competition_error_record(item: LegalAnswerRequest, exc: Exception) -> dict:
    """Tạo record vẫn đúng format submit khi một câu bị lỗi runtime."""

    return {
        "id": item.id,
        "question": item.question,
        "answer": f"Lỗi khi xử lý câu hỏi: {exc}",
        "relevant_docs": [],
        "relevant_articles": [],
    }


async def _run_competition_item(
    item: LegalAnswerRequest,
    agent: LegalAssistantAgent,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Chạy một câu competition, dùng semaphore để giới hạn concurrency."""

    async with semaphore:
        try:
            answer = await agent.answer(_competition_request(item))
            return answer.to_competition_record()
        except Exception as exc:  # pragma: no cover - fallback runtime
            return _competition_error_record(item, exc)


@router.post("/competition/stream")
async def answer_competition_stream(
    request: list[LegalAnswerRequest],
    agent: LegalAssistantAgent = Depends(get_legal_assistant_agent),
) -> StreamingResponse:
    """Stream tiến độ; mỗi câu xong được lưu ngay vào outputs."""

    async def events():
        def pack(event: str, data: dict | list) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        started = perf_counter()
        total = len(request)
        competition = agent.settings.legal_assistant.competition
        max_concurrency = competition.max_concurrency
        paths = new_output_paths(competition.output_dir)
        semaphore = asyncio.Semaphore(max_concurrency)
        write_lock = asyncio.Lock()
        progress_queue: asyncio.Queue[tuple[str, dict | list]] = asyncio.Queue()
        results: list[dict | None] = [None] * total

        persist_progress(paths, results)
        append_report(paths, f"START total={total} max_concurrency={max_concurrency} running={paths.running_path}")

        async def save_partial(message: str) -> None:
            async with write_lock:
                persist_progress(paths, results)
                append_report(paths, message)

        async def run_one(index: int, item: LegalAnswerRequest) -> None:
            item_started = perf_counter()
            async with semaphore:
                await progress_queue.put(
                    (
                        "status",
                        {
                            "message": f"Câu {index}/{total}: bắt đầu xử lý",
                            "stage": "competition_item",
                            "status": "started",
                            "elapsed_ms": 0,
                            "detail": item.question,
                            "metadata": {"index": index, "total": total, "id": item.id, "max_concurrency": max_concurrency},
                        },
                    )
                )

                async def on_progress(payload: dict) -> None:
                    data = dict(payload)
                    data["message"] = f"Câu {index}/{total}: {data.get('message', 'đang xử lý')}"
                    metadata = dict(data.get("metadata") or {})
                    metadata.update({"index": index, "total": total, "id": item.id, "max_concurrency": max_concurrency})
                    data["metadata"] = metadata
                    await progress_queue.put(("status", data))
                    await asyncio.sleep(0)

                try:
                    answer = await agent.answer_with_progress(_competition_request(item), on_progress)
                    record = answer.to_competition_record()
                    results[index - 1] = record
                    await save_partial(
                        f"ITEM_DONE index={index}/{total} id={item.id} articles={len(record.get('relevant_articles', []))}"
                    )
                    await progress_queue.put(("competition_item_result", {**record, "index": index, "total": total}))
                    await progress_queue.put(
                        (
                            "status",
                            {
                                "message": f"Hoàn tất câu {index}/{total}",
                                "stage": "competition_item",
                                "status": "completed",
                                "elapsed_ms": round((perf_counter() - item_started) * 1000),
                                "detail": f"Có {len(record.get('relevant_articles', []))} điều luật liên quan.",
                                "metadata": {"index": index, "total": total, "id": item.id, "running_path": str(paths.running_path)},
                            },
                        )
                    )
                except Exception as exc:  # pragma: no cover - fallback runtime
                    record = _competition_error_record(item, exc)
                    results[index - 1] = record
                    await save_partial(f"ITEM_ERROR index={index}/{total} id={item.id} error={exc}")
                    await progress_queue.put(
                        (
                            "status",
                            {
                                "message": f"Lỗi câu {index}/{total}: {exc}",
                                "stage": "competition_item",
                                "status": "error",
                                "elapsed_ms": round((perf_counter() - item_started) * 1000),
                                "detail": exc.__class__.__name__,
                                "metadata": {"index": index, "total": total, "id": item.id, "running_path": str(paths.running_path)},
                            },
                        )
                    )
                    await progress_queue.put(("competition_item_result", {**record, "index": index, "total": total}))

        tasks = [asyncio.create_task(run_one(index, item)) for index, item in enumerate(request, start=1)]
        yield pack(
            "status",
            {
                "message": "Bắt đầu competition mode",
                "stage": "competition",
                "status": "started",
                "elapsed_ms": 0,
                "detail": f"Nhận {total} câu hỏi. Chạy tối đa {max_concurrency} câu cùng lúc. Intent sẽ được bỏ qua.",
                "metadata": {
                    "total": total,
                    "max_concurrency": max_concurrency,
                    "running_path": str(paths.running_path),
                    "report_path": str(paths.report_path),
                },
            },
        )

        try:
            while any(not task.done() for task in tasks) or not progress_queue.empty():
                try:
                    event, payload = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                    yield pack(event, payload)
                except asyncio.TimeoutError:
                    completed = completed_count(results)
                    yield pack(
                        "status",
                        {
                            "message": f"Đang chạy competition mode ({completed}/{total} câu xong)",
                            "stage": "competition",
                            "status": "running",
                            "elapsed_ms": round((perf_counter() - started) * 1000),
                            "detail": "Backend vẫn đang xử lý; kết nối SSE còn hoạt động.",
                            "metadata": {
                                "completed": completed,
                                "total": total,
                                "max_concurrency": max_concurrency,
                                "running_path": str(paths.running_path),
                            },
                        },
                    )

            await asyncio.gather(*tasks)
            final_results = compact_results(results)
            status = output_status(final_results)
            final_path = persist_final(paths, results, status)
            append_report(paths, f"FINISH status={status} completed={len(final_results)}/{total} output={final_path}")
            yield pack("competition_result", final_results)
            yield pack(
                "done",
                {
                    "message": "Hoàn tất competition mode",
                    "stage": "competition",
                    "status": "completed",
                    "elapsed_ms": round((perf_counter() - started) * 1000),
                    "detail": f"Đã xử lý {len(final_results)}/{total} câu hỏi.",
                    "metadata": {
                        "total": total,
                        "output_path": str(final_path),
                        "running_path": str(paths.running_path),
                        "report_path": str(paths.report_path),
                        "max_concurrency": max_concurrency,
                    },
                },
            )
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            partial_results = compact_results(results)
            final_path = persist_final(paths, results, "error")
            append_report(paths, f"CANCELLED completed={len(partial_results)}/{total} output={final_path}")
            raise
        except Exception as exc:
            partial_results = compact_results(results)
            final_path = persist_final(paths, results, "error")
            append_report(paths, f"FATAL_ERROR completed={len(partial_results)}/{total} error={exc} output={final_path}")
            yield pack(
                "error",
                {
                    "message": f"Lỗi competition mode: {exc}",
                    "stage": "competition",
                    "status": "error",
                    "elapsed_ms": round((perf_counter() - started) * 1000),
                    "detail": exc.__class__.__name__,
                    "metadata": {
                        "completed": len(partial_results),
                        "total": total,
                        "output_path": str(final_path),
                        "running_path": str(paths.running_path),
                        "report_path": str(paths.report_path),
                    },
                },
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.post("/competition", response_model=list[CompetitionRecord])
async def answer_competition(
    request: list[LegalAnswerRequest],
    agent: LegalAssistantAgent = Depends(get_legal_assistant_agent),
) -> list[dict]:
    """Nhận JSON array tập test, chạy song song và lưu output sau từng câu."""

    competition = agent.settings.legal_assistant.competition
    max_concurrency = competition.max_concurrency
    semaphore = asyncio.Semaphore(max_concurrency)
    write_lock = asyncio.Lock()
    paths = new_output_paths(competition.output_dir)
    results: list[dict | None] = [None] * len(request)
    persist_progress(paths, results)
    append_report(paths, f"START total={len(request)} max_concurrency={max_concurrency} running={paths.running_path}")

    async def run_one(index: int, item: LegalAnswerRequest) -> None:
        record = await _run_competition_item(item, agent, semaphore)
        results[index - 1] = record
        async with write_lock:
            persist_progress(paths, results)
            prefix = "ITEM_ERROR" if output_status([record]) == "error" else "ITEM_DONE"
            append_report(paths, f"{prefix} index={index}/{len(request)} id={item.id}")

    try:
        await asyncio.gather(*(run_one(index, item) for index, item in enumerate(request, start=1)))
        final_results = compact_results(results)
        status = output_status(final_results)
        final_path = persist_final(paths, results, status)
        append_report(paths, f"FINISH status={status} completed={len(final_results)}/{len(request)} output={final_path}")
        return final_results
    except asyncio.CancelledError:
        final_path = persist_final(paths, results, "error")
        append_report(paths, f"CANCELLED completed={completed_count(results)}/{len(request)} output={final_path}")
        raise
    except Exception as exc:
        final_path = persist_final(paths, results, "error")
        append_report(paths, f"FATAL_ERROR completed={completed_count(results)}/{len(request)} error={exc} output={final_path}")
        raise


@router.post("/batch", response_model=CompetitionBatchResponse)
async def answer_batch(
    request: CompetitionBatchRequest,
    agent: LegalAssistantAgent = Depends(get_legal_assistant_agent),
) -> CompetitionBatchResponse:
    """Trả lời nhiều câu hỏi, phù hợp khi chạy tập test của cuộc thi."""

    results = [await agent.answer(item) for item in request.items]
    return CompetitionBatchResponse(results=results)
