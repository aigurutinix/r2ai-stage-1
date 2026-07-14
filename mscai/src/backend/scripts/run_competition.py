"""Chạy competition dataset không cần UI, phù hợp chạy ngầm trong tmux.

Ví dụ:
    uv run python scripts/run_competition.py --file test.json

Script sẽ:
- khởi tạo retrieval index như backend startup;
- chạy tối đa legal_assistant.competition.max_concurrency câu cùng lúc;
- ghi outputs/competition_<run_id>_running.json sau mỗi câu;
- ghi outputs/competition_<run_id>_success.json hoặc _error.json khi kết thúc;
- append mọi diễn biến vào outputs/report.log.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.config import get_settings
from src.schemas.legal import LegalAnswerRequest
from src.services.agents.legal_assistant import LegalAssistantAgent
from src.services.competition.output import (
    append_report,
    compact_results,
    completed_count,
    new_output_paths,
    output_status,
    persist_final,
    persist_progress,
)
from src.services.llm.client import LLMClient
from src.services.vector_store.index_builder import initialize_legal_index


def parse_args() -> argparse.Namespace:
    """Đọc tham số chạy competition từ CLI."""

    parser = argparse.ArgumentParser(description="Chạy tập test competition và lưu output theo tiến độ")
    parser.add_argument("--file", type=Path, required=True, help="File JSON array gồm id/question")
    parser.add_argument("--max-concurrency", type=int, help="Ghi đè legal_assistant.competition.max_concurrency")
    parser.add_argument("--output-dir", type=Path, help="Ghi đè legal_assistant.competition.output_dir")
    return parser.parse_args()


def load_requests(path: Path) -> list[LegalAnswerRequest]:
    """Đọc JSON array và validate thành LegalAnswerRequest."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Input phải là JSON array")
    requests: list[LegalAnswerRequest] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} không phải object")
        requests.append(LegalAnswerRequest.model_validate({**item, "competition_mode": True}))
    return requests


def error_record(item: LegalAnswerRequest, exc: Exception) -> dict:
    """Tạo record đúng format submit khi một câu lỗi."""

    return {
        "id": item.id,
        "question": item.question,
        "answer": f"Lỗi khi xử lý câu hỏi: {exc}",
        "relevant_docs": [],
        "relevant_articles": [],
    }


async def run() -> None:
    """Entry async chính để chạy competition job."""

    args = parse_args()
    settings = get_settings()
    competition = settings.legal_assistant.competition
    if args.max_concurrency is not None:
        competition.max_concurrency = args.max_concurrency
    if args.output_dir is not None:
        competition.output_dir = args.output_dir.resolve()

    requests = load_requests(args.file.resolve())
    paths = new_output_paths(competition.output_dir)
    results: list[dict | None] = [None] * len(requests)
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(competition.max_concurrency)

    persist_progress(paths, results)
    append_report(paths, f"START_CLI file={args.file.resolve()} total={len(requests)} max_concurrency={competition.max_concurrency} running={paths.running_path}")
    print(f"[competition] output running: {paths.running_path}")
    print(f"[competition] report: {paths.report_path}")

    try:
        print("[competition] initializing legal index...")
        await initialize_legal_index(settings)
        # Competition luôn cần LLM cho bước tổng hợp câu trả lời.
        # Rewrite/HyDE nếu bật trong config.yaml cũng dùng chung client này.
        agent = LegalAssistantAgent(llm=LLMClient())

        async def run_one(index: int, item: LegalAnswerRequest) -> None:
            async with semaphore:
                try:
                    answer = await agent.answer(
                        item.model_copy(update={"competition_mode": True, "session_id": None, "include_debug": False})
                    )
                    record = answer.to_competition_record()
                    prefix = "ITEM_DONE"
                except Exception as exc:  # giữ job chạy tiếp nếu một câu lỗi
                    record = error_record(item, exc)
                    prefix = "ITEM_ERROR"
                results[index - 1] = record
                async with write_lock:
                    persist_progress(paths, results)
                    append_report(paths, f"{prefix} index={index}/{len(requests)} id={item.id} completed={completed_count(results)}/{len(requests)}")
                print(f"[competition] {prefix} {index}/{len(requests)} id={item.id}")

        tasks = [asyncio.create_task(run_one(index, item)) for index, item in enumerate(requests, start=1)]
        await asyncio.gather(*tasks)
        final_results = compact_results(results)
        status = output_status(final_results)
        final_path = persist_final(paths, results, status)
        append_report(paths, f"FINISH_CLI status={status} completed={len(final_results)}/{len(requests)} output={final_path}")
        print(f"[competition] finished: {final_path}")
    except asyncio.CancelledError:
        final_path = persist_final(paths, results, "error")
        append_report(paths, f"CANCELLED_CLI completed={completed_count(results)}/{len(requests)} output={final_path}")
        print(f"[competition] cancelled, partial output: {final_path}")
        raise
    except KeyboardInterrupt:
        final_path = persist_final(paths, results, "error")
        append_report(paths, f"KEYBOARD_INTERRUPT completed={completed_count(results)}/{len(requests)} output={final_path}")
        print(f"[competition] interrupted, partial output: {final_path}")
    except Exception as exc:
        final_path = persist_final(paths, results, "error")
        append_report(paths, f"FATAL_ERROR_CLI completed={completed_count(results)}/{len(requests)} error={exc} output={final_path}")
        print(f"[competition] error, partial output: {final_path}")
        raise


if __name__ == "__main__":
    asyncio.run(run())
