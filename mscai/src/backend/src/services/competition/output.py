"""Ghi output và report cho competition job.

Module này không phụ thuộc FastAPI/UI. Endpoint và script tmux đều dùng chung để
đảm bảo chạy tới đâu lưu tới đó, khi lỗi/dừng vẫn có file report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompetitionOutputPaths:
    """Các đường dẫn output của một lần chạy competition."""

    run_id: str
    output_dir: Path
    running_path: Path
    report_path: Path


def new_output_paths(output_dir: Path) -> CompetitionOutputPaths:
    """Tạo bộ path cho một competition run mới."""

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return CompetitionOutputPaths(
        run_id=run_id,
        output_dir=output_dir,
        running_path=output_dir / f"competition_{run_id}_running.json",
        report_path=output_dir / "report.log",
    )


def final_output_path(paths: CompetitionOutputPaths, status: str) -> Path:
    """Path file final success/error cho run hiện tại."""

    return paths.output_dir / f"competition_{paths.run_id}_{status}.json"


def write_json(path: Path, data: Any) -> None:
    """Ghi JSON an toàn bằng file tạm rồi replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def append_report(paths: CompetitionOutputPaths, message: str) -> None:
    """Append một dòng report có timestamp."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.report_path.open("a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] [{paths.run_id}] {message}\n")


def completed_count(results: list[dict | None]) -> int:
    """Đếm số câu đã có record output."""

    return sum(1 for item in results if item is not None)


def compact_results(results: list[dict | None]) -> list[dict]:
    """Bỏ slot None để ghi JSON hợp lệ cho các câu đã chạy xong."""

    return [item for item in results if item is not None]


def output_status(results: list[dict]) -> str:
    """Trả success/error dựa trên record lỗi trong output."""

    has_error = any(str(item.get("answer", "")).startswith("Lỗi khi xử lý câu hỏi:") for item in results)
    return "error" if has_error else "success"


def persist_progress(paths: CompetitionOutputPaths, results: list[dict | None]) -> None:
    """Lưu output partial sau mỗi câu hoàn tất."""

    write_json(paths.running_path, compact_results(results))


def persist_final(paths: CompetitionOutputPaths, results: list[dict | None], status: str | None = None) -> Path:
    """Lưu file final success/error và trả đường dẫn."""

    final_results = compact_results(results)
    final_status = status or output_status(final_results)
    path = final_output_path(paths, final_status)
    write_json(path, final_results)
    return path
