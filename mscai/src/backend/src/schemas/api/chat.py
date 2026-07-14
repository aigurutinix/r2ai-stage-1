"""Schema cho endpoint chat và batch theo format bài thi."""
from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field

from src.schemas.legal import LegalAnswerRequest, LegalAnswerResponse


class LegalRuntimeConfig(BaseModel):
    """Config runtime tối thiểu để UI chọn endpoint chat phù hợp."""

    chat_streaming: bool = True
    token_streaming: bool = True
    competition_enabled: bool = False


class ChatRequest(BaseModel):
    """Request kiểu chat, được map nội bộ sang ``LegalAnswerRequest``."""

    message: str
    session_id: str | None = None
    databases: list[str] = Field(default_factory=lambda: ["default"])
    top_k: int = 8
    competition_mode: bool | None = None


class ChatResponse(BaseModel):
    """Response kiểu chat, giữ lại debug tool-call để dễ quan sát agent."""

    session_id: str | None = None
    message: str
    answer: LegalAnswerResponse
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class ChatStreamMessagePayload(BaseModel):
    """Payload tiến độ có cấu trúc để UI biết request đang dừng ở đâu."""

    message: str
    stage: str = "request"
    status: Literal["started", "running", "completed", "warning", "error"] = "running"
    elapsed_ms: int | None = None
    detail: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatStreamStatusEvent(BaseModel):
    """Event báo UI biết agent đang ở bước xử lý nào."""

    event: Literal["status"] = "status"
    data: ChatStreamMessagePayload


class ChatStreamTokenPayload(BaseModel):
    """Payload token do LLM stream ra trong node answer."""

    token: str
    stage: str = "answer"


class ChatStreamTokenEvent(BaseModel):
    """Event token-by-token để UI append vào bubble assistant."""

    event: Literal["token"] = "token"
    data: ChatStreamTokenPayload


class ChatStreamResultEvent(BaseModel):
    """Event chứa response cuối cùng của chat."""

    event: Literal["result"] = "result"
    data: ChatResponse


class ChatStreamDoneEvent(BaseModel):
    """Event báo stream đã hoàn tất."""

    event: Literal["done"] = "done"
    data: ChatStreamMessagePayload


class ChatStreamErrorEvent(BaseModel):
    """Event báo lỗi runtime để UI hiển thị trong bubble assistant."""

    event: Literal["error"] = "error"
    data: ChatStreamMessagePayload


ChatStreamEvent: TypeAlias = (
    ChatStreamStatusEvent
    | ChatStreamTokenEvent
    | ChatStreamResultEvent
    | ChatStreamDoneEvent
    | ChatStreamErrorEvent
)


class CompetitionRecord(BaseModel):
    """Một dòng kết quả submit tối giản cho cuộc thi."""

    id: int | None = None
    question: str
    answer: str
    relevant_docs: list[str] = Field(default_factory=list)
    relevant_articles: list[str] = Field(default_factory=list)


class CompetitionBatchRequest(BaseModel):
    """Batch request; mỗi item dùng đúng schema hỏi đáp đơn lẻ."""

    items: list[LegalAnswerRequest]


class CompetitionBatchResponse(BaseModel):
    """Batch response có helper xuất ra ``results.json``."""

    results: list[LegalAnswerResponse]

    def to_results_json(self) -> list[dict[str, Any]]:
        """Chỉ lấy các field mà bài thi thường yêu cầu khi submit."""

        return [item.to_competition_record() for item in self.results]
