"""Base class chung cho mọi agent trong hệ thống.

Hiện tại project chỉ dùng một agent pháp lý, nhưng vẫn giữ base class để chuẩn
hóa lifecycle: request -> state -> graph/fallback -> response. Khi muốn nâng cấp
agent hoặc thêm agent mới, chỉ cần override các điểm mở rộng ở đây.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import uuid
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from src.services.agents.base.state import AgentState

RequestT = TypeVar("RequestT", bound=BaseModel)
ResponseT = TypeVar("ResponseT", bound=BaseModel)
StateT = TypeVar("StateT", bound=AgentState)


class BaseAgent(ABC, Generic[RequestT, ResponseT, StateT]):
    """Khung chạy chung cho agent.

    Concrete agent chỉ cần định nghĩa cách tạo state ban đầu, cách compile graph,
    cách chạy fallback khi thiếu LangGraph và cách build response. Public method
    ``answer`` nhờ vậy luôn ổn định với mọi agent.
    """

    name: str = "base-agent"
    description: str = "Abstract base agent"

    def __init__(self) -> None:
        # Nếu langgraph được cài, graph sẽ được compile một lần khi khởi tạo.
        # Nếu không, self.graph=None và agent dùng đường chạy thủ công.
        self.graph: Any | None = self._compile_graph()

    async def answer(self, request: RequestT) -> ResponseT:
        """Chạy trọn lifecycle từ request bên ngoài tới response API."""

        state = self.build_initial_state(request)
        state = await self.run_graph(state, self.build_graph_config(request))
        return self.build_response(state, request)

    def build_graph_config(self, request: RequestT) -> dict[str, Any]:
        """Tạo config LangGraph, dùng session_id làm thread_id cho checkpoint memory."""

        thread_id = getattr(request, "session_id", None) or uuid.uuid4().hex
        return {"configurable": {"thread_id": thread_id}}

    async def run_graph(self, state: StateT, config: dict[str, Any] | None = None) -> StateT:
        """Chạy LangGraph nếu có, nếu không chạy fallback tuần tự."""

        if self.graph is not None:
            return await self.graph.ainvoke(state, config=config)
        return await self.run_without_graph(state)

    async def stream_graph(self, state: StateT, config: dict[str, Any] | None = None) -> StateT:
        """Chạy LangGraph bằng astream để endpoint SSE nhận update khi graph đang chạy."""

        if self.graph is None or not hasattr(self.graph, "astream"):
            return await self.run_without_graph(state)

        final_state: StateT = state
        async for chunk in self.graph.astream(state, config=config, stream_mode="updates"):
            if not isinstance(chunk, dict):
                continue
            for node_update in chunk.values():
                if isinstance(node_update, dict):
                    final_state = {**final_state, **node_update}
        return final_state

    @abstractmethod
    def build_initial_state(self, request: RequestT) -> StateT:
        """Chuyển request API thành state nội bộ cho graph."""

    @abstractmethod
    def build_response(self, state: StateT, request: RequestT) -> ResponseT:
        """Chuyển state cuối cùng thành response API."""

    @abstractmethod
    async def run_without_graph(self, state: StateT) -> StateT:
        """Đường chạy dự phòng khi môi trường chưa cài LangGraph."""

    @abstractmethod
    def _compile_graph(self) -> Any | None:
        """Compile LangGraph hoặc trả ``None`` để dùng fallback."""
