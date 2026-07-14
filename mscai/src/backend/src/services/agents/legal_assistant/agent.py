"""Legal assistant service: build state, graph, and response."""
from __future__ import annotations

from functools import partial
from typing import Awaitable, Callable

try:
    from langchain_core.messages import HumanMessage
except ImportError:  # pragma: no cover - fallback khi chưa cài LangChain
    HumanMessage = None

from src.config import get_settings
from src.schemas.legal import LegalAnswerRequest, LegalAnswerResponse
from src.services.agents.base.client import BaseAgent
from src.services.agents.base.context import AgentContext
from src.services.agents.legal_assistant.node import (
    analyze_intent_node,
    format_submission_node,
    generate_answer_node,
    llm_filter_node,
    prepare_retrieval_query_node,
    rerank_node,
    retrieve_node,
)
from src.services.agents.legal_assistant.state import LegalAssistantState
from src.services.agents.progress import emit_progress, reset_progress_callback, set_progress_callback
from src.services.vector_store import VectorStoreFactory, VectorStoreRegistry, vector_store_registry


class LegalAssistantAgent(BaseAgent[LegalAnswerRequest, LegalAnswerResponse, LegalAssistantState]):
    """Agent pháp lý chính của backend."""

    name = "legal-assistant"
    description = "Vietnamese legal retrieval and grounded QA agent"

    def __init__(self, registry: VectorStoreRegistry = vector_store_registry, llm=None) -> None:
        self.registry = registry
        self.llm = llm
        self.settings = get_settings()
        self.store_factory = VectorStoreFactory(self.settings.legal_assistant.vector_store)
        super().__init__()

    def build_initial_state(self, request: LegalAnswerRequest) -> LegalAssistantState:
        competition_mode = (
            request.competition_mode
            if request.competition_mode is not None
            else self.settings.legal_assistant.competition.enabled
        )
        context = AgentContext(
            session_id=request.session_id,
            top_k=request.top_k or self.settings.legal_assistant.vector_store.top_k,
        )
        return {
            "question_id": request.id,
            "session_id": request.session_id,
            "question": request.question,
            "competition_mode": bool(competition_mode),
            "retrieval_question": request.question,
            "retrieval_mode": _retrieval_mode(self.settings),
            "query_variants": [request.question],
            "messages": [HumanMessage(content=request.question)] if HumanMessage else [],
            "context": context,
            "tool_calls": [],
            "debug": {},
        }

    async def answer_with_progress(
        self,
        request: LegalAnswerRequest,
        callback: Callable[[dict], Awaitable[None]],
    ) -> LegalAnswerResponse:
        """Chạy agent và phát tiến độ riêng cho request streaming hiện tại."""

        token = set_progress_callback(callback)
        try:
            await emit_progress(
                "memory",
                "started",
                "Đang khôi phục short-memory của đoạn chat",
                detail="LangGraph dùng session_id làm thread_id.",
                metadata={"enabled": self.settings.short_memory.enabled, "session_id": request.session_id},
            )
            state = self.build_initial_state(request)
            state = await self.stream_graph(state, self.build_graph_config(request))
            return self.build_response(state, request)
        finally:
            reset_progress_callback(token)

    def build_response(self, state: LegalAssistantState, request: LegalAnswerRequest) -> LegalAnswerResponse:
        debug = state.get("debug", {}).copy() if request.include_debug else {}
        if request.include_debug:
            debug.update(
                {
                    "tool_calls": state.get("tool_calls", []),
                    "legal_flag": state.get("legal_flag"),
                    "competition_mode": state.get("competition_mode", False),
                    "retrieval_mode": state.get("retrieval_mode"),
                    "retrieval_question": state.get("retrieval_question"),
                    "query_variants": state.get("query_variants", []),
                    "rewritten_question": state.get("rewritten_question"),
                    "hypothetical_answer": state.get("hypothetical_answer"),
                    "skip_retrieval": state.get("skip_retrieval", False),
                    "reranker_enabled": self.settings.legal_assistant.reranker.enabled,
                    "reranker_threshold": self.settings.legal_assistant.reranker.threshold,
                    "reranked_count": len(state.get("reranked", [])),
                    "llm_filter_enabled": self.settings.legal_assistant.llm_filter.enabled,
                    "llm_filtered_count": len(state.get("llm_filtered", [])),
                    "selected_count": len(state.get("selected_articles", [])),
                    "memory_enabled": self.settings.short_memory.enabled,
                }
            )
        return LegalAnswerResponse(
            id=state.get("question_id"),
            session_id=state.get("session_id"),
            question=state["question"],
            answer=state.get("answer", ""),
            relevant_docs=state.get("relevant_docs", []),
            relevant_articles=state.get("relevant_articles", []),
            selected_articles=state.get("selected_articles", []),
            debug=debug,
        )

    def _compile_graph(self):
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.graph import END, StateGraph
        except ImportError:
            return None

        workflow = StateGraph(LegalAssistantState)
        workflow.add_node("analyze_intent", partial(analyze_intent_node, self))
        workflow.add_node("prepare_retrieval_query", partial(prepare_retrieval_query_node, self))
        workflow.add_node("retrieve", partial(retrieve_node, self))
        workflow.add_node("rerank", partial(rerank_node, self))
        workflow.add_node("llm_filter", partial(llm_filter_node, self))
        workflow.add_node("generate_answer", partial(generate_answer_node, self))
        workflow.add_node("format_submission", partial(format_submission_node, self))

        workflow.set_entry_point("analyze_intent")
        workflow.add_edge("analyze_intent", "prepare_retrieval_query")
        workflow.add_edge("prepare_retrieval_query", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "llm_filter")
        workflow.add_edge("llm_filter", "generate_answer")
        workflow.add_edge("generate_answer", "format_submission")
        workflow.add_edge("format_submission", END)

        checkpointer = InMemorySaver() if self.settings.short_memory.enabled else None
        return workflow.compile(checkpointer=checkpointer, name="legal-assistant-workflow")

    async def run_without_graph(self, state: LegalAssistantState) -> LegalAssistantState:
        for node in (
            analyze_intent_node,
            prepare_retrieval_query_node,
            retrieve_node,
            rerank_node,
            llm_filter_node,
            generate_answer_node,
            format_submission_node,
        ):
            state = await node(self, state)
        return state


def _retrieval_mode(settings) -> str:
    """Tên mode retrieval hiển thị trong debug/progress."""

    rewrite = settings.legal_assistant.rewrite.enabled
    hyde = settings.legal_assistant.hyde.enabled
    if rewrite and hyde:
        return "rewrite+hyde"
    if rewrite:
        return "rewrite"
    if hyde:
        return "hyde"
    return "none"
