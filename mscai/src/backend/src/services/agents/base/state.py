"""TypedDict mô tả state được truyền giữa các node LangGraph."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

try:
    from langchain_core.messages import BaseMessage
    from langgraph.graph.message import add_messages
except ImportError:  # pragma: no cover - fallback khi chưa cài LangGraph/LangChain
    BaseMessage = Any

    def add_messages(left, right):
        return [*(left or []), *(right or [])]

from src.schemas.legal import LegalArticle, RetrievedCandidate
from src.services.agents.base.context import AgentContext


class AgentState(TypedDict, total=False):
    """State mutable đi từ node này sang node khác.

    ``total=False`` cho phép mỗi node chỉ thêm field mình tạo ra. Ví dụ node
    prepare query thêm ``retrieval_question``, node retrieve thêm
    ``selected_articles`` và node format thêm nguồn trả về.
    """

    question_id: int | None
    session_id: str | None
    question: str
    competition_mode: bool
    legal_flag: str
    rewritten_question: str
    hypothetical_answer: str
    retrieval_question: str
    retrieval_mode: str
    query_variants: list[str]
    search_spaces: list[str]
    skip_retrieval: bool
    retrieval_top_k: int
    context: AgentContext
    messages: Annotated[list[BaseMessage], add_messages]
    tool_calls: list[dict[str, Any]]
    retrieved: list[RetrievedCandidate]
    reranked: list[RetrievedCandidate]
    llm_filtered: list[RetrievedCandidate]
    selected_articles: list[LegalArticle]
    answer: str
    relevant_docs: list[str]
    relevant_articles: list[str]
    debug: dict[str, Any]
