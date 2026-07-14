from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    question_id: int
    question: str 

    search_targets: List[Dict[str, Any]] 
    plan: Dict[str, Any]
    planner_think: str

    retrieved_documents: List[Dict[str, Any]]

    extracted_evidence: str 
    relevant_chunk_ids: List[str]
    evidence_id_map: Dict[str, str]
    applied_chunk_ids: List[str]
    candidate_but_not_applied_chunk_ids: List[str]
    evidence_selection_notes: str

    search_retries: int
