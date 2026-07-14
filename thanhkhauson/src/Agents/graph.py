from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from pydantic import BaseModel, Field
from Agents.llm_client import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from Agents.mem.state import AgentState
from Agents.tools.search_legal import (
    perform_batch_hybrid_search, perform_batch_local_rerank,
    hybrid_search_tool
)
from Agents.logs.agent_logger import logger
from Agents.config import (
    MAIN_LLM_TEMPERATURE, MAIN_LLM_TOP_P, MAIN_LLM_TOP_K, MAIN_LLM_ENABLE_THINKING, PLANNER_THINKING_TOKEN_BUDGET,
    COMPRESS_LLM_TEMPERATURE, COMPRESS_LLM_TOP_P, COMPRESS_LLM_TOP_K, COMPRESS_LLM_ENABLE_THINKING, REASONING_THINKING_TOKEN_BUDGET,
    REASONING_ENABLE_TOOLS, REASONING_MAX_TOOL_CALLS, RETRIEVER_TOP_K, RERANKER_TOP_K
)
from typing import Any
import json
import os
import re
import threading
import time

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")
_filter_stats_log_lock = threading.Lock()

with open(os.path.join(SKILLS_DIR, "vietnamese_legal_hybrid_search_analysis.md"), "r", encoding="utf-8") as f:
    PLANNER_SKILL_PROMPT = f.read()
with open(os.path.join(SKILLS_DIR, "vietnamese_legal_reasoning_generation.md"), "r", encoding="utf-8") as f:
    REASONING_SKILL_PROMPT = f.read()

class MetadataFilters(BaseModel):
    applicable_time_point: str = Field(description="Mốc thời gian xảy ra sự kiện để quét văn bản tương ứng (nếu có, không có để 'Hiện tại')")

class SearchTarget(BaseModel):
    purpose: str = Field(description="Mục đích cụ thể của mục tiêu tra cứu này (Ví dụ: Tra cứu nghĩa vụ nền hoặc điều kiện hưởng)")
    expected_evidence_type: str = Field(description="Nhãn ngắn mô tả loại căn cứ mong muốn, ví dụ: definition, condition, permission_prohibition, obligation, penalty, procedure, exception, remedial_measures, compensation, authority, validity, scope, document_hierarchy; có thể dùng nhãn khác nếu phù hợp câu hỏi")
    bm25_query: str = Field(description="Từ khóa pháp lý ngắn gọn, cứng, cốt lõi, BẮT BUỘC bao gồm tên lĩnh vực luật (dùng cho Elasticsearch)")
    dense_query: str = Field(description="Câu truy vấn tự nhiên, giàu ngữ cảnh, BẮT BUỘC bao gồm tên lĩnh vực luật (dùng cho Vector DB)")

class PlannerOutput(BaseModel):
    intent: str = Field(description="Mục tiêu tra cứu chính của người dùng")
    actors: list[str] = Field(description="Danh sách các chủ thể liên quan trong tình huống")
    events: list[str] = Field(description="Hành vi hoặc sự kiện pháp lý xảy ra")
    legal_issues: list[str] = Field(description="Các vấn đề pháp lý cần tra cứu và làm rõ")
    metadata_filters: MetadataFilters = Field(description="Các bộ lọc dữ liệu")
    search_targets: list[SearchTarget] = Field(description="Danh sách các mục tiêu tra cứu độc lập (tách nhỏ câu hỏi thành nhiều phần)")

def _pydantic_to_dict(model: Any) -> dict:
    if model is None:
        raise ValueError("Structured output returned None")
    if isinstance(model, dict):
        return model
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

def _validate_structured_value(schema: type[BaseModel], value: Any):
    if hasattr(schema, "model_validate"):
        return schema.model_validate(value)
    return schema.parse_obj(value)

def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")

def _extract_json_object(text: str) -> Any:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[idx:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON object found in model response")

def _invoke_structured_with_retries(chain, payload: dict[str, Any], label: str, max_attempts: int = 3):
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = chain.invoke(payload)
            if result is None:
                raise ValueError("Structured output returned None")
            return result
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                logger.warning(f"{label} structured output lỗi lần {attempt}/{max_attempts}, retry: {e}")
            else:
                logger.error(f"{label} structured output lỗi sau {max_attempts} lần: {e}")
    raise RuntimeError(f"{label} structured output failed after {max_attempts} attempts") from last_error

def _invoke_structured_json_fallback(
    prompt: ChatPromptTemplate,
    llm,
    payload: dict[str, Any],
    schema: type[BaseModel],
    label: str,
    max_attempts: int = 2,
):
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    base_messages = prompt.format_messages(**payload)
    instruction = (
        "Endpoint hiện tại không trả về tool call ổn định. "
        "Hãy trả về DUY NHẤT một JSON object hợp lệ, không markdown, không giải thích, "
        "khớp schema sau:\n"
        f"{schema_json}"
    )

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            messages = base_messages + [HumanMessage(content=instruction)]
            if last_error is not None:
                messages.append(HumanMessage(content=f"Lỗi parse lần trước: {last_error}. Hãy sửa và chỉ trả về JSON hợp lệ."))
            response = llm.invoke(messages)
            raw_text = _content_to_text(getattr(response, "content", response))
            value = _extract_json_object(raw_text)
            return _validate_structured_value(schema, value)
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                logger.warning(f"{label} JSON fallback lỗi lần {attempt}/{max_attempts}, retry: {e}")
            else:
                logger.error(f"{label} JSON fallback lỗi sau {max_attempts} lần: {e}")
    raise RuntimeError(f"{label} JSON fallback failed after {max_attempts} attempts") from last_error

def _required_structured_tool(llm, schema: type[BaseModel]):
    from langchain_core.output_parsers.openai_tools import PydanticToolsParser

    return llm.bind_tools([schema], tool_choice="required") | PydanticToolsParser(
        tools=[schema],
        first_tool_only=True,
    )

def _structured_output_mode(label: str) -> str:
    env_by_label = {
        "Planner": "PLANNER_STRUCTURED_OUTPUT_MODE",
        "Compress Node": "COMPRESS_STRUCTURED_OUTPUT_MODE",
        "Applied Evidence Node": "APPLIED_EVIDENCE_STRUCTURED_OUTPUT_MODE",
    }
    specific_env = env_by_label.get(label)
    if specific_env and os.getenv(specific_env):
        return os.getenv(specific_env, "auto").strip().lower()
    return os.getenv("STRUCTURED_OUTPUT_MODE", "auto").strip().lower()

def _invoke_structured_auto(
    prompt: ChatPromptTemplate,
    llm,
    structured_chain,
    payload: dict[str, Any],
    schema: type[BaseModel],
    label: str,
):
    mode = _structured_output_mode(label)
    if mode == "json":
        return _invoke_structured_json_fallback(prompt, llm, payload, schema, label, max_attempts=3)

    try:
        return _invoke_structured_with_retries(
            structured_chain,
            payload,
            label,
        )
    except Exception as e:
        if mode == "tool":
            raise
        logger.warning(f"{label} structured output dùng tool thất bại, chuyển sang JSON fallback: {e}")
        return _invoke_structured_json_fallback(
            prompt,
            llm,
            payload,
            schema,
            label,
        )

def _assign_target_ids(search_targets: list[dict]) -> list[dict]:
    normalized_targets = []
    for idx, target in enumerate(search_targets, start=1):
        target_dict = dict(target)
        target_dict["target_id"] = f"T_{idx:02d}"
        normalized_targets.append(target_dict)
    return normalized_targets

def _format_evidence_with_temp_ids(docs: list[dict], start_index: int = 1) -> str:
    parts = []
    for idx, doc in enumerate(docs, start=start_index):
        parts.append(f"--- Nguồn (ID: DOC_{idx}) ---\n{doc.get('text', '')}")
    return "\n\n".join(parts)

def _build_evidence_id_map(docs: list[dict], start_index: int = 1) -> dict[str, str]:
    id_map = {}
    for idx, doc in enumerate(docs, start=start_index):
        real_id = str(doc.get("id") or "").strip()
        if real_id:
            id_map[f"DOC_{idx}"] = real_id
    return id_map

def _question_preview(question: str, max_chars: int = 160) -> str:
    preview = " ".join(str(question or "").split())
    return preview[:max_chars].rstrip()

def _log_filter_stats(label: str, total: int, kept: int, **extra: Any) -> None:
    total = max(0, int(total or 0))
    kept = max(0, int(kept or 0))
    removed = max(0, total - kept)
    removed_pct = (removed / total * 100.0) if total else 0.0
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "node": label.strip("[]"),
        "input": total,
        "kept": kept,
        "removed": removed,
        "removed_pct": round(removed_pct, 3),
        **extra,
    }
    extra_text = " ".join(f"{key}={value}" for key, value in extra.items())
    if extra_text:
        extra_text = " " + extra_text
    logger.info(
        "%s input=%s kept=%s removed=%s removed_pct=%.1f%%%s",
        label,
        total,
        kept,
        removed,
        removed_pct,
        extra_text,
    )
    log_path = os.getenv("FILTER_STATS_LOG_PATH", "").strip()
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with _filter_stats_log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Không ghi được filter stats log: {e}")

def planner_node(state: AgentState):
    """Phân tích câu hỏi, gọi LLM và ép trả về JSON cấu trúc chuẩn theo Skill."""
    question = state.get("question", "")
    llm = get_llm(
        temperature=MAIN_LLM_TEMPERATURE,
        top_p=MAIN_LLM_TOP_P,
        top_k=MAIN_LLM_TOP_K,
        enable_thinking=MAIN_LLM_ENABLE_THINKING,
        thinking_token_budget=PLANNER_THINKING_TOKEN_BUDGET
    )
    structured_llm = _required_structured_tool(llm, PlannerOutput)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=PLANNER_SKILL_PROMPT),
        ("human", "{question}")
    ])
    chain = prompt | structured_llm

    payload = {"question": question}
    parsed_result = _invoke_structured_auto(
        prompt,
        llm,
        chain,
        payload,
        PlannerOutput,
        "Planner",
    )
    plan = _pydantic_to_dict(parsed_result)
    search_targets = _assign_target_ids(plan.get("search_targets", []))
    plan["search_targets"] = search_targets

    return {
        "search_targets": search_targets,
        "plan": plan,
        "planner_think": "",
        "search_retries": 0
    }

def batch_hybrid_search_node(state: AgentState):
    search_targets = state.get("search_targets", [])
    if not search_targets:
        return {"retrieved_documents": []}
        
    bm25_queries = [t.get("bm25_query", "") for t in search_targets]
    dense_queries = [t.get("dense_query", "") for t in search_targets]
    target_ids = [t.get("target_id", "") for t in search_targets]
    
    raw_docs_list = perform_batch_hybrid_search(bm25_queries, dense_queries, top_k=RETRIEVER_TOP_K)
    reranked_docs_list = perform_batch_local_rerank(dense_queries, raw_docs_list, top_k=RERANKER_TOP_K)

    all_retrieved_documents = []
    for target_id, docs in zip(target_ids, reranked_docs_list):
        for doc in docs:
            doc["target_id"] = target_id
            all_retrieved_documents.append(doc)
            
    return {"retrieved_documents": all_retrieved_documents} 

class CompressOutput(BaseModel):
    relevant_chunk_ids: list[str] = Field(description="Danh sách ID (Ví dụ: DOC_1, DOC_2) của các tài liệu THỰC SỰ TRỰC TIẾP trả lời câu hỏi. Bỏ qua các ID không có giá trị.")

def compress_node(state: AgentState):
    docs = state.get("retrieved_documents", [])
    question = state.get("question", "")
    question_id = state.get("question_id")
    search_targets = state.get("search_targets", [])
    total_docs = len(docs)
    
    if not docs:
        _log_filter_stats(
            "[Compress Stats]",
            0,
            0,
            question_id=question_id,
            question_preview=_question_preview(question),
            reason="no_docs",
        )
        return {
            "extracted_evidence": "Không tìm thấy căn cứ pháp lý nào phù hợp từ cơ sở dữ liệu.",
            "relevant_chunk_ids": [],
            "evidence_id_map": {},
        }
        
    from collections import defaultdict
    docs_by_target = defaultdict(list)
    for doc in docs:
        docs_by_target[doc.get("target_id", "Unknown")].append(doc)
        
    context_text = ""
    doc_mapping = {}
    doc_counter = 1

    for target in search_targets:
        t_id = target.get('target_id', 'Unknown')
        purpose = target.get('purpose', '')
        evidence_type = target.get('expected_evidence_type', '')
        dense_query = target.get('dense_query', '')
        
        context_text += f"=========================================\n"
        context_text += f"[MỤC TIÊU TRA CỨU: {t_id}]\n"
        context_text += f"- Mục đích tìm kiếm: {purpose}\n"
        context_text += f"- Loại căn cứ cần tìm: {evidence_type}\n"
        context_text += f"- Truy vấn (Dense Query): {dense_query}\n"
        context_text += f"--- Các tài liệu tìm được cho mục tiêu này ---\n"
        
        target_docs = docs_by_target.get(t_id, [])
        if not target_docs:
            context_text += "(Không có tài liệu nào)\n\n"
            continue
            
        for doc in target_docs:
            real_chunk_id = doc.get("id", f"unknown_id_{doc_counter}")
            temp_id = f"DOC_{doc_counter}"
            doc_mapping[temp_id] = real_chunk_id
            
            text = doc.get("text", "")
            context_text += f">> Tài liệu: {temp_id} <<\nNội dung:\n{text}\n\n"
            doc_counter += 1

    llm = get_llm(
        temperature=COMPRESS_LLM_TEMPERATURE,
        top_p=COMPRESS_LLM_TOP_P,
        top_k=COMPRESS_LLM_TOP_K,
        enable_thinking=False,
    )
    structured_llm = _required_structured_tool(llm, CompressOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một Thẩm định viên pháp lý chuyên nghiệp. Nhiệm vụ của bạn là LỌC dữ liệu đầu vào để tìm ra căn cứ pháp lý chính xác nhất.
Quy tắc cực kỳ nghiêm ngặt:
1. Bạn sẽ nhận được các 'Mục tiêu tra cứu', ngay bên dưới mỗi mục tiêu là các 'Tài liệu' được hệ thống tìm về cho mục tiêu đó.
2. Đối chiếu từng tài liệu với 'Câu hỏi tổng thể' VÀ 'Mục đích tìm kiếm' của chính nhóm mục tiêu đó.
3. Lưu ý kiểm tra kỹ các metadata đi kèm (như legal_type, document_number, article_no, effect_date) để xác định đúng hiệu lực.
4. NẾU tài liệu KHÔNG LIÊN QUAN hoặc không giải quyết được Mục đích của nhóm mục tiêu đó -> BỎ QUA ID ĐÓ.
5. NẾU tài liệu LIÊN QUAN TRỰC TIẾP và đáp ứng đúng Mục đích -> Đưa ID (DOC_X) của nó vào danh sách trả về.
Mục đích của bạn là cung cấp bộ chứng cứ sạch, chính xác và có giá trị pháp lý cao nhất."""),
        ("human", "Câu hỏi tổng thể: {question}\n\nChi tiết kết quả tìm kiếm theo từng Mục tiêu:\n{context_text}")
    ])
    chain = prompt | structured_llm

    try:
        payload = {"question": question, "context_text": context_text}
        result = _invoke_structured_auto(
            prompt,
            llm,
            chain,
            payload,
            CompressOutput,
            "Compress Node",
        )
        valid_temp_ids = result.relevant_chunk_ids
        
        valid_real_ids = [doc_mapping[tid] for tid in valid_temp_ids if tid in doc_mapping]
        filtered_docs = [doc for doc in docs if doc.get("id") in valid_real_ids]

        if not filtered_docs:
            logger.warning("[!] Không còn tài liệu sau bước lọc. Dùng fallback từ rerank.")
            filtered_docs = docs[:RERANKER_TOP_K]
            valid_real_ids = [d.get("id") for d in filtered_docs]
            _log_filter_stats(
                "[Compress Stats]",
                total_docs,
                len(filtered_docs),
                question_id=question_id,
                question_preview=_question_preview(question),
                mode="fallback_empty_filter",
                selected_temp_ids=len(valid_temp_ids),
            )
        else:
            _log_filter_stats(
                "[Compress Stats]",
                total_docs,
                len(filtered_docs),
                question_id=question_id,
                question_preview=_question_preview(question),
                mode="structured",
                selected_temp_ids=len(valid_temp_ids),
            )

        extracted_evidence = _format_evidence_with_temp_ids(filtered_docs)
        evidence_id_map = _build_evidence_id_map(filtered_docs)
    except Exception as e:
        logger.error(f"[!] Lỗi tại Compress Node: {e}")
        filtered_docs = docs[:RERANKER_TOP_K]
        valid_real_ids = [d.get("id") for d in filtered_docs]
        extracted_evidence = _format_evidence_with_temp_ids(filtered_docs)
        evidence_id_map = _build_evidence_id_map(filtered_docs)
        _log_filter_stats(
            "[Compress Stats]",
            total_docs,
            len(filtered_docs),
            question_id=question_id,
            question_preview=_question_preview(question),
            mode="fallback_error",
        )
        
    return {
        "extracted_evidence": extracted_evidence,
        "relevant_chunk_ids": valid_real_ids,
        "evidence_id_map": evidence_id_map,
    }

def reasoning_node(state: AgentState):
    llm = get_llm(
        temperature=MAIN_LLM_TEMPERATURE,
        top_p=MAIN_LLM_TOP_P,
        top_k=MAIN_LLM_TOP_K,
        enable_thinking=MAIN_LLM_ENABLE_THINKING,
        thinking_token_budget=REASONING_THINKING_TOKEN_BUDGET
    )

    messages = state.get("messages", [])
    retries = state.get("search_retries", 0)

    if messages and getattr(messages[-1], "type", "") == "tool":
        retries += 1

    tools_enabled_for_call = REASONING_ENABLE_TOOLS and retries < REASONING_MAX_TOOL_CALLS

    if not REASONING_ENABLE_TOOLS:
        llm_with_tools = llm
    elif retries >= REASONING_MAX_TOOL_CALLS:
        logger.warning(
            "[!] Đã đạt giới hạn %s lần gọi tool, ép model trả lời luôn.",
            REASONING_MAX_TOOL_CALLS,
        )
        llm_with_tools = llm
    else:
        llm_with_tools = llm.bind_tools([hybrid_search_tool])

    question = state.get("question", "")
    evidence = state.get("extracted_evidence", "")
    plan = state.get("plan", {})

    initial_plan = json.dumps(plan, ensure_ascii=False, indent=2) if plan else "{}"

    retrieval_context = (
        "Dữ liệu hệ thống đã chuẩn bị cho lượt tư vấn này.\n\n"
        f"Kế hoạch truy hồi ban đầu do planner tạo ra:\n{initial_plan}\n\n"
        f"Căn cứ pháp lý ban đầu đã truy xuất và nén:\n{evidence}\n\n"
        "Nếu có các lượt tìm kiếm bổ sung bằng hybrid_search_tool, hãy đọc truy vấn và kết quả "
        "tìm thêm trực tiếp trong lịch sử tool call/tool message của cuộc hội thoại.\n\n"
        "Chỉ trả lời dựa trên các căn cứ pháp lý được cung cấp. "
    )
    
    if not REASONING_ENABLE_TOOLS:
        retrieval_context += "Không gọi công cụ bổ sung ở bước reasoning; hãy trả lời từ các căn cứ đã truy xuất và nén."
    elif retries >= REASONING_MAX_TOOL_CALLS:
        retrieval_context += f"(Hệ thống: BẠN ĐÃ ĐẠT GIỚI HẠN {REASONING_MAX_TOOL_CALLS} LẦN TÌM KIẾM. KHÔNG THỂ GỌI CÔNG CỤ NỮA. HÃY ĐƯA RA CÂU TRẢ LỜI CUỐI CÙNG NGAY BÂY GIỜ TỪ DỮ LIỆU HIỆN CÓ.)"
    else:
        retrieval_context += "Nếu căn cứ còn thiếu hoặc chưa đủ chắc chắn, hãy gọi hybrid_search_tool với truy vấn cụ thể để tìm thêm trước khi kết luận."

    clean_messages = [m for m in messages if getattr(m, "type", "") != "system"]
    system_msg = SystemMessage(content=REASONING_SKILL_PROMPT + "\n\n" + retrieval_context)

    def _invoke_reasoning(messages_to_send):
        response = llm_with_tools.invoke(messages_to_send)
        response_text = _content_to_text(getattr(response, "content", ""))
        if not getattr(response, "tool_calls", None) and not response_text.strip():
            if tools_enabled_for_call:
                logger.warning("[!] Reasoning Node trả về message rỗng. Retry lại với tool.")
                retry_response = llm_with_tools.invoke(messages_to_send)
                retry_text = _content_to_text(getattr(retry_response, "content", ""))
                if getattr(retry_response, "tool_calls", None) or retry_text.strip():
                    return retry_response

            logger.error("[!] Reasoning Node không hoàn thành: response rỗng sau retry, không fallback bỏ tool.")
            raise RuntimeError("Reasoning Node returned empty response after retry")
        return response

    if clean_messages and getattr(clean_messages[-1], "type", "") == "tool":
        response = _invoke_reasoning([system_msg] + clean_messages)
        return {"messages": [response], "search_retries": retries}

    current_human_msg = HumanMessage(content=question)
    response = _invoke_reasoning([system_msg] + clean_messages + [current_human_msg])

    return {"messages": [current_human_msg, response], "search_retries": retries}

class AppliedEvidenceOutput(BaseModel):
    applied_chunk_ids: list[str] = Field(description="Các ID tạm dạng Doc_1, Doc_2 được dùng trực tiếp để kết luận hoặc phân tích trong câu trả lời cuối.")
    candidate_but_not_applied_chunk_ids: list[str] = Field(description="Các ID tạm dạng Doc_1, Doc_2 có liên quan nhưng không được chốt áp dụng trực tiếp.")
    selection_notes: str = Field(description="Một câu ngắn giải thích tiêu chí chọn căn cứ áp dụng.")

def _last_final_ai_answer(messages):
    for msg in reversed(messages or []):
        if getattr(msg, "type", "") == "ai" and not getattr(msg, "tool_calls", None):
            return str(getattr(msg, "content", "") or "").strip()
    return ""

def _collect_evidence_docs(state: AgentState) -> list[dict]:
    """Gom nguồn đã đưa cho model và đổi UUID thật thành DOC_N cho bước hậu xử lý."""
    evidence_id_map = state.get("evidence_id_map", {}) or {}
    evidence_parts = [state.get("extracted_evidence", "") or ""]
    for msg in state.get("messages", []) or []:
        if getattr(msg, "type", "") == "tool":
            evidence_parts.append(str(getattr(msg, "content", "") or ""))

    source_pattern = re.compile(
        r"---\s*Nguồn \(ID:\s*([^)]+?)\s*\)\s*---\s*\n(.*?)(?=\n\n---\s*Nguồn \(ID:|\Z)",
        re.DOTALL,
    )

    docs_by_id = {}
    for evidence_text in evidence_parts:
        for match in source_pattern.finditer(evidence_text):
            source_id = match.group(1).strip()
            if source_id.startswith("DOC_") and source_id not in evidence_id_map:
                continue
            real_id = evidence_id_map.get(source_id, source_id)
            text = match.group(2).strip()
            if not real_id or not text:
                continue
            if real_id not in docs_by_id or len(text) > len(docs_by_id[real_id]):
                docs_by_id[real_id] = text

    if not docs_by_id:
        relevant_ids = set(state.get("relevant_chunk_ids", []) or [])
        for doc in state.get("retrieved_documents", []) or []:
            real_id = str(doc.get("id") or "").strip()
            text = str(doc.get("text") or "").strip()
            if real_id and text and (not relevant_ids or real_id in relevant_ids):
                docs_by_id[real_id] = text

    fake_docs = []
    for idx, (real_id, text) in enumerate(docs_by_id.items(), start=1):
        fake_docs.append({
            "fake_id": f"Doc_{idx}",
            "real_id": real_id,
            "text": text,
        })
    return fake_docs

def applied_evidence_node(state: AgentState):
    """Hỏi riêng model sau khi đã có answer để chốt nguồn thật sự được áp dụng."""
    question = state.get("question", "")
    question_id = state.get("question_id")
    messages = state.get("messages", [])
    answer = _last_final_ai_answer(messages)
    fake_docs = _collect_evidence_docs(state)
    fake_to_real = {doc["fake_id"]: doc["real_id"] for doc in fake_docs}
    valid_fake_ids = set(fake_to_real.keys())
    total_docs = len(fake_docs)

    evidence_context = "\n\n".join(
        f"[{doc['fake_id']}]\n{doc['text']}"
        for doc in fake_docs
    )

    if not answer or not fake_docs:
        _log_filter_stats(
            "[Applied Evidence Stats]",
            total_docs,
            0,
            question_id=question_id,
            question_preview=_question_preview(question),
            mode="no_answer_or_docs",
            candidate_not_applied=total_docs,
        )
        return {
            "applied_chunk_ids": [],
            "candidate_but_not_applied_chunk_ids": [doc["real_id"] for doc in fake_docs],
            "evidence_selection_notes": "Không đủ dữ liệu để chọn căn cứ áp dụng sau câu trả lời.",
        }

    llm = get_llm(
        temperature=MAIN_LLM_TEMPERATURE,
        top_p=MAIN_LLM_TOP_P,
        top_k=MAIN_LLM_TOP_K,
        enable_thinking=False
    )
    structured_llm = _required_structured_tool(llm, AppliedEvidenceOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là bước hậu xử lý độc lập sau khi câu trả lời cuối đã được tạo.
	Nhiệm vụ: đọc câu hỏi, câu trả lời cuối và các nguồn đã truy hồi; chọn ID tạm của nguồn thật sự được áp dụng trong câu trả lời.

Quy tắc:
1. Chỉ trả về ID tạm dạng Doc_1, Doc_2, ... xuất hiện ngay trước từng nguồn.
2. ID tạm và nội dung nguồn nằm liền nhau; hãy dựa vào chính nội dung dưới ID đó để chọn.
3. `applied_chunk_ids` chỉ gồm nguồn được dùng trực tiếp để kết luận hoặc phân tích.
4. Nguồn chỉ liên quan sơ bộ, bị bỏ qua, hoặc không đúng tình huống đưa vào `candidate_but_not_applied_chunk_ids`.
5. Nếu có văn bản mới/cũ, sửa đổi/bổ sung, thay thế, bãi bỏ hoặc hiệu lực không rõ, ưu tiên nguồn có khả năng đang áp dụng; nguồn còn nghi ngờ đưa vào `candidate_but_not_applied_chunk_ids`.
6. Nếu nhiều nguồn cùng quy định một nội dung pháp luật, ưu tiên nguồn là văn bản mới hơn hoặc có thời điểm hiệu lực phù hợp hơn; nguồn cũ hơn đưa vào `candidate_but_not_applied_chunk_ids`.
7. Nếu hai nguồn trùng nội dung về cùng một luật và một nguồn là VBHN/văn bản hợp nhất, ưu tiên nguồn còn lại không phải VBHN; đưa VBHN vào `candidate_but_not_applied_chunk_ids`.
8. Không viết lại câu trả lời pháp lý. Chỉ xuất đúng JSON theo schema."""),
        ("human", """Câu hỏi:
{question}

Câu trả lời cuối:
{answer}

Các nguồn ứng viên, mỗi ID tạm nằm ngay cạnh nội dung nguồn tương ứng:
{evidence_context}

Hãy chọn căn cứ áp dụng thực sự cho câu trả lời trên.""")
    ])

    try:
        payload = {
            "question": question,
            "answer": answer,
            "evidence_context": evidence_context,
        }
        result = _invoke_structured_auto(
            prompt,
            llm,
            prompt | structured_llm,
            payload,
            AppliedEvidenceOutput,
            "Applied Evidence Node",
        )

        def fake_ids_to_real(values):
            output = []
            seen = set()
            for value in values or []:
                fake_id = str(value or "").strip()
                if fake_id not in valid_fake_ids:
                    continue
                real_id = fake_to_real[fake_id]
                if real_id not in seen:
                    seen.add(real_id)
                    output.append(real_id)
            return output

        applied_real_ids = fake_ids_to_real(result.applied_chunk_ids)
        candidate_real_ids = fake_ids_to_real(result.candidate_but_not_applied_chunk_ids)
        _log_filter_stats(
            "[Applied Evidence Stats]",
            total_docs,
            len(applied_real_ids),
            question_id=question_id,
            question_preview=_question_preview(question),
            mode="structured",
            candidate_not_applied=len(candidate_real_ids),
        )

        return {
            "applied_chunk_ids": applied_real_ids,
            "candidate_but_not_applied_chunk_ids": candidate_real_ids,
            "evidence_selection_notes": result.selection_notes,
        }
    except Exception as e:
        logger.error(f"[!] Lỗi tại Applied Evidence Node: {e}")
        fallback_ids = state.get("relevant_chunk_ids", []) or []
        _log_filter_stats(
            "[Applied Evidence Stats]",
            total_docs,
            len(fallback_ids),
            question_id=question_id,
            question_preview=_question_preview(question),
            mode="fallback_error",
            candidate_not_applied=0,
        )
        return {
            "applied_chunk_ids": fallback_ids,
            "candidate_but_not_applied_chunk_ids": [],
            "evidence_selection_notes": "Fallback dùng relevant_chunk_ids do lỗi bước chọn căn cứ áp dụng.",
        }


workflow = StateGraph(AgentState)

workflow.add_node("planner_node", planner_node)
workflow.add_node("batch_hybrid_search_node", batch_hybrid_search_node)
workflow.add_node("compress_node", compress_node)
workflow.add_node("reasoning_node", reasoning_node)
workflow.add_node("applied_evidence_node", applied_evidence_node)
workflow.add_node("search_tool_node", ToolNode([hybrid_search_tool]))

workflow.add_edge(START, "planner_node")
workflow.add_edge("planner_node", "batch_hybrid_search_node")
workflow.add_edge("batch_hybrid_search_node", "compress_node")
workflow.add_edge("compress_node", "reasoning_node")

workflow.add_conditional_edges(
    "reasoning_node", 
    tools_condition, 
    {"tools": "search_tool_node", "__end__": "applied_evidence_node"}
)
workflow.add_edge("search_tool_node", "reasoning_node")
workflow.add_edge("applied_evidence_node", END)

from Agents.mem.checkpointer import get_checkpointer

def compile_graph(checkpointer: Any | None = None):
    """Compile graph với checkpointer được truyền vào; None nghĩa là stateless."""
    return workflow.compile(checkpointer=checkpointer)

memory = get_checkpointer()
app = compile_graph(memory)

stateless_app = compile_graph(None)
