"""Các node xử lý của legal assistant workflow."""
from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ImportError:  # pragma: no cover
    AIMessage = None
    HumanMessage = None
    SystemMessage = None

from src.schemas.legal import LegalArticle, RetrievalQuery, RetrievedCandidate
from src.services.agents.legal_assistant.prompt import (
    SYSTEM_PROMPT,
    build_hyde_messages,
    build_intent_messages,
    build_legal_context_message,
    build_llm_filter_messages,
    build_rewrite_query_messages,
)
from src.services.agents.legal_assistant.state import LegalAssistantState
from src.services.agents.legal_assistant.tools import search_legal_articles
from src.services.agents.progress import emit_progress, emit_token, has_progress_callback
from src.services.reranker.client import get_reranker_client


async def analyze_intent_node(runtime: Any, state: LegalAssistantState) -> LegalAssistantState:
    """Bước 1: phân tích query pháp luật hay chat thường."""

    started = perf_counter()
    await emit_progress(
        "memory",
        "completed",
        "Đã sẵn sàng ngữ cảnh hội thoại",
        metadata={"enabled": runtime.settings.short_memory.enabled},
    )
    if state.get("competition_mode"):
        state["legal_flag"] = "NEXT"
        state["skip_retrieval"] = False
        state.setdefault("tool_calls", []).append(
            {"name": "analyze_intent", "provider": "competition", "result": "NEXT"}
        )
        await emit_progress(
            "intent",
            "completed",
            "Competition mode: bỏ qua intent",
            elapsed_ms=_elapsed_ms(started),
            detail="Đi thẳng vào legal RAG để chạy tập test.",
            metadata={"legal_flag": "NEXT", "competition_mode": True},
        )
        return state

    await emit_progress(
        "intent",
        "started",
        "Đang phân tích câu hỏi có cần tra cứu pháp luật hay không",
        detail="Gọi LLM để chọn SKIP hoặc NEXT.",
    )
    flag = "NEXT"
    error: str | None = None
    if runtime.llm is not None:
        try:
            system_prompt, human_prompt = build_intent_messages(state["question"])
            flag = (await _invoke_prompt_messages(runtime, system_prompt, human_prompt)).strip().upper()
        except Exception as exc:  # pragma: no cover
            error = str(exc)
            state.setdefault("debug", {})["intent_error"] = error
    if flag != "SKIP":
        flag = "NEXT"
    state["legal_flag"] = flag
    state["skip_retrieval"] = flag == "SKIP"
    state.setdefault("tool_calls", []).append(
        {"name": "analyze_intent", "provider": "llm" if runtime.llm else "fallback", "result": flag}
    )
    await emit_progress(
        "intent",
        "warning" if error else "completed",
        "Phân tích ý định hoàn tất" if not error else "LLM intent lỗi, tiếp tục theo NEXT",
        elapsed_ms=_elapsed_ms(started),
        detail=error or ("Bỏ qua legal retrieval." if flag == "SKIP" else "Tiếp tục legal RAG."),
        metadata={"legal_flag": flag},
    )
    return state


async def prepare_retrieval_query_node(runtime: Any, state: LegalAssistantState) -> LegalAssistantState:
    """Bước 2: tạo query variants từ rewrite và/hoặc HyDE."""

    started = perf_counter()
    question = state["question"]
    rewrite_enabled = runtime.settings.legal_assistant.rewrite.enabled
    hyde_enabled = runtime.settings.legal_assistant.hyde.enabled
    mode = _retrieval_mode(rewrite_enabled, hyde_enabled)
    state["retrieval_mode"] = mode
    await emit_progress(
        "prepare_query",
        "started",
        f"Đang chuẩn bị truy vấn retrieval theo mode {mode}",
        detail="Không gọi retrieval nếu intent là SKIP." if state.get("skip_retrieval") else None,
        metadata={"mode": mode, "rewrite_enabled": rewrite_enabled, "hyde_enabled": hyde_enabled},
    )

    if state.get("skip_retrieval") or mode == "none":
        result = _set_retrieval_text(runtime, state, question, mode="none", provider="config")
        await emit_progress(
            "prepare_query",
            "completed",
            "Dùng trực tiếp câu hỏi gốc",
            elapsed_ms=_elapsed_ms(started),
            metadata={"mode": "none", "retrieval_question": question},
        )
        return result

    rewritten_question: str | None = None
    hypothetical_answer: str | None = None
    errors: list[str] = []
    provider = "llm" if runtime.llm is not None else "fallback"

    if runtime.llm is not None and rewrite_enabled:
        try:
            system_prompt, human_prompt = build_rewrite_query_messages(question)
            candidate = (await _invoke_prompt_messages(runtime, system_prompt, human_prompt)).strip()
            rewritten_question = _clean_retrieval_text(candidate, "rewrite") or question
            state["rewritten_question"] = rewritten_question
        except Exception as exc:  # pragma: no cover
            errors.append(f"rewrite: {exc}")
            state.setdefault("debug", {})["rewrite_error"] = str(exc)

    if runtime.llm is not None and hyde_enabled:
        try:
            system_prompt, human_prompt = build_hyde_messages(question)
            candidate = (await _invoke_prompt_messages(runtime, system_prompt, human_prompt)).strip()
            hypothetical_answer = _clean_retrieval_text(candidate, "hyde") or question
            state["hypothetical_answer"] = hypothetical_answer
        except Exception as exc:  # pragma: no cover
            errors.append(f"hyde: {exc}")
            state.setdefault("debug", {})["hyde_error"] = str(exc)

    retrieval_text = hypothetical_answer or rewritten_question or question
    state["retrieval_question"] = retrieval_text
    state["query_variants"] = _build_query_variants(runtime, question, rewritten_question, hypothetical_answer)
    state.setdefault("tool_calls", []).append(
        {
            "name": "prepare_retrieval_query",
            "provider": provider,
            "args": {"mode": mode},
            "result": retrieval_text,
            "query_variants": state["query_variants"],
        }
    )
    await emit_progress(
        "prepare_query",
        "warning" if errors else "completed",
        f"Đã chuẩn bị truy vấn {mode}" if not errors else "Một phần rewrite/HyDE lỗi, dùng phần còn lại",
        elapsed_ms=_elapsed_ms(started),
        detail="; ".join(errors) or None,
        metadata={
            "mode": mode,
            "provider": provider,
            "retrieval_question": retrieval_text,
            "query_variant_count": len(state["query_variants"]),
        },
    )
    return state


async def retrieve_node(runtime: Any, state: LegalAssistantState) -> LegalAssistantState:
    """Bước 3: search global top-k trên toàn bộ retrieval registry."""

    started = perf_counter()
    if state.get("skip_retrieval"):
        state["retrieved"] = []
        state["selected_articles"] = []
        state.setdefault("tool_calls", []).append({"name": "skip_retrieval", "provider": "backend", "num_results": 0})
        await emit_progress(
            "retrieval",
            "completed",
            "Bỏ qua retrieval",
            elapsed_ms=_elapsed_ms(started),
            detail="Intent là SKIP.",
        )
        return state

    search_spaces = runtime.registry.list_databases() or ["default"]
    query = RetrievalQuery(
        question=state.get("retrieval_question") or state["question"],
        original_question=state["question"],
        query_variants=state.get("query_variants", [state["question"]]),
        search_spaces=search_spaces,
        top_k=runtime.settings.legal_assistant.vector_store.top_k,
    )
    state["search_spaces"] = search_spaces
    state["retrieval_top_k"] = query.top_k
    search_mode = runtime.settings.legal_assistant.vector_store.mode
    progress_metadata = _retrieval_progress_metadata(query, search_mode)
    await emit_progress(
        "retrieval",
        "started",
        f"Đang chạy {search_mode} search",
        detail="Embedding query, tìm Chroma/BM25 và hợp nhất RRF." if search_mode == "hybrid" else None,
        metadata=progress_metadata,
    )
    error: str | None = None
    try:
        # Retrieval là code đồng bộ và có thể nặng; chạy trong thread để SSE
        # heartbeat vẫn tiếp tục báo thời gian chờ cho UI.
        candidates = await asyncio.to_thread(search_legal_articles, query, runtime.registry, runtime.store_factory)
    except Exception as exc:
        candidates = []
        error = str(exc)
        state.setdefault("debug", {})["local_retrieval_error"] = error

    state.setdefault("tool_calls", []).append(
        {
            "name": "search_legal_articles",
            "provider": "backend",
            "args": _retrieval_tool_args(query, progress_metadata),
            "num_results": len(candidates),
            "error": error,
        }
    )
    state["retrieved"] = candidates
    state["selected_articles"] = [candidate.article for candidate in candidates]
    await emit_progress(
        "retrieval",
        "error" if error else ("completed" if candidates else "warning"),
        "Retrieval lỗi" if error else f"Retrieval trả về {len(candidates)} kết quả",
        elapsed_ms=_elapsed_ms(started),
        detail=error or (None if candidates else "Không tìm thấy candidate phù hợp."),
        metadata={
            **progress_metadata,
            "num_results": len(candidates),
            "top_results": _top_retrieval_results(candidates, limit=5),
        },
    )
    return state


async def rerank_node(runtime: Any, state: LegalAssistantState) -> LegalAssistantState:
    """Bước 4: dùng reranker lọc điều luật theo threshold."""

    started = perf_counter()
    settings = runtime.settings.legal_assistant.reranker
    candidates = state.get("retrieved", [])
    if state.get("skip_retrieval") or not candidates:
        state["reranked"] = []
        state["selected_articles"] = []
        await emit_progress(
            "rerank",
            "completed",
            "Bỏ qua reranker",
            elapsed_ms=_elapsed_ms(started),
            detail="Không có candidate retrieval để rerank.",
            metadata={"enabled": settings.enabled, "num_candidates": len(candidates)},
        )
        return state

    if not settings.enabled:
        state["reranked"] = candidates
        state["selected_articles"] = [candidate.article for candidate in candidates]
        await emit_progress(
            "rerank",
            "completed",
            "Reranker đang tắt",
            elapsed_ms=_elapsed_ms(started),
            detail="Dùng nguyên kết quả retrieval.",
            metadata={"enabled": False, "num_candidates": len(candidates)},
        )
        return state

    query = state.get("retrieval_question") or state["question"]
    await emit_progress(
        "rerank",
        "started",
        "Đang rerank các điều luật tìm được",
        detail=f"Gọi Qwen3 reranker và lọc theo mode {settings.filter_mode}.",
        metadata={
            "model": settings.model,
            "filter_mode": settings.filter_mode,
            "threshold": settings.threshold,
            "min_gap": settings.min_gap,
            "min_keep": settings.min_keep,
            "num_candidates": len(candidates),
            "query": query,
        },
    )

    error: str | None = None
    scored: list[RetrievedCandidate] = []
    try:
        client = get_reranker_client()
        scores = await client.score_many(query, [candidate.article for candidate in candidates])
        for candidate, score in zip(candidates, scores):
            scored.append(candidate.model_copy(update={"score": score, "reason": f"rerank_score={score}"}))
        scored.sort(key=lambda item: item.score, reverse=True)
        kept, filter_info = _filter_reranked_candidates(scored, settings)
    except Exception as exc:  # pragma: no cover - endpoint/runtime guard
        error = str(exc)
        state.setdefault("debug", {})["reranker_error"] = error
        scored = candidates
        kept = candidates
        filter_info = {"filter_mode": settings.filter_mode, "fallback": True}

    if "filter_info" not in locals():
        filter_info = {"filter_mode": settings.filter_mode}

    state["reranked"] = scored
    state["selected_articles"] = [candidate.article for candidate in kept]
    state.setdefault("tool_calls", []).append(
        {
            "name": "rerank_legal_articles",
            "provider": "qwen3-reranker",
            "args": {
                "model": settings.model,
                "filter_mode": settings.filter_mode,
                "threshold": settings.threshold,
                "min_gap": settings.min_gap,
                "min_keep": settings.min_keep,
                "query": query,
            },
            "num_candidates": len(candidates),
            "num_kept": len(kept),
            "error": error,
        }
    )
    await emit_progress(
        "rerank",
        "warning" if error else "completed",
        "Reranker lỗi, dùng nguyên retrieval" if error else f"Rerank giữ {len(kept)}/{len(candidates)} kết quả",
        elapsed_ms=_elapsed_ms(started),
        detail=error,
        metadata={
            "model": settings.model,
            "filter_mode": settings.filter_mode,
            "threshold": settings.threshold,
            "min_gap": settings.min_gap,
            "min_keep": settings.min_keep,
            "num_candidates": len(candidates),
            "num_kept": len(kept),
            **filter_info,
            "top_results": _top_rerank_results(scored, kept, limit=5),
        },
    )
    return state


async def llm_filter_node(runtime: Any, state: LegalAssistantState) -> LegalAssistantState:
    """Bước 5: dùng LLM kiểm tra từng điều luật sau rerank là PASS hay DROP."""

    started = perf_counter()
    settings = runtime.settings.legal_assistant.llm_filter
    selected = state.get("selected_articles", [])
    if state.get("skip_retrieval") or not selected:
        state["llm_filtered"] = []
        await emit_progress(
            "llm_filter",
            "completed",
            "Bỏ qua LLM filter",
            elapsed_ms=_elapsed_ms(started),
            detail="Không có điều luật sau rerank để đánh giá.",
            metadata={"enabled": settings.enabled, "num_candidates": len(selected)},
        )
        return state

    if not settings.enabled or runtime.llm is None:
        state["llm_filtered"] = state.get("reranked", [])
        await emit_progress(
            "llm_filter",
            "completed",
            "LLM filter đang tắt",
            elapsed_ms=_elapsed_ms(started),
            detail="Dùng nguyên kết quả sau rerank.",
            metadata={"enabled": settings.enabled, "has_llm": runtime.llm is not None, "num_candidates": len(selected)},
        )
        return state

    selected_ids = {article.id for article in selected}
    candidates = [candidate for candidate in state.get("reranked", []) if candidate.article.id in selected_ids]
    if not candidates:
        candidates = [RetrievedCandidate(article=article, source="hybrid", score=0.0) for article in selected]

    query, query_source = _llm_filter_query(runtime, state)
    await emit_progress(
        "llm_filter",
        "started",
        "Đang để LLM đánh giá lại từng điều luật",
        detail="Mỗi điều luật sau rerank được chấm PASS/DROP theo query.",
        metadata={
            "enabled": True,
            "num_candidates": len(candidates),
            "max_concurrency": settings.max_concurrency,
            "min_keep": settings.min_keep,
            "filter_query_source": query_source,
            "filter_query": query,
        },
    )

    semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def judge(candidate: RetrievedCandidate) -> tuple[RetrievedCandidate, str, str | None]:
        async with semaphore:
            try:
                system_prompt, human_prompt = build_llm_filter_messages(query, candidate.article, query_source)
                result = (await _invoke_prompt_messages(runtime, system_prompt, human_prompt)).strip().upper()
                label = "PASS" if result.startswith("PASS") else "DROP"
                return candidate, label, None
            except Exception as exc:  # pragma: no cover - endpoint/runtime guard
                return candidate, "PASS", str(exc)

    judged = await asyncio.gather(*(judge(candidate) for candidate in candidates))
    errors = [error for _, _, error in judged if error]
    kept_candidates = [candidate for candidate, label, _ in judged if label == "PASS"]
    if len(kept_candidates) < settings.min_keep:
        kept_ids = {candidate.article.id for candidate in kept_candidates}
        for candidate in candidates:
            if len(kept_candidates) >= settings.min_keep:
                break
            if candidate.article.id not in kept_ids:
                kept_candidates.append(candidate)
                kept_ids.add(candidate.article.id)

    state["llm_filtered"] = kept_candidates
    state["selected_articles"] = [candidate.article for candidate in kept_candidates]
    state.setdefault("tool_calls", []).append(
        {
            "name": "llm_filter_legal_articles",
            "provider": "llm",
            "args": {
                "query": query,
                "query_source": query_source,
                "max_concurrency": settings.max_concurrency,
                "min_keep": settings.min_keep,
            },
            "num_candidates": len(candidates),
            "num_kept": len(kept_candidates),
            "num_errors": len(errors),
        }
    )
    await emit_progress(
        "llm_filter",
        "warning" if errors else "completed",
        f"LLM filter giữ {len(kept_candidates)}/{len(candidates)} kết quả",
        elapsed_ms=_elapsed_ms(started),
        detail=(f"Có {len(errors)} lỗi, các item lỗi được giữ lại." if errors else None),
        metadata={
            "num_candidates": len(candidates),
            "num_kept": len(kept_candidates),
            "num_dropped": len(candidates) - len(kept_candidates),
            "num_errors": len(errors),
            "filter_query_source": query_source,
            "filter_query": query,
            "top_results": _top_llm_filter_results(judged, kept_candidates, limit=5),
        },
    )
    return state


async def generate_answer_node(runtime: Any, state: LegalAssistantState) -> LegalAssistantState:
    """Bước 5: tổng hợp điều luật tìm được và trả lời user."""

    started = perf_counter()
    articles = None if state.get("skip_retrieval") else state.get("selected_articles", [])
    await emit_progress(
        "answer",
        "started",
        "Đang gọi LLM tổng hợp câu trả lời",
        detail=f"Đưa {len(articles or [])} điều luật vào context." if articles is not None else "Trả lời hội thoại thông thường.",
        metadata={"article_count": len(articles or []), "skip_retrieval": state.get("skip_retrieval", False)},
    )
    state["answer"] = await _chat_answer(runtime, state, articles)
    error = state.get("debug", {}).get("llm_error")
    await emit_progress(
        "answer",
        "warning" if error else "completed",
        "Đã tạo câu trả lời" if not error else "LLM trả lời lỗi, đã dùng fallback",
        elapsed_ms=_elapsed_ms(started),
        detail=str(error) if error else None,
        metadata={"answer_length": len(state.get("answer", ""))},
    )
    return state


async def format_submission_node(_: Any, state: LegalAssistantState) -> LegalAssistantState:
    """Tạo nguồn theo format bài thi và ghi AIMessage vào memory."""

    started = perf_counter()
    await emit_progress("format", "started", "Đang chuẩn hóa nguồn và lưu short-memory")
    if state.get("skip_retrieval"):
        state["relevant_docs"] = []
        state["relevant_articles"] = []
        _append_ai_message(state)
        await emit_progress(
            "format",
            "completed",
            "Đã hoàn tất câu trả lời không retrieval",
            elapsed_ms=_elapsed_ms(started),
        )
        return state

    docs: list[str] = []
    article_refs: list[str] = []
    for article in state.get("selected_articles", []):
        _append_once(article.doc_ref, docs)
        _append_once(article.article_ref, article_refs)
        for related_ref in sorted(article.extra):
            doc_ref, article_ref = normalize_related_ref(related_ref)
            _append_once(doc_ref, docs)
            _append_once(article_ref, article_refs)
    state["relevant_docs"] = docs
    state["relevant_articles"] = article_refs
    _append_ai_message(state)
    await emit_progress(
        "format",
        "completed",
        "Đã chuẩn hóa kết quả và cập nhật short-memory",
        elapsed_ms=_elapsed_ms(started),
        metadata={"document_count": len(docs), "article_count": len(article_refs)},
    )
    return state


def _elapsed_ms(started: float) -> int:
    return round((perf_counter() - started) * 1000)


def _retrieval_progress_metadata(query: RetrievalQuery, search_mode: str) -> dict[str, Any]:
    """Metadata ngắn gọn cho UI; search luôn là global top-k."""

    return {
        "mode": search_mode,
        "scope": "global",
        "top_k": query.top_k,
        "search_space_count": len(query.search_spaces),
    }


def _top_retrieval_results(candidates: list[RetrievedCandidate], limit: int = 5) -> list[dict[str, Any]]:
    """Rút gọn top retrieval candidates để UI debug score/rank dễ đọc."""

    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:limit], start=1):
        article = candidate.article
        score = candidate.score if candidate.score is not None else 0.0
        results.append(
            {
                "rank": candidate.rank or index,
                "score": round(float(score), 6),
                "source": candidate.source,
                "law_id": article.law_id,
                "law_name": article.law_name,
                "article": article.article,
                "article_title": article.article_title or "",
            }
        )
    return results


def _filter_reranked_candidates(
    candidates: list[RetrievedCandidate],
    settings: Any,
) -> tuple[list[RetrievedCandidate], dict[str, Any]]:
    """Lọc candidate sau rerank theo threshold tĩnh hoặc gap động."""

    if settings.filter_mode == "fixed":
        kept = [candidate for candidate in candidates if (candidate.score or 0.0) >= settings.threshold]
        return kept, {"filter_mode": "fixed", "effective_threshold": settings.threshold}

    if len(candidates) <= settings.min_keep:
        return candidates, {
            "filter_mode": "largest_gap",
            "largest_gap": 0.0,
            "cut_rank": len(candidates),
            "dynamic_threshold": None,
            "reason": "candidate_count <= min_keep",
        }

    gaps: list[dict[str, float | int]] = []
    for index in range(len(candidates) - 1):
        upper = float(candidates[index].score or 0.0)
        lower = float(candidates[index + 1].score or 0.0)
        gaps.append({"rank": index + 1, "gap": upper - lower, "upper": upper, "lower": lower})

    best_gap = max(gaps, key=lambda item: float(item["gap"]))
    largest_gap = float(best_gap["gap"])
    if largest_gap < settings.min_gap:
        return candidates, {
            "filter_mode": "largest_gap",
            "largest_gap": round(largest_gap, 6),
            "cut_rank": len(candidates),
            "dynamic_threshold": None,
            "reason": "largest_gap < min_gap",
        }

    cut_rank = max(int(best_gap["rank"]), settings.min_keep)
    dynamic_threshold = (float(best_gap["upper"]) + float(best_gap["lower"])) / 2
    return candidates[:cut_rank], {
        "filter_mode": "largest_gap",
        "largest_gap": round(largest_gap, 6),
        "cut_rank": cut_rank,
        "dynamic_threshold": round(dynamic_threshold, 6),
    }


def _top_rerank_results(
    candidates: list[RetrievedCandidate],
    kept: list[RetrievedCandidate],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Rút gọn rerank candidates để UI hiển thị score và trạng thái lọc."""

    kept_ids = {candidate.article.id for candidate in kept}
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:limit], start=1):
        article = candidate.article
        score = candidate.score if candidate.score is not None else 0.0
        results.append(
            {
                "rank": index,
                "score": round(float(score), 6),
                "source": candidate.source,
                "law_id": article.law_id,
                "law_name": article.law_name,
                "article": article.article,
                "article_title": article.article_title or "",
                "passed_threshold": article.id in kept_ids,
            }
        )
    return results


def _llm_filter_query(runtime: Any, state: LegalAssistantState) -> tuple[str, str]:
    """Chọn query dùng riêng cho LLM filter theo config rewrite/HyDE."""

    assistant = runtime.settings.legal_assistant
    if assistant.hyde.enabled and state.get("hypothetical_answer"):
        return state["hypothetical_answer"], "hyde"
    if assistant.rewrite.enabled and state.get("rewritten_question"):
        return state["rewritten_question"], "rewrite"
    if state.get("retrieval_question"):
        return state["retrieval_question"], state.get("retrieval_mode", "retrieval_question")
    return state["question"], "original"


def _top_llm_filter_results(
    judged: list[tuple[RetrievedCandidate, str, str | None]],
    kept: list[RetrievedCandidate],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Rút gọn kết quả LLM filter để UI hiển thị PASS/DROP."""

    kept_ids = {candidate.article.id for candidate in kept}
    results: list[dict[str, Any]] = []
    for index, (candidate, label, error) in enumerate(judged[:limit], start=1):
        article = candidate.article
        score = candidate.score if candidate.score is not None else 0.0
        results.append(
            {
                "rank": index,
                "score": round(float(score), 6),
                "source": candidate.source,
                "law_id": article.law_id,
                "law_name": article.law_name,
                "article": article.article,
                "article_title": article.article_title or "",
                "llm_label": label,
                "passed_threshold": article.id in kept_ids,
                "error": error,
            }
        )
    return results


def _retrieval_tool_args(query: RetrievalQuery, progress_metadata: dict[str, Any]) -> dict[str, Any]:
    """Args debug gọn; không trả full danh sách retrieval spaces."""

    payload = query.model_dump(mode="json")
    payload["search_spaces"] = {
        "scope": "global",
        "search_space_count": len(query.search_spaces),
    }
    payload["top_k"] = progress_metadata["top_k"]
    payload["scope"] = progress_metadata["scope"]
    return payload


def _set_retrieval_text(runtime: Any, state: LegalAssistantState, retrieval_text: str, mode: str, provider: str):
    """Cập nhật query retrieval khi không cần gọi LLM rewrite/HyDE."""

    question = state["question"]
    state["retrieval_mode"] = mode
    state["retrieval_question"] = retrieval_text
    state["query_variants"] = _build_query_variants(runtime, question, retrieval_text, None)
    state.setdefault("tool_calls", []).append(
        {"name": "prepare_retrieval_query", "provider": provider, "args": {"mode": mode}, "result": retrieval_text}
    )
    return state


def _prompt_messages(system_prompt: str, human_prompt: str) -> list[Any] | str:
    """Tạo messages đúng role; fallback về plain text nếu thiếu LangChain."""

    if SystemMessage is None or HumanMessage is None:
        return f"system: {system_prompt}\n\nhuman: {human_prompt}"
    return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]


async def _invoke_prompt_messages(runtime: Any, system_prompt: str, human_prompt: str) -> str:
    """Gọi LLM cho các node nội bộ bằng SystemMessage/HumanMessage."""

    messages = _prompt_messages(system_prompt, human_prompt)
    if isinstance(messages, str):
        return await runtime.llm.ainvoke(messages)
    if hasattr(runtime.llm, "ainvoke_messages"):
        return await runtime.llm.ainvoke_messages(messages)
    return await runtime.llm.ainvoke(_messages_to_prompt(messages))


def _clean_retrieval_text(text: str, mode: str) -> str:
    return text.splitlines()[0].strip(" -\t") if mode == "rewrite" else text.strip()


def _retrieval_mode(rewrite_enabled: bool, hyde_enabled: bool) -> str:
    if rewrite_enabled and hyde_enabled:
        return "rewrite+hyde"
    if rewrite_enabled:
        return "rewrite"
    if hyde_enabled:
        return "hyde"
    return "none"


def _build_query_variants(
    runtime: Any,
    question: str,
    rewritten_question: str | None,
    hypothetical_answer: str | None,
) -> list[str]:
    variants: list[str] = []
    for item in [hypothetical_answer, rewritten_question, question]:
        if item and item not in variants:
            variants.append(item)
    return variants[: runtime.settings.legal_assistant.rewrite.max_variants]


async def _chat_answer(runtime: Any, state: LegalAssistantState, articles: list[LegalArticle] | None) -> str:
    messages = _build_llm_messages(state, articles)
    if runtime.llm is None:
        return _fallback_answer(state["question"], articles)
    try:
        if runtime.settings.legal_assistant.chat.token_streaming and has_progress_callback():
            streamed = await _stream_chat_answer(runtime, messages)
            if streamed:
                return streamed
        if isinstance(messages, str):
            return await runtime.llm.ainvoke(messages)
        if hasattr(runtime.llm, "ainvoke_messages"):
            return await runtime.llm.ainvoke_messages(messages)
        return await runtime.llm.ainvoke(_messages_to_prompt(messages))
    except Exception as exc:
        state.setdefault("debug", {})["llm_error"] = str(exc)
        return _fallback_answer(state["question"], articles)


async def _stream_chat_answer(runtime: Any, messages: list[Any] | str) -> str:
    """Stream token trong node LangGraph answer và gom lại full answer."""

    chunks: list[str] = []
    if isinstance(messages, str):
        if not hasattr(runtime.llm, "astream"):
            return ""
        async for token in runtime.llm.astream(messages):
            chunks.append(token)
            await emit_token(token)
    else:
        if hasattr(runtime.llm, "astream_messages"):
            async for token in runtime.llm.astream_messages(messages):
                chunks.append(token)
                await emit_token(token)
        elif hasattr(runtime.llm, "astream"):
            async for token in runtime.llm.astream(_messages_to_prompt(messages)):
                chunks.append(token)
                await emit_token(token)
    return "".join(chunks).strip()


def _build_llm_messages(state: LegalAssistantState, articles: list[LegalArticle] | None) -> list[Any] | str:
    history_messages = list(state.get("messages", []))
    if SystemMessage is None or HumanMessage is None:
        chunks = [SYSTEM_PROMPT]
        chunks.extend(getattr(message, "content", str(message)) for message in history_messages)
        if articles is not None:
            chunks.append(build_legal_context_message(articles))
        return "\n\n".join(str(chunk) for chunk in chunks if chunk)
    messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT), *history_messages]
    if articles is not None:
        messages.append(HumanMessage(content=build_legal_context_message(articles)))
    return messages


def _messages_to_prompt(messages: list[Any]) -> str:
    return "\n\n".join(
        f"{getattr(message, 'type', message.__class__.__name__)}: {getattr(message, 'content', str(message))}"
        for message in messages
    )


def _fallback_answer(question: str, articles: list[LegalArticle] | None) -> str:
    if articles is None:
        return "Mình là MscAI. Bạn muốn hỏi vấn đề pháp lý nào?"
    if not articles:
        return "Mình chưa tìm thấy căn cứ trong kho dữ liệu đã đăng ký, nên chưa thể kết luận nội dung pháp lý cụ thể."
    lead = f"Dựa trên các căn cứ đã truy hồi cho câu hỏi: {question}"
    bullets = [f"- {article.article} của {article.law_name}: {' '.join(article.content.split())[:450]}" for article in articles[:5]]
    return "\n".join([lead, *bullets])


def _append_ai_message(state: LegalAssistantState) -> None:
    answer = state.get("answer")
    if not answer or AIMessage is None:
        return
    messages = state.setdefault("messages", [])
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].content == answer:
        return
    messages.append(AIMessage(content=answer))


def _append_once(value: str | None, values: list[str]) -> None:
    if value and value not in values:
        values.append(value)


def normalize_related_ref(reference: str) -> tuple[str | None, str | None]:
    parts = [part.strip() for part in reference.split("|") if part.strip()]
    if len(parts) == 4:
        _, law_id, law_name, article = parts
        return f"{law_id}|{law_name}", f"{law_id}|{law_name}|{article}"
    if len(parts) == 3:
        law_id, law_name, article = parts
        return f"{law_id}|{law_name}", f"{law_id}|{law_name}|{article}"
    return None, None
