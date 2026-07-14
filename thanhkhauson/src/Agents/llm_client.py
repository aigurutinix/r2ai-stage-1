from typing import Optional

from langchain_openai import ChatOpenAI
from Agents.config import LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, LLM_MAX_TOKENS, LLM_REASONING_FORMAT

def get_llm(
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 20,
    enable_thinking: bool = True,
    thinking_token_budget: Optional[int] = None,
) -> ChatOpenAI:
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}
    }
    if top_k > 0:
        extra_body["top_k"] = top_k
    if thinking_token_budget is not None:
        extra_body["thinking_budget_tokens"] = thinking_token_budget
    if LLM_REASONING_FORMAT:
        extra_body["reasoning_format"] = LLM_REASONING_FORMAT

    return ChatOpenAI(
        model=LLM_MODEL, 
        base_url=LLM_BASE_URL, 
        api_key=LLM_API_KEY, 
        temperature=temperature,
        top_p=top_p,
        extra_body=extra_body,
        max_tokens=LLM_MAX_TOKENS,
        max_retries=3
    )
