"""LLM client (OpenAI-compatible API)."""
from __future__ import annotations

import logging
from typing import Iterator

from openai import OpenAI

from backend.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self) -> None:
        s = get_settings()
        if not s.llm_api_key:
            raise RuntimeError("LLM_API_KEY chưa được set trong .env")
        self.client = OpenAI(
            api_key=s.llm_api_key,
            base_url=s.llm_base_url or None,
            timeout=120.0,
        )
        self.model = s.llm_model
        self.max_tokens = s.llm_max_tokens
        self.temperature = s.llm_temperature
        self.reasoning_effort = s.llm_reasoning_effort

    def _base_kwargs(self, system: str, user: str) -> dict:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        # Tắt "thinking" cho model reasoning (Qwen3.5/Ollama) để token dồn vào
        # câu trả lời, tránh content rỗng. Chỉ gửi khi được cấu hình.
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs

    def complete(self, system: str, user: str, think: bool = False) -> str:
        """Gọi LLM. think=True bật reasoning (Qwen3: /think prefix, output stripping <think>)."""
        msg_user = ("/think\n" + user) if think else user
        kwargs = self._base_kwargs(system, msg_user)
        resp = self.client.chat.completions.create(**kwargs, stream=False)
        raw = resp.choices[0].message.content or ""
        return self._strip_think(raw) if think else raw

    @staticmethod
    def _strip_think(text: str) -> str:
        """Bỏ phần <think>...</think> của Qwen3 khỏi output."""
        import re
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    def stream(self, system: str, user: str) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            **self._base_kwargs(system, user),
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
