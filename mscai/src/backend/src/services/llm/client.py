"""Wrapper cho chat model OpenAI-compatible.

Local vLLM thường expose API tương thích OpenAI, nên dùng ``ChatOpenAI`` của
LangChain giúp agent gọi model local giống như gọi OpenAI API.
"""
from __future__ import annotations

from src.config import settings


class LLMClient:
    """Client mỏng bọc LangChain ``ChatOpenAI``."""

    def __init__(self):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - guard khi thiếu dependency
            raise RuntimeError("Install langchain-openai to use the LLM endpoint") from exc

        self.base_url = settings.llm.base_url
        self.model_name = settings.llm.model_name
        self.chat = ChatOpenAI(
            model=self.model_name,
            base_url=self.base_url,
            api_key=settings.llm.api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": settings.llm.enable_thinking}},
        )

    async def ainvoke(self, prompt: str) -> str:
        """Gọi chat model bất đồng bộ bằng plain prompt và trả về plain text."""

        try:
            response = await self.chat.ainvoke(prompt)
        except Exception as exc:
            raise RuntimeError(
                f"Không gọi được LLM endpoint {self.base_url}, model {self.model_name}: {exc}"
            ) from exc
        return response.content

    async def ainvoke_messages(self, messages) -> str:
        """Gọi chat model bằng danh sách LangChain messages."""

        try:
            response = await self.chat.ainvoke(messages)
        except Exception as exc:
            raise RuntimeError(
                f"Không gọi được LLM endpoint {self.base_url}, model {self.model_name}: {exc}"
            ) from exc
        return response.content

    async def astream(self, prompt: str):
        """Stream token từ chat model bằng plain prompt."""

        try:
            async for chunk in self.chat.astream(prompt):
                token = getattr(chunk, "content", "") or ""
                if token:
                    yield str(token)
        except Exception as exc:
            raise RuntimeError(
                f"Không stream được LLM endpoint {self.base_url}, model {self.model_name}: {exc}"
            ) from exc

    async def astream_messages(self, messages):
        """Stream token từ chat model bằng danh sách LangChain messages."""

        try:
            async for chunk in self.chat.astream(messages):
                token = getattr(chunk, "content", "") or ""
                if token:
                    yield str(token)
        except Exception as exc:
            raise RuntimeError(
                f"Không stream được LLM endpoint {self.base_url}, model {self.model_name}: {exc}"
            ) from exc
