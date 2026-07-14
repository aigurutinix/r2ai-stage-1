from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "qwen/qwen3.5-9b")
NVIDIA_LLM_BASE_URL = os.getenv("NVIDIA_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")


def get_nvidia_llm(
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    enable_thinking: bool = False,
) -> ChatOpenAI:
    if not NVIDIA_API_KEY:
        raise RuntimeError("Missing NVIDIA_API_KEY. Set it in .env or export it before running.")

    return ChatOpenAI(
        model=NVIDIA_LLM_MODEL,
        base_url=NVIDIA_LLM_BASE_URL,
        api_key=NVIDIA_API_KEY,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_retries=3,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            }
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick test for NVIDIA Qwen NIM chat endpoint.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Xin chao, hay tra loi ngan gon: ban dang chay model nao?",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    llm = get_nvidia_llm(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    response = llm.invoke(args.prompt)
    print(response.content)


if __name__ == "__main__":
    main()
