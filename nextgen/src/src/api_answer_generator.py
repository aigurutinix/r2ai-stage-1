"""
src/api_answer_generator.py  (LUỒNG RIÊNG cho app demo — KHÔNG dùng trong main.py / fast_retrieval.py)

Sinh câu trả lời qua API hosted OpenAI-compatible (Together / DeepInfra / Novita...) thay vì nạp
Qwen-7B tại chỗ — vì máy demo/HF Spaces chỉ có CPU. Có streaming.

Khác bản thi: app demo cho phép câu trả lời TRÌNH BÀY NHIỀU ĐOẠN / markdown nhẹ (xuống dòng, gạch
đầu dòng, bước tính toán) cho dễ đọc — nên KHÔNG gộp về một đoạn như src/answer_generator.py.
Vẫn giữ yêu cầu: bám ngữ cảnh, nêu căn cứ "Theo quy định tại Điều...", không bịa.
"""
import re
import logging
from typing import Iterator, List, Dict

from config.settings import Settings

logger = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_THINK_OPEN = re.compile(r"<think>.*", flags=re.DOTALL)

# Prompt riêng cho demo: cho phép trình bày có cấu trúc (markdown nhẹ), nhiều đoạn.
DEMO_SYSTEM_PROMPT = """Bạn là Trợ lý NextGen — trợ lý AI tư vấn pháp luật Việt Nam cho doanh nghiệp nhỏ và vừa (SME).

NHIỆM VỤ: Trả lời câu hỏi pháp lý của người dùng một cách chính xác, rõ ràng, khách quan.
YÊU CẦU:
- Chỉ dựa vào phần [NGỮ CẢNH PHÁP LÝ] được cung cấp; TUYỆT ĐỐI không bịa thông tin ngoài ngữ cảnh.
- Nêu rõ căn cứ pháp lý bằng cụm "Theo quy định tại Điều X ..." kèm số hiệu/tên văn bản lấy CHÍNH XÁC từ ngữ cảnh.
- Được phép trình bày nhiều đoạn, xuống dòng, dùng gạch đầu dòng và các bước tính toán khi cần cho dễ đọc.
- Không dùng tiêu đề markdown lớn (không "# ", "## "); chỉ dùng đoạn văn, gạch đầu dòng "- " và **in đậm** khi cần nhấn mạnh.
- Chỉ trả lời bằng tiếng Việt."""


def build_demo_user_prompt(query: str, contexts: List[Dict]) -> str:
    """Đóng gói ngữ cảnh giống bản thi nhưng yêu cầu trình bày rõ ràng (cho phép xuống dòng)."""
    parts = []
    total = 0
    for i, doc in enumerate(contexts, start=1):
        meta = doc.get("metadata", {})
        header = f"[Văn bản {i}]"
        if meta.get("title"):
            header += f" {meta.get('title')}"
        if meta.get("doc_number"):
            header += f" - Số hiệu: {meta.get('doc_number')}"
        if meta.get("article_id"):
            header += f" - {meta.get('article_id')}"
        chunk = f"{header}\n{doc.get('text', '')}"
        if total + len(chunk) > Settings.MAX_CONTEXT_CHARS:
            remain = Settings.MAX_CONTEXT_CHARS - total
            if remain > 200:
                parts.append(chunk[:remain] + "...[cắt bớt]")
            break
        parts.append(chunk)
        total += len(chunk)
    context_text = "\n\n---\n\n".join(parts)
    return (
        f"[NGỮ CẢNH PHÁP LÝ]\n\n{context_text}\n\n"
        f"[CÂU HỎI]\n{query}\n\n"
        f"[YÊU CẦU]\nDựa hoàn toàn vào [NGỮ CẢNH PHÁP LÝ], hãy trả lời câu hỏi trên rõ ràng, "
        f"có căn cứ pháp lý cụ thể."
    )


def clean_answer(text: str) -> str:
    """Hậu xử lý cho demo: bỏ thẻ <think>, GIỮ xuống dòng, chỉ gọn khoảng trắng thừa và dòng trống liên tiếp."""
    if not text:
        return ""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_OPEN.sub("", text)
    # Gọn khoảng trắng cuối mỗi dòng, gộp >2 dòng trống thành 1 dòng trống.
    lines = [ln.rstrip() for ln in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


class ApiAnswerGenerator:
    """Bộ sinh câu trả lời qua API. Không load model nặng → khởi tạo tức thì."""

    def __init__(self):
        from openai import OpenAI  # import trễ để không bắt buộc khi chạy đường thi

        if not Settings.LLM_API_KEY:
            logger.warning("[ApiLLM] LLM_API_KEY rỗng — hãy đặt trong .env/Secrets, nếu không sẽ lỗi khi sinh câu trả lời.")
        self.client = OpenAI(
            base_url=Settings.LLM_API_BASE_URL,
            api_key=Settings.LLM_API_KEY or "sk-missing",
        )
        self.model = Settings.LLM_API_MODEL
        logger.info(f"[ApiLLM] Sẵn sàng: model={self.model} @ {Settings.LLM_API_BASE_URL}")

    def _messages(self, query: str, contexts: List[Dict]) -> List[Dict]:
        return [
            {"role": "system", "content": DEMO_SYSTEM_PROMPT},
            {"role": "user", "content": build_demo_user_prompt(query, contexts)},
        ]

    def stream_raw(
        self,
        query: str,
        contexts: List[Dict],
        temperature: float = Settings.LLM_TEMPERATURE,
    ) -> Iterator[str]:
        """Stream các mảnh (delta) text thô từ API theo thời gian thực (UI hiển thị dần)."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(query, contexts),
            temperature=temperature,
            top_p=Settings.LLM_TOP_P,
            max_tokens=max(1024, getattr(Settings, "LLM_MAX_NEW_TOKENS", 1024)),
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = getattr(chunk.choices[0].delta, "content", None)
            if piece:
                yield piece

    def generate(
        self,
        query: str,
        contexts: List[Dict],
        temperature: float = Settings.LLM_TEMPERATURE,
    ) -> str:
        """Sinh trọn câu trả lời (không stream) + hậu xử lý."""
        return clean_answer("".join(self.stream_raw(query, contexts, temperature)))
