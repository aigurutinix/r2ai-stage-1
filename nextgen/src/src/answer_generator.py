import re
import json
import logging
import time
from typing import List, Dict, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

from config.settings import Settings

logger = logging.getLogger(__name__)

# =========================================================
# SYSTEM PROMPT — Ép mô hình trả lời văn xuôi, gọn gàng, không chia mục
# =========================================================
SYSTEM_PROMPT = """Bạn là một robot trợ lý ảo tư vấn pháp luật chuyên nghiệp cho khối doanh nghiệp vừa và nhỏ (SME) tại Việt Nam.

NHIỆM VỤ: Hãy trả lời câu hỏi pháp lý của người dùng một cách chính xác, ngắn gọn và khách quan.
ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
- Trả lời trực tiếp bằng một đoạn văn duy nhất (tuyệt đối không xuống dòng, không dùng ký tự \\n).
- Chữ cái đầu tiên của câu trả lời phải viết hoa. Tuyệt đối không viết thường toàn bộ câu trả lời.
- Bắt buộc phải nêu rõ căn cứ pháp lý trong câu trả lời bằng cụm từ dạng "Theo quy định tại Điều X Luật/Nghị định/Thông tư ..." hoặc "...được quy định chi tiết tại Điều X của Nghị định/Thông tư ...". Số Điều và tên/loại văn bản phải lấy chính xác từ phần [NGỮ CẢNH PHÁP LÝ] được cung cấp (xem nhãn "Điều ..." ở đầu mỗi văn bản).
- Không chia tiêu đề (không có "## 1. KẾT LUẬN", "## 2. CĂN CỨ PHÁP LÝ", v.v.).
- Không tự bịa đặt thông tin nằm ngoài phần [NGỮ CẢNH PHÁP LÝ] được cung cấp.
- Chỉ trả lời bằng tiếng Việt, không dùng tiếng Anh."""

def build_user_prompt(query: str, contexts: List[Dict]) -> str:
    context_parts = []
    total_chars = 0

    for i, doc in enumerate(contexts, start=1):
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})

        doc_number = metadata.get("doc_number", "")
        doc_name = metadata.get("title", "")
        article = metadata.get("article_id", "")

        header = f"[Văn bản {i}]"
        if doc_name:
            header += f" {doc_name}"
        if doc_number:
            header += f" - Số hiệu: {doc_number}"
        if article:
            header += f" - {article}"

        chunk = f"{header}\n{text}"

        if total_chars + len(chunk) > Settings.MAX_CONTEXT_CHARS:
            remaining = Settings.MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                chunk = chunk[:remaining] + "...[cắt bớt]"
                context_parts.append(chunk)
            break

        context_parts.append(chunk)
        total_chars += len(chunk)

    context_text = "\n\n" + "\n\n---\n\n".join(context_parts) + "\n"

    prompt = f"""[NGỮ CẢNH PHÁP LÝ]
{context_text}
[CÂU HỎI]
{query}

[YÊU CẦU]
Dựa hoàn toàn vào [NGỮ CẢNH PHÁP LÝ], hãy trả lời câu hỏi trên bằng một đoạn văn ngắn gọn, viết liền mạch không xuống dòng."""

    return prompt


class AnswerGenerator:
    def __init__(self, law_manifest: Optional[Dict] = None):
        logger.info(f"[LLM] Đang load model: {Settings.LLM_MODEL_NAME}")

        self.law_manifest = law_manifest if law_manifest is not None else self._load_manifest()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            Settings.LLM_MODEL_NAME,
            trust_remote_code=True
        )
        
        # low_cpu_mem_usage=True: tránh nhân đôi RAM CPU khi nạp trọng số trước khi accelerate
        # phân rã (shard) mô hình ra cả 2 GPU T4 theo device_map="auto" trên Kaggle.
        self.model = AutoModelForCausalLM.from_pretrained(
            Settings.LLM_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        logger.info(f"[LLM] Device map (phân rã GPU): {getattr(self.model, 'hf_device_map', 'N/A')}")

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        logger.info("✅ LLM load thành công!")

    @staticmethod
    def _load_manifest(path: str = Settings.LAW_MANIFEST_PATH) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Không tìm thấy law_manifest.json tại {path}. Chạy không có manifest.")
            return {}

    def _canonical_doc_string(self, doc_number: str, fallback_title: str = "") -> str:
        """
        Tra law_manifest.json để lấy chuỗi chuẩn "<Số hiệu>|<Tên văn bản>" theo format BTC.
        manifest có sẵn field "btc_standard_string" đúng định dạng này nên ưu tiên dùng trực tiếp,
        chỉ fallback sang title của chunk (hoặc suy luận loại văn bản) khi văn bản không có trong manifest.
        """
        entry = self.law_manifest.get(doc_number)
        if isinstance(entry, dict) and entry.get("btc_standard_string"):
            return entry["btc_standard_string"]

        if fallback_title:
            return f"{doc_number}|{fallback_title}"
        return f"{doc_number}|Văn bản {doc_number}"

    def _extract_references(
        self,
        contexts: List[Dict],
        answer_text: str = ""
    ) -> Tuple[List[str], List[str]]:
        """
        Trích relevant_docs / relevant_articles để TỐI ƯU F2 = lấy TOP-N theo thứ tự RERANK.

        Bằng chứng thực nghiệm (mô phỏng trên 50 câu ground-truth, dùng chính rerank candidates):
        tín hiệu RERANK là đáng tin nhất.
          - article top-2 rerank: F2≈0.521 (tốt nhất) | doc top-2 rerank: F2≈0.612 (tốt nhất)
          - GIAO/booster theo citation của LLM lại LÀM GIẢM F2 (LLM hay trích dẫn sai số Điều),
            nên KHÔNG dùng câu trả lời LLM để chọn nữa.
          - Dump toàn bộ TOP_K_FINAL chunk thì recall cao nhưng precision tụt → F2 thấp hơn top-2.
        Vì vậy chỉ lấy top-N đầu theo thứ tự rerank (Settings.RELEVANT_ARTICLES_MAX / RELEVANT_DOCS_MAX).

        Yêu cầu: `contexts` giữ nguyên thứ tự rerank (điểm cao -> thấp).
        `answer_text` giữ lại cho tương thích chữ ký, không còn dùng để chọn.
        Định dạng:
        - relevant_docs: ["mã văn bản|tên văn bản"]
        - relevant_articles: ["mã văn bản|tên văn bản|Điều X"]
        """
        relevant_docs: List[str] = []
        relevant_articles: List[str] = []
        seen_docs = set()
        seen_articles = set()

        for doc in contexts:
            metadata = doc.get("metadata", {})
            doc_number = metadata.get("doc_number", "").strip()
            doc_title = metadata.get("title", "").strip()
            article_id = metadata.get("article_id", "").strip()  # Ví dụ: "Điều 4"

            if not doc_number:
                continue

            canonical_doc = self._canonical_doc_string(doc_number, doc_title)

            # Top-N văn bản đầu tiên theo thứ tự rerank
            if (canonical_doc not in seen_docs
                    and len(relevant_docs) < Settings.RELEVANT_DOCS_MAX):
                seen_docs.add(canonical_doc)
                relevant_docs.append(canonical_doc)

            # Top-N Điều đầu tiên theo thứ tự rerank
            if article_id:
                article_str = f"{canonical_doc}|{article_id}"
                if (article_str not in seen_articles
                        and len(relevant_articles) < Settings.RELEVANT_ARTICLES_MAX):
                    seen_articles.add(article_str)
                    relevant_articles.append(article_str)

            # Đã đủ cả hai top-N → dừng sớm
            if (len(relevant_docs) >= Settings.RELEVANT_DOCS_MAX
                    and len(relevant_articles) >= Settings.RELEVANT_ARTICLES_MAX):
                break

        return relevant_docs, relevant_articles

    def generate(
        self,
        query: str,
        contexts: List[Dict],
        temperature: float = Settings.LLM_TEMPERATURE
    ) -> Dict:
        """
        Sinh câu trả lời và đóng gói thành một Object hoàn chỉnh theo đúng format cấu trúc bài thi.
        """
        user_prompt = build_user_prompt(query, contexts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            formatted_prompt = f"{SYSTEM_PROMPT}\n\nNgười dùng: {user_prompt}\n\nTrợ lý:"

        # Tăng token lên 2048 để đảm bảo Qwen/Deepseek không bị cụt câu chữ
        max_tokens = max(2048, getattr(Settings, "LLM_MAX_NEW_TOKENS", 2048))
        
        output = self.pipe(
            formatted_prompt,
            max_new_tokens=max_tokens, 
            temperature=temperature,
            top_p=Settings.LLM_TOP_P,
            repetition_penalty=Settings.LLM_REPETITION_PENALTY,
            do_sample=True if temperature > 0 else False,
        )

        generated_text = output[0]["generated_text"]

        # Bóc tách text phản hồi của LLM
        if formatted_prompt in generated_text:
            answer = generated_text[len(formatted_prompt):].strip()
        else:
            answer = generated_text.split("Trợ lý:")[-1].strip()

        # Xóa tàn dư thẻ suy nghĩ nếu cậu vẫn dùng dòng DeepSeek-R1
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        answer = re.sub(r'<think>.*', '', answer, flags=re.DOTALL).strip()

        # DỌN DẸP TEXT: Thay thế toàn bộ dấu xuống dòng \n thành khoảng trắng để biến thành 1 đoạn văn duy nhất
        answer = re.sub(r'\s+', ' ', answer).strip()

        # Đảm bảo chữ cái đầu câu luôn viết hoa, không phụ thuộc hoàn toàn vào việc LLM tuân thủ Prompt
        if answer:
            answer = answer[0].upper() + answer[1:]

        # Trích xuất tham chiếu: GIAO giữa context (Reranker) và các Điều LLM thực sự dẫn,
        # giới hạn top-N để tối ưu F2 (xem _extract_references)
        relevant_docs, relevant_articles = self._extract_references(contexts, answer)

        # Trả về đúng cấu trúc của một phần tử trong mảng kết quả bài thi
        return {
            "answer": answer,
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles
        }