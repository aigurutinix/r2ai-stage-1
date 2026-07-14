"""
src/llm_selector.py  (LUỒNG RIÊNG cho fast_retrieval --llm-select)

LLM đóng vai BỘ CHỌN (selector), KHÔNG sinh đáp án:
  đưa cho LLM câu hỏi + danh sách ứng viên (đã rerank, đánh số, kèm số hiệu + Điều),
  bắt LLM liệt kê SỐ THỨ TỰ của những điều TRỰC TIẾP làm căn cứ. Số lượng BIẾN THIÊN
  (LLM tự quyết 1..max_select) để khắc phục việc cố định 2 điều (thừa ở câu 1-điều,
  thiếu ở câu nhiều-điều). Output cực ngắn -> nhanh hơn nhiều so với sinh answer.

Chọn theo SỐ THỨ TỰ trong danh sách ứng viên (grounded) -> không bịa số hiệu/điều,
định dạng luôn khớp corpus. Nhược: recall vẫn bị chặn bởi retrieval (chỉ chọn được điều
đã truy hồi). LƯU Ý: phải đo lại trên ground_truth_50 — chưa chắc thắng top-2.
"""
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

SELECT_SYSTEM = (
    "Bạn là chuyên gia pháp luật Việt Nam. Từ danh sách điều luật ứng viên, hãy LIỆT KÊ "
    "những điều TRỰC TIẾP làm căn cứ trả lời câu hỏi. Chọn ĐỦ các điều liên quan (có thể 1 "
    "hoặc nhiều), nhưng KHÔNG chọn điều không liên quan. Ưu tiên độ chính xác."
)


def select_candidates(
    question: str,
    ranked: List[Dict],
    pipe,
    tokenizer,
    pool_k: int = 8,
    max_select: int = 5,
) -> List[Dict]:
    """
    Trả về danh sách context (con) LLM chọn, theo thứ tự ưu tiên của LLM (1..max_select phần tử).
    Fallback: nếu LLM không trả số hợp lệ -> lấy top-1 theo rerank (giữ tối thiểu 1 để không rỗng).
    """
    pool = ranked[:pool_k]
    if not pool:
        return []

    lines = []
    for i, c in enumerate(pool, 1):
        md = c.get("metadata", {})
        article = md.get("article_id", "")
        doc_number = md.get("doc_number", "")
        title = (md.get("title", "") or "")[:55]
        snippet = re.sub(r"\s+", " ", (c.get("text", "") or "")).strip()[:200]
        lines.append(f"[{i}] {article} | Số hiệu: {doc_number} | {title}: {snippet}")

    user_prompt = (
        f"CÂU HỎI: {question}\n\n"
        f"DANH SÁCH ĐIỀU LUẬT ỨNG VIÊN:\n" + "\n".join(lines) +
        f"\n\nHãy LIỆT KÊ các điều luật TRỰC TIẾP làm căn cứ trả lời câu hỏi trên "
        f"(chọn TỐI ĐA {max_select} điều, quan trọng nhất trước; chỉ chọn điều thật sự liên quan). "
        f"CHỈ TRẢ LỜI bằng các SỐ THỨ TỰ trong [1-{len(pool)}], cách nhau bởi dấu phẩy "
        f"(ví dụ: 1,3,4). Tuyệt đối không giải thích, không viết chữ."
    )

    messages = [
        {"role": "system", "content": SELECT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"{SELECT_SYSTEM}\n\n{user_prompt}\n\nTrả lời:"

    try:
        out = pipe(prompt, max_new_tokens=40, do_sample=False)
        gen = out[0]["generated_text"]
        ans = gen[len(prompt):] if prompt in gen else gen.split("Trả lời:")[-1]
    except Exception as e:
        logger.warning(f"[llm_selector] LLM lỗi, fallback top-1 rerank: {e}")
        return pool[:1]

    # Bóc các số thứ tự LLM chọn: giữ thứ tự, loại trùng, trong khoảng hợp lệ
    chosen: List[Dict] = []
    seen = set()
    for n in re.findall(r"\d+", ans):
        idx = int(n)
        if 1 <= idx <= len(pool) and idx not in seen:
            seen.add(idx)
            chosen.append(pool[idx - 1])
        if len(chosen) >= max_select:
            break

    if not chosen:
        # LLM không trả số hợp lệ -> giữ top-1 theo rerank để không rỗng (không mất recall)
        chosen = pool[:1]

    return chosen
