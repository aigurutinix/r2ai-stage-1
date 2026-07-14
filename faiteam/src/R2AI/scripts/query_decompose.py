"""Sub-query decomposition for long legal questions."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

DEFAULT_THRESHOLD_1 = 30
DEFAULT_THRESHOLD_2 = 58

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_QUESTION_END = re.compile(
    r"(?:như thế nào|ra sao|bao nhiêu|bao lâu|gì|nào|thế nào)\??$",
    re.IGNORECASE,
)
_ASPECT = re.compile(
    r"điều kiện|thủ tục|mức phạt|hình thức|thời hạn|trách nhiệm|"
    r"xử lý|khắc phục|nghĩa vụ|quyền|chi phí|hồ sơ|thời gian|thủ tục",
    re.IGNORECASE,
)
# "và" inside fixed phrases — not a split point
_SKIP_AND = re.compile(
    r"nhỏ và vừa|trong và ngoài|thương binh và xã hội|"
    r"khoa học và công nghệ|đầu tư và phát triển|"
    r"lao động và bảo hiểm xã hội|lao động, thương binh",
    re.IGNORECASE,
)
_BAD_OUTPUT = re.compile(
    r"không có|tôi không thể|sub-query\s*\d|json array|tuy nhiên|"
    r"kinh nghiệm chung|không thể cung cấp",
    re.IGNORECASE,
)


def load_embed_tokenizer(embed_model_path: Path):
    return AutoTokenizer.from_pretrained(str(embed_model_path))


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def num_subqueries(
    token_count: int,
    *,
    threshold_1: int = DEFAULT_THRESHOLD_1,
    threshold_2: int = DEFAULT_THRESHOLD_2,
) -> int:
    if token_count < threshold_1:
        return 1
    if token_count < threshold_2:
        return 2
    return 3


def _ensure_question(text: str) -> str:
    text = text.strip().rstrip("?")
    return f"{text}?"


def _question_context(question: str) -> str:
    """Leading situational clause shared by split sub-queries."""
    for pat in (
        r"^(Nếu .+? thì)",
        r"^(Khi .+? thì)",
        r"^(Khi .+?,)",
        r"^(Trong trường hợp .+?,)",
        r"^(Công ty .+? thì)",
        r"^(Doanh nghiệp .+? thì)",
    ):
        m = re.match(pat, question, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(",")
    return ""


def _with_context(clause: str, question: str) -> str:
    clause = clause.strip()
    if re.match(r"^(?:Nếu|Khi|Trong|Công ty|Doanh nghiệp|Hộ)", clause, re.IGNORECASE):
        return _ensure_question(clause)
    ctx = _question_context(question)
    if ctx:
        return _ensure_question(f"{ctx} {clause.lstrip('phải ')}")
    return _ensure_question(clause)


def _is_valid_subquery(sub: str, original: str) -> bool:
    sub = sub.strip()
    if len(sub) < 15 or len(sub) > len(original) * 1.2:
        return False
    if _BAD_OUTPUT.search(sub):
        return False
    if not _QUESTION_END.search(sub.rstrip("?")):
        return False
    # Must share some content words with original (avoid hallucinated subs)
    orig_words = set(re.findall(r"\w{4,}", original.lower()))
    sub_words = set(re.findall(r"\w{4,}", sub.lower()))
    return len(orig_words & sub_words) >= 2


def _validate_subqueries(subs: list[str], original: str, n: int) -> list[str] | None:
    cleaned = [s.strip() for s in subs if s.strip()]
    if len(cleaned) < 2 or len(cleaned) > n:
        return None
    if not all(_is_valid_subquery(s, original) for s in cleaned):
        return None
    if len({s.lower() for s in cleaned}) != len(cleaned):
        return None
    return cleaned[:n]


def decompose_by_rules(question: str, *, max_subqueries: int = 3) -> list[str] | None:
    """Conservative rule-based split. Returns None if no confident split."""
    q = question.strip()
    if max_subqueries <= 1:
        return None

    # Pattern A: "... như thế nào và (phải) ... ra sao?"
    m = re.search(
        r"^(.+?\s+như thế nào)\s+và\s+(?:phải\s+)?(.+ra sao)\??$",
        q,
        re.IGNORECASE,
    )
    if m:
        subs = [_ensure_question(m.group(1)), _with_context(m.group(2), q)]
        return _validate_subqueries(subs, q, max_subqueries)

    # Pattern B: two distinct legal aspects joined by " và "
    if not _SKIP_AND.search(q):
        parts = re.split(r"\s+và\s+", q, maxsplit=1)
        if len(parts) == 2:
            left, right = parts[0].strip(), parts[1].strip()
            left_a = _ASPECT.search(left)
            right_a = _ASPECT.search(right)
            if (
                left_a
                and right_a
                and left_a.group().lower() != right_a.group().lower()
                and _QUESTION_END.search(right.rstrip("?"))
            ):
                subs = [_ensure_question(left), _with_context(right, q)]
                validated = _validate_subqueries(subs, q, max_subqueries)
                if validated:
                    return validated

    # Pattern C: comma + second clause with its own question focus
    m = re.search(
        r"^(.+?),\s*((?:nếu|và nếu|công ty|doanh nghiệp).+(?:nào|gì|bao nhiêu|bao lâu|"
        r"như thế nào|ra sao).+\?)$",
        q,
        re.IGNORECASE,
    )
    if m:
        subs = [_ensure_question(m.group(1)), _ensure_question(m.group(2))]
        validated = _validate_subqueries(subs, q, max_subqueries)
        if validated:
            return validated

    # Pattern D: "vừa ... , vừa ... ?"
    m = re.search(r"^(.+?vừa .+?),\s*vừa (.+\?)$", q, re.IGNORECASE)
    if m:
        ctx = _question_context(q)
        subs = [
            _ensure_question(m.group(1)),
            _ensure_question(f"{ctx} vừa {m.group(2).rstrip('?')}" if ctx else m.group(2)),
        ]
        validated = _validate_subqueries(subs, q, max_subqueries)
        if validated:
            return validated

    return None


def plan_queries(
    question: str,
    tokenizer=None,
    *,
    threshold_1: int = DEFAULT_THRESHOLD_1,
    threshold_2: int = DEFAULT_THRESHOLD_2,
    use_token_budget: bool = False,
) -> tuple[int, list[str] | None]:
    """Return (n_subqueries, queries or None for LLM fallback).

    Default: chỉ tách khi rule-based khớp pattern rõ ràng; còn lại giữ nguyên câu hỏi.
    use_token_budget=True: logic cũ (ép tách theo độ dài token, cần LLM fallback).
    """
    if use_token_budget and tokenizer is not None:
        n = num_subqueries(
            count_tokens(question, tokenizer),
            threshold_1=threshold_1,
            threshold_2=threshold_2,
        )
        if n == 1:
            return 1, [question]
        ruled = decompose_by_rules(question, max_subqueries=n)
        if ruled:
            return len(ruled), ruled
        return n, None

    for max_n in (3, 2):
        ruled = decompose_by_rules(question, max_subqueries=max_n)
        if ruled:
            return len(ruled), ruled
    return 1, [question]


def _build_decompose_prompt(tokenizer, question: str, n: int) -> str:
    system = (
        "Bạn là chuyên gia pháp luật Việt Nam. "
        f"Tách câu hỏi pháp luật thành đúng {n} sub-query độc lập, "
        "mỗi sub-query tập trung một khía cạnh (điều kiện, mức phạt, thủ tục, thời hạn, v.v.). "
        "Chỉ trả về JSON array các chuỗi, không giải thích."
    )
    user = (
        f"Câu hỏi: {question}\n\n"
        f'Trả về đúng {n} sub-query dạng JSON array, ví dụ: ["sub-query 1", "sub-query 2"]'
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _parse_subqueries(text: str, n: int, fallback: str) -> list[str]:
    text = text.strip()
    match = _JSON_ARRAY_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                subs = [str(s).strip() for s in parsed if str(s).strip()]
                if subs:
                    validated = _validate_subqueries(subs, fallback, n)
                    if validated:
                        return validated
        except json.JSONDecodeError:
            pass
    return [fallback]


def decompose_question(
    question: str,
    n: int,
    model,
    tokenizer,
    eos_ids: list[int],
    *,
    max_new_tokens: int = 256,
) -> list[str]:
    if n <= 1:
        return [question]

    ruled = decompose_by_rules(question, max_subqueries=n)
    if ruled:
        return ruled

    prompt = _build_decompose_prompt(tokenizer, question, n)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    raw = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
    return _parse_subqueries(raw, n, question)


def resolve_queries(
    question: str,
    tokenizer=None,
    *,
    threshold_1: int = DEFAULT_THRESHOLD_1,
    threshold_2: int = DEFAULT_THRESHOLD_2,
    use_token_budget: bool = False,
) -> list[str]:
    """Best-effort sub-queries without LLM (preferred path)."""
    n, preset = plan_queries(
        question,
        tokenizer,
        threshold_1=threshold_1,
        threshold_2=threshold_2,
        use_token_budget=use_token_budget,
    )
    if preset is not None:
        return preset
    ruled = decompose_by_rules(question, max_subqueries=n)
    return ruled if ruled else [question]


def batch_decompose_questions(
    questions: list[dict],
    model,
    tokenizer,
    eos_ids: list[int],
    embed_tokenizer,
    *,
    threshold_1: int = DEFAULT_THRESHOLD_1,
    threshold_2: int = DEFAULT_THRESHOLD_2,
    use_token_budget: bool = False,
    use_llm: bool = True,
) -> dict[int, list[str]]:
    """Return question id -> list of query strings."""
    result: dict[int, list[str]] = {}
    pending: list[tuple[dict, int]] = []

    for q in questions:
        qid = q["id"]
        n, preset = plan_queries(
            q["question"],
            embed_tokenizer,
            threshold_1=threshold_1,
            threshold_2=threshold_2,
            use_token_budget=use_token_budget,
        )
        if preset is not None:
            result[qid] = preset
        else:
            pending.append((q, n))

    if not use_llm:
        for q, _n in pending:
            result[q["id"]] = resolve_queries(
                q["question"],
                embed_tokenizer,
                threshold_1=threshold_1,
                threshold_2=threshold_2,
                use_token_budget=use_token_budget,
            )
        return result

    for q, n in pending:
        subs = decompose_question(
            q["question"],
            n,
            model,
            tokenizer,
            eos_ids,
        )
        result[q["id"]] = subs
        print(f"  Decompose id={q['id']} → {len(subs)} sub-query")

    return result
