#!/usr/bin/env python3
"""
Generate answers for R2AI Task 3.1 submission.
Uses retrieved articles + dynamic few-shot from data_final.

Usage:
    python3 gen_answer.py --port 8011 --model Qwen3-8B-AWQ --submission submission_3_1_llm8b_rerankerv2ck8k_top5.json --workers 4
"""

import json
import re
import pickle
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

BASE = Path(__file__).resolve().parent.parent.parent
API_KEY = "token-abc123"
FEWSHOT_MIN_SCORE = 0.6
FEWSHOT_TOP_K = 2


def load_content_cache():
    CHUNKS_DIR = BASE / "vbpl_dataset" / "chunks"
    DOC_TYPES = ["hien_phap", "bo_luat", "luat", "phap_lenh", "nghi_dinh",
                 "nghi_quyet", "nghi_quyet_lien_tich", "thong_tu", "thong_tu_lien_tich", "quyet_dinh"]
    cc = {}
    for dtype in DOC_TYPES:
        d = CHUNKS_DIR / dtype
        if not d.exists():
            continue
        for fpath in sorted(d.glob("*.json")):
            data = json.loads(fpath.read_text())
            for c in data.get("chunks", []):
                cid = c.get("chunk_id", "")
                content = "\n".join(c.get("content", [])).strip()
                if cid and content:
                    cc[cid] = content
    return cc


def load_metadata_cache():
    meta_cache = {}
    for mf in (BASE / "vbpl_dataset" / "metadata").glob("*.json"):
        docs = json.loads(mf.read_text())
        for doc in docs:
            item_id = doc.get("item_id", "")
            if item_id:
                meta_cache[item_id] = doc
    return meta_cache


def parse_art(art_str):
    parts = art_str.split("|")
    if len(parts) >= 3:
        m = re.search(r'\d+', parts[2])
        return (parts[0], m.group() if m else "")
    return None


def get_article_info(art_str, pool_cands, content_cache, meta_cache):
    k = parse_art(art_str)
    if not k:
        return None

    law_id, art_no = k
    parts = art_str.split("|")
    doc_title = parts[1] if len(parts) >= 2 else ""

    for c in pool_cands:
        ck = (c.get("law_id", ""), str(c.get("article_number", "")))
        if ck == k:
            cid = c.get("chunk_id", "")
            content = content_cache.get(cid, "")
            if not content and cid.startswith("qa_"):
                vbid = c.get("van_ban_id", "")
                ano = c.get("article_number", "")
                if vbid and ano:
                    content = content_cache.get(f"{vbid}#dieu_{ano}", "")

            van_ban_id = c.get("van_ban_id", "")
            item_id = van_ban_id.split("/")[-1] if "/" in van_ban_id else van_ban_id
            meta = meta_cache.get(item_id, {})

            hieu_luc = c.get("hieu_luc", "")
            status = "còn hiệu lực" if hieu_luc == "con_hieu_luc" else "hết hiệu lực" if hieu_luc == "het_hieu_luc" else hieu_luc

            return {
                "doc_title": c.get("doc_title", ""),
                "doc_description": c.get("doc_description", "") or meta.get("description", ""),
                "law_id": c.get("law_id", ""),
                "article_number": c.get("article_number", ""),
                "article_title": c.get("article_title", ""),
                "status": status,
                "ngay_ban_hanh": meta.get("ngay_ban_hanh", ""),
                "ngay_hieu_luc": meta.get("ngay_hieu_luc", ""),
                "content": content[:8000],
            }

    # Fallback: article not in pool, look up content_cache directly
    content = ""
    for cid, text in content_cache.items():
        if cid.endswith(f"#dieu_{art_no}") and law_id in cid:
            content = text
            break
    meta = meta_cache.get(law_id.split("/")[-1] if "/" in law_id else law_id, {})
    return {
        "doc_title": doc_title,
        "doc_description": meta.get("description", ""),
        "law_id": law_id,
        "article_number": art_no,
        "article_title": "",
        "status": "",
        "ngay_ban_hanh": meta.get("ngay_ban_hanh", ""),
        "ngay_hieu_luc": meta.get("ngay_hieu_luc", ""),
        "content": content[:8000],
    }


def build_fewshot_block(examples):
    if not examples:
        return ""

    lines = ["=== VÍ DỤ THAM KHẢO ==="]
    for ex in examples:
        lines.append(f"\nCâu hỏi: {ex['question']}")
        if ex.get("full_answer"):
            lines.append(f"Trả lời: {ex['full_answer'][:800]}")
        elif ex.get("answer"):
            lines.append(f"Trả lời: {ex['answer']}")
    lines.append("\n=== HẾT VÍ DỤ ===")
    return "\n".join(lines)


def build_articles_block(articles_info):
    blocks = []
    for i, a in enumerate(articles_info):
        header = f"[{i+1}] {a['doc_title']} ({a['law_id']})"
        if a["doc_description"]:
            header += f" - {a['doc_description'][:100]}"
        header += f"\n    Điều {a['article_number']}"
        if a["article_title"]:
            header += f". {a['article_title']}"
        header += f"\n    Trạng thái: {a['status']}"
        if a["ngay_ban_hanh"]:
            header += f" | Ban hành: {a['ngay_ban_hanh']}"
        if a["ngay_hieu_luc"]:
            header += f" | Hiệu lực từ: {a['ngay_hieu_luc']}"
        header += f"\n    Nội dung:\n{a['content']}"
        blocks.append(header)
    return "\n\n".join(blocks)


SYSTEM_PROMPT_BASE = """Bạn là chuyên gia tư vấn pháp luật Việt Nam. Nhiệm vụ: trả lời câu hỏi pháp luật dựa HOÀN TOÀN vào các điều luật được cung cấp bên dưới.

QUY TẮC BẮT BUỘC:
- CHỈ trích dẫn và sử dụng thông tin từ các điều luật được cung cấp. TUYỆT ĐỐI KHÔNG bổ sung điều luật hoặc văn bản pháp luật nào khác ngoài những điều đã cho.
- Mỗi luận điểm PHẢI kèm trích dẫn cụ thể theo mẫu: "Theo khoản X Điều Y [Tên văn bản pháp luật số hiệu]"
- Nếu thông tin không đủ để trả lời toàn bộ câu hỏi, chỉ trả lời phần có căn cứ từ các điều luật đã cho.
- Ưu tiên văn bản còn hiệu lực và mới nhất khi có nhiều văn bản quy định cùng vấn đề.

CẤU TRÚC TRẢ LỜI:
1. Tóm tắt: trả lời trực tiếp câu hỏi trong 1-2 câu đầu.
2. Phân tích chi tiết: trình bày từng khía cạnh pháp lý, trích dẫn khoản/điểm/điều cụ thể.
3. Lưu ý thực tiễn: hướng dẫn áp dụng trong thực tế nếu phù hợp (thời hạn, thủ tục, điều kiện cần lưu ý).
4. Kết thúc bằng dòng "Căn cứ pháp lý:" liệt kê TẤT CẢ các điều luật đã trích dẫn, theo mẫu: Điều X [Tên văn bản số hiệu].

PHONG CÁCH: rõ ràng, dễ hiểu cho người không chuyên luật. Dùng gạch đầu dòng khi liệt kê. Giải thích thuật ngữ pháp lý khi cần thiết."""

REFLECTION_INSTRUCTION = """

QUAN TRỌNG - QUY TRÌNH TRẢ LỜI:
Trước khi đưa ra câu trả lời cuối cùng, bạn PHẢI viết phần tự kiểm tra trong thẻ <reflection>...</reflection>. Trong phần reflection, hãy:
1. Liệt kê các điều luật đã được cung cấp và nội dung chính của từng điều.
2. Xác định câu hỏi yêu cầu trả lời những gì.
3. Kiểm tra: mình có trích dẫn điều luật nào NGOÀI danh sách được cung cấp không? Nếu có, phải loại bỏ.
4. Kiểm tra: câu trả lời đã bao quát đủ các khía cạnh của câu hỏi chưa?
5. Kiểm tra: thông tin trích dẫn có khớp với nội dung điều luật gốc không?

Sau </reflection>, viết câu trả lời cuối cùng."""


def strip_reflection(text):
    """Remove <reflection>...</reflection> block, return only the final answer."""
    pattern = r'<reflection>.*?</reflection>\s*'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text


def gen_answer(question, articles_info, fewshot_examples, port, model, reflection=False, max_retries=2):
    fewshot_block = build_fewshot_block(fewshot_examples)
    articles_block = build_articles_block(articles_info)

    system_prompt = SYSTEM_PROMPT_BASE
    if reflection:
        system_prompt += REFLECTION_INSTRUCTION

    user_prompt = f"""{fewshot_block}

Câu hỏi: {question}

Các điều luật được cung cấp (CHỈ sử dụng các điều luật này):
{articles_block}

Hãy trả lời câu hỏi trên. Nhớ: chỉ trích dẫn từ các điều luật đã cung cấp ở trên, không thêm điều luật nào khác."""

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"http://localhost:{port}/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 8192,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                timeout=180,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            raw = (msg.get("content") or "").strip()
            if reflection:
                return strip_reflection(raw)
            return raw
        except Exception:
            if attempt < max_retries:
                time.sleep(1)
            else:
                return ""


def run(port, model, submission_file, workers=4, reflection=False):
    questions = json.loads((BASE / "R2AIStage1DATA.json").read_text())
    submission = json.loads((BASE / submission_file).read_text())
    with open(BASE / "rerank_intersection_scores.pkl", "rb") as f:
        pool = pickle.load(f)
    fewshot_all = pickle.load(open(BASE / "fewshot_top5_reranked.pkl", "rb"))

    print("Loading content cache...")
    content_cache = load_content_cache()
    print(f"  {len(content_cache)} chunks")

    print("Loading metadata cache...")
    meta_cache = load_metadata_cache()
    print(f"  {len(meta_cache)} docs")

    mode_str = "reflection" if reflection else "standard"
    print(f"\nGenerating answers (port={port}, model={model}, workers={workers}, mode={mode_str})...")
    answers = [None] * len(questions)
    done = 0
    t0 = time.time()

    def process(qi):
        q = questions[qi]
        sub = submission[qi]
        cands = pool.get(qi, [])

        # Get article info
        articles_info = []
        for art_str in sub["relevant_articles"]:
            info = get_article_info(art_str, cands, content_cache, meta_cache)
            if info:
                articles_info.append(info)

        if not articles_info:
            return qi, "Không có đủ thông tin từ các điều luật được cung cấp để trả lời câu hỏi này."

        # Get fewshot examples (filter by score)
        fewshot_examples = []
        for ex in fewshot_all[qi][:FEWSHOT_TOP_K * 2]:
            if ex["score_dense"] >= FEWSHOT_MIN_SCORE and (ex.get("answer") or ex.get("full_answer")):
                fewshot_examples.append(ex)
                if len(fewshot_examples) >= FEWSHOT_TOP_K:
                    break

        answer = gen_answer(q["question"], articles_info, fewshot_examples, port, model, reflection=reflection)
        return qi, answer

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, qi): qi for qi in range(len(questions))}
        for future in as_completed(futures):
            qi, answer = future.result()
            answers[qi] = answer
            done += 1
            if done % 100 == 0 or done == len(questions):
                elapsed = time.time() - t0
                eta = elapsed / done * (len(questions) - done)
                print(f"  {done}/{len(questions)} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    print(f"Done in {time.time()-t0:.1f}s")

    # Update submission with answers
    for qi in range(len(questions)):
        submission[qi]["answer"] = answers[qi] or ""

    # Save
    suffix = "_with_answer_reflection.json" if reflection else "_with_answer.json"
    out_name = submission_file.replace(".json", suffix)
    out = BASE / out_name
    out.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_len = np.mean([len(a) for a in answers if a])
    empty = sum(1 for a in answers if not a)
    print(f"Saved: {out}")
    print(f"Avg answer length: {avg_len:.0f} chars, empty: {empty}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", type=str, default="Qwen3-8B-AWQ")
    parser.add_argument("--submission", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reflection", action="store_true", help="Enable reflection before answering")
    args = parser.parse_args()
    run(args.port, args.model, args.submission, args.workers, reflection=args.reflection)
