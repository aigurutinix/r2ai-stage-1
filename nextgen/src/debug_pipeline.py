"""
debug_pipeline.py
Chạy thử pipeline trên một mẫu nhỏ câu hỏi (mặc định 50 câu) và in/lưu kết quả
trung gian ở TỪNG GIAI ĐOẠN ra các file riêng để tìm nguyên nhân giảm độ chính xác:

  1) Retrieval (Hybrid: Dense + Sparse + RRF) — kèm nội dung chunk đã truy hồi
  2) Reranking (Cross-Encoder BGE)            — kèm phân phối điểm trước/sau lọc
  3) LLM Generation (Giai đoạn 4)              — câu trả lời + relevant_docs/articles
  4) Self-Verification (Giai đoạn 5)           — vi phạm/cảnh báo từng quy tắc
  5) Final Result (sau Post-Processing)        — so sánh với ground truth (P/R/F1)

So sánh với ground truth (nếu có) ở 3 mốc: sau Retrieval, sau Rerank, và Final,
để xác định đứt gãy độ chính xác xảy ra ở giai đoạn nào.

Chạy (trên Kaggle GPU — script load model thật, không mock):
    python debug_pipeline.py \
        --input data/R2AIStage1DATA.json \
        --num-questions 50 \
        --output-dir debug_output
"""

import os
import re
import sys
import json
import time
import logging
import argparse
from typing import List, Dict, Optional, Tuple, Set

import torch

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/debug_pipeline.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("debug_pipeline")

from config.settings import Settings
from src.index_bm25 import BM25IndexBuilder
from src.hybrid_retriever import HybridRetriever
from src.reranker import LegalReranker
from src.answer_generator import AnswerGenerator, build_user_prompt, SYSTEM_PROMPT
from src.self_verifier import SelfVerifier
from src.post_processor import PostProcessor

import main as pipeline_main  # tái dùng load_corpus / build_embedding_fn / merge_unique_preserve_order


# =========================================================
# HELPERS: PARSE GROUND TRUTH / CANDIDATE ĐỂ TÍNH P/R/F1
# =========================================================
ARTICLE_NUM_PATTERN = re.compile(r'[Đđ]iều\s+(\d+)')


def extract_article_num(text: str) -> Optional[str]:
    m = ARTICLE_NUM_PATTERN.search(text or "")
    return m.group(1) if m else None


def parse_pipe_doc(doc_string: str) -> str:
    """'04/2017/QH14|Luật Hỗ trợ...' -> '04/2017/QH14'"""
    return doc_string.split("|")[0].strip()


def parse_pipe_article(article_string: str) -> Optional[Tuple[str, str]]:
    """'04/2017/QH14|Luật...|Điều 12' -> ('04/2017/QH14', '12')"""
    parts = article_string.split("|")
    doc_number = parts[0].strip()
    art_num = extract_article_num(parts[-1]) if len(parts) >= 2 else None
    return (doc_number, art_num) if art_num else None


def gt_doc_set(gt_item: Dict) -> Set[str]:
    return {parse_pipe_doc(d) for d in gt_item.get("relevant_docs", []) if d}


def gt_article_set(gt_item: Dict) -> Set[Tuple[str, str]]:
    result = set()
    for a in gt_item.get("relevant_articles", []):
        parsed = parse_pipe_article(a)
        if parsed:
            result.add(parsed)
    return result


def candidate_doc_set(candidates: List[Dict]) -> Set[str]:
    return {
        c.get("metadata", {}).get("doc_number", "").strip()
        for c in candidates
        if c.get("metadata", {}).get("doc_number", "").strip()
    }


def candidate_article_set(candidates: List[Dict]) -> Set[Tuple[str, str]]:
    result = set()
    for c in candidates:
        meta = c.get("metadata", {})
        doc_number = meta.get("doc_number", "").strip()
        art_num = extract_article_num(meta.get("article_id", ""))
        if doc_number and art_num:
            result.add((doc_number, art_num))
    return result


def recall(gt_set: Set, got_set: Set) -> Optional[float]:
    if not gt_set:
        return None
    return len(gt_set & got_set) / len(gt_set)


def precision(gt_set: Set, got_set: Set) -> Optional[float]:
    if not got_set:
        return None
    return len(gt_set & got_set) / len(got_set)


def f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)


def fmt(x: Optional[float]) -> str:
    return f"{x:.2f}" if x is not None else "N/A"


# =========================================================
# LOAD INPUT / GROUND TRUTH
# =========================================================
def load_questions(path: str, num_questions: int) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    questions = raw if isinstance(raw, list) else raw.get("data", raw.get("questions", []))
    if num_questions > 0:
        questions = questions[:num_questions]
    logger.info(f"[Load] {len(questions)} câu hỏi từ {path}")
    return questions


def load_ground_truth(path: Optional[str]) -> Dict[str, Dict]:
    if not path or not os.path.exists(path):
        logger.warning("[Ground Truth] Không có file ground truth — chỉ in kết quả, không tính P/R/F1.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else raw.get("data", [])
    gt = {str(item.get("id", item.get("question_id", ""))): item for item in items}
    logger.info(f"[Ground Truth] Load {len(gt)} nhãn đúng từ {path}")
    return gt


# =========================================================
# RERANK VỚI ĐẦY ĐỦ PHÂN PHỐI ĐIỂM (trước khi lọc threshold)
# =========================================================
def rerank_with_diagnostics(reranker: LegalReranker, query: str, candidates: List[Dict]) -> Dict:
    """
    Lặp lại logic của LegalReranker.rerank() nhưng giữ lại TOÀN BỘ điểm số
    (trước khi lọc threshold) để chẩn đoán xem ngưỡng RERANKER_THRESHOLD
    có đang cắt bỏ văn bản đúng (ground truth) hay không.
    """
    if not candidates:
        return {"scored": [], "filtered": [], "final": [], "fallback_used": False}

    pairs = [[query, doc.get("text", "")] for doc in candidates]
    with torch.no_grad():
        scores = reranker.model.predict(pairs, batch_size=16, show_progress_bar=False)

    scored = []
    for doc, score in zip(candidates, scores):
        d = dict(doc)
        d["rerank_score"] = float(score)
        scored.append(d)
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    threshold = Settings.RERANKER_THRESHOLD
    top_k = Settings.TOP_K_FINAL
    filtered = [d for d in scored if d["rerank_score"] >= threshold]

    fallback_used = False
    if len(filtered) < max(1, top_k // 2):
        fallback_used = True
        filtered = scored[: max(1, top_k // 2)]

    final = filtered[:top_k]
    return {"scored": scored, "filtered": filtered, "final": final, "fallback_used": fallback_used}


def reconstruct_prompt(generator: AnswerGenerator, query: str, contexts: List[Dict]) -> Dict:
    """
    Tái dựng CHÍNH XÁC prompt mà AnswerGenerator.generate() gửi vào LLM
    (cùng build_user_prompt + apply_chat_template), để lưu lại phục vụ debug.
    Prompt giống nhau giữa các lần retry vì context không đổi (chỉ temperature đổi).
    """
    user_prompt = build_user_prompt(query, contexts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        formatted_prompt = generator.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted_prompt = f"{SYSTEM_PROMPT}\n\nNgười dùng: {user_prompt}\n\nTrợ lý:"

    try:
        token_count = len(generator.tokenizer(formatted_prompt)["input_ids"])
    except Exception:
        token_count = None

    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "formatted_prompt": formatted_prompt,
        "user_prompt_char_len": len(user_prompt),
        "formatted_prompt_char_len": len(formatted_prompt),
        "prompt_token_count": token_count,
        "max_context_chars": Settings.MAX_CONTEXT_CHARS,
        "context_was_truncated": "...[cắt bớt]" in user_prompt,
        "num_context_docs": len(contexts),
    }


def doc_snapshot(doc: Dict, text_len: int = 300) -> Dict:
    meta = doc.get("metadata", {})
    return {
        "id": doc.get("id"),
        "doc_number": meta.get("doc_number", ""),
        "title": meta.get("title", ""),
        "article_id": meta.get("article_id", ""),
        "dense_score": doc.get("dense_score"),
        "bm25_score": doc.get("bm25_score"),
        "rrf_score": doc.get("rrf_score"),
        "rerank_score": doc.get("rerank_score"),
        "text_snippet": (doc.get("text", "") or "")[:text_len],
    }


# =========================================================
# XỬ LÝ 1 CÂU HỎI VỚI INSTRUMENTATION ĐẦY ĐỦ
# =========================================================
def debug_process_question(
    question_item: Dict,
    gt_item: Optional[Dict],
    retriever: HybridRetriever,
    reranker: LegalReranker,
    generator: AnswerGenerator,
    verifier: SelfVerifier,
    post_processor: PostProcessor,
) -> Dict:
    item_id = str(question_item.get("id", question_item.get("question_id", "")))
    query = question_item.get("question", question_item.get("query", ""))
    gt_item = gt_item or {}

    record = {"id": item_id, "question": query}

    # --------- GIAI ĐOẠN 2: RETRIEVAL (+ chunking content) ---------
    query_variants = retriever.expander.expand(query)
    dense_results, sparse_results = retriever._multi_query_retrieve(query)
    merged = retriever._rrf_merge([dense_results, sparse_results])

    gt_docs = gt_doc_set(gt_item)
    gt_arts = gt_article_set(gt_item)

    retr_doc_recall = recall(gt_docs, candidate_doc_set(merged))
    retr_art_recall = recall(gt_arts, candidate_article_set(merged))

    record["retrieval"] = {
        "query_variants": query_variants,
        "dense_count": len(dense_results),
        "sparse_count": len(sparse_results),
        "merged_count": len(merged),
        "candidates": [doc_snapshot(d) for d in merged],
        "gt_doc_count": len(gt_docs),
        "gt_article_count": len(gt_arts),
        "doc_recall": retr_doc_recall,
        "article_recall": retr_art_recall,
    }

    # --------- GIAI ĐOẠN 3: RERANKING ---------
    rerank_diag = rerank_with_diagnostics(reranker, query, merged)
    final_contexts = rerank_diag["final"]
    scored = rerank_diag["scored"]

    rerank_doc_recall = recall(gt_docs, candidate_doc_set(final_contexts))
    rerank_art_recall = recall(gt_arts, candidate_article_set(final_contexts))

    record["rerank"] = {
        "input_count": len(merged),
        "score_min": min((d["rerank_score"] for d in scored), default=None),
        "score_max": max((d["rerank_score"] for d in scored), default=None),
        "threshold": Settings.RERANKER_THRESHOLD,
        "threshold_pass_count": len(rerank_diag["filtered"]) if not rerank_diag["fallback_used"] else None,
        "fallback_used": rerank_diag["fallback_used"],
        "final_count": len(final_contexts),
        "final_candidates": [doc_snapshot(d) for d in final_contexts],
        "doc_recall": rerank_doc_recall,
        "article_recall": rerank_art_recall,
    }

    # --------- GIAI ĐOẠN 4 + 5: GENERATION + SELF-VERIFICATION (với retry) ---------
    # Lưu lại CHÍNH XÁC prompt gửi vào LLM (trước khi gọi generate) để debug context.
    prompt_info = reconstruct_prompt(generator, query, final_contexts)

    attempts = []
    llm_output: Dict = {}
    verify_passed = False
    verify_violations: List[str] = []

    for attempt in range(Settings.LLM_MAX_RETRIES + 1):
        temp = Settings.LLM_TEMPERATURE if attempt == 0 else Settings.LLM_REGEN_TEMPERATURE
        llm_output = generator.generate(query, final_contexts, temperature=temp)
        answer_text = llm_output.get("answer", "")

        verify_result = verifier.verify(answer_text, query, final_contexts)
        verify_violations = verify_result.violations

        attempts.append({
            "attempt": attempt + 1,
            "temperature": temp,
            "answer": answer_text,
            "relevant_docs": llm_output.get("relevant_docs", []),
            "relevant_articles": llm_output.get("relevant_articles", []),
            "verify_passed": verify_result.passed,
            "verify_violations": verify_result.violations,
            "verify_warnings": verify_result.warnings,
            "extracted_articles": verify_result.extracted_articles,
            "extracted_doc_numbers": verify_result.extracted_doc_numbers,
        })

        if verify_result.passed:
            verify_passed = True
            break

    record["generation"] = {
        "prompt": prompt_info,
        "attempts": attempts,
        "was_regenerated": len(attempts) > 1,
        "final_answer": llm_output.get("answer", ""),
    }
    record["verification"] = {
        "passed": verify_passed,
        "final_violations": verify_violations,
        "num_attempts": len(attempts),
    }

    # --------- GIAI ĐOẠN 6: POST-PROCESSING + SO SÁNH GROUND TRUTH ---------
    result = post_processor.process_single(
        item_id=item_id, query=query, answer=llm_output.get("answer", ""), context_docs=final_contexts
    )
    # Precision-first (khớp main.py): ưu tiên set chọn lọc của generator
    # (top-N + GIAO với Điều LLM thực sự dẫn), chỉ dùng PostProcessor làm fallback.
    result["relevant_docs"] = (
        llm_output.get("relevant_docs") or result.get("relevant_docs", [])
    )
    result["relevant_articles"] = (
        llm_output.get("relevant_articles") or result.get("relevant_articles", [])
    )

    final_doc_set = {parse_pipe_doc(d) for d in result["relevant_docs"]}
    final_art_set = {p for p in (parse_pipe_article(a) for a in result["relevant_articles"]) if p}

    final_doc_recall = recall(gt_docs, final_doc_set)
    final_doc_precision = precision(gt_docs, final_doc_set)
    final_art_recall = recall(gt_arts, final_art_set)
    final_art_precision = precision(gt_arts, final_art_set)

    record["final"] = {
        "answer": result["answer"],
        "relevant_docs": result["relevant_docs"],
        "relevant_articles": result["relevant_articles"],
        "missing_docs": sorted(gt_docs - final_doc_set),
        "extra_docs": sorted(final_doc_set - gt_docs),
        "doc_precision": final_doc_precision,
        "doc_recall": final_doc_recall,
        "doc_f1": f1(final_doc_precision, final_doc_recall),
        "article_precision": final_art_precision,
        "article_recall": final_art_recall,
        "article_f1": f1(final_art_precision, final_art_recall),
    }

    logger.info(
        f"[Q-{item_id}] retrieval_doc_recall={fmt(retr_doc_recall)} | "
        f"rerank_doc_recall={fmt(rerank_doc_recall)} (fallback={rerank_diag['fallback_used']}) | "
        f"verify_pass={verify_passed} (attempts={len(attempts)}) | "
        f"final_doc_f1={fmt(record['final']['doc_f1'])} article_f1={fmt(record['final']['article_f1'])}"
    )

    return record


# =========================================================
# SUMMARY REPORT
# =========================================================
def build_summary(records: List[Dict]) -> str:
    def avg(values: List[Optional[float]]) -> Optional[float]:
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else None

    n = len(records)
    retr_doc_recalls = [r["retrieval"]["doc_recall"] for r in records]
    retr_art_recalls = [r["retrieval"]["article_recall"] for r in records]
    rerank_doc_recalls = [r["rerank"]["doc_recall"] for r in records]
    rerank_art_recalls = [r["rerank"]["article_recall"] for r in records]
    fallback_count = sum(1 for r in records if r["rerank"]["fallback_used"])
    verify_first_pass = sum(1 for r in records if r["verification"]["num_attempts"] == 1 and r["verification"]["passed"])
    verify_regenerated = sum(1 for r in records if r["verification"]["num_attempts"] > 1)
    verify_failed_all = sum(1 for r in records if not r["verification"]["passed"])
    final_doc_f1s = [r["final"]["doc_f1"] for r in records]
    final_art_f1s = [r["final"]["article_f1"] for r in records]
    final_doc_precisions = [r["final"]["doc_precision"] for r in records]
    final_doc_recalls = [r["final"]["doc_recall"] for r in records]
    final_art_precisions = [r["final"]["article_precision"] for r in records]
    final_art_recalls = [r["final"]["article_recall"] for r in records]

    def f2(p: Optional[float], r: Optional[float]) -> Optional[float]:
        # F2 chính thức của BTC: tính từ macro Precision & macro Recall
        if p is None or r is None or (4 * p + r) == 0:
            return None
        return 5 * p * r / (4 * p + r)

    macro_art_p, macro_art_r = avg(final_art_precisions), avg(final_art_recalls)
    macro_doc_p, macro_doc_r = avg(final_doc_precisions), avg(final_doc_recalls)

    # Chẩn đoán: câu nào retrieval tốt nhưng rerank tệ -> nghi ngờ threshold/reranker
    rerank_drop = sorted(
        [
            (r["id"], r["retrieval"]["doc_recall"], r["rerank"]["doc_recall"])
            for r in records
            if r["retrieval"]["doc_recall"] is not None and r["rerank"]["doc_recall"] is not None
            and r["retrieval"]["doc_recall"] - r["rerank"]["doc_recall"] > 0.3
        ],
        key=lambda x: (x[1] - x[2]),
        reverse=True,
    )

    # Chẩn đoán: rerank tốt nhưng final tệ -> nghi ngờ LLM/extraction
    final_drop = sorted(
        [
            (r["id"], r["rerank"]["doc_recall"], r["final"]["doc_recall"])
            for r in records
            if r["rerank"]["doc_recall"] is not None and r["final"]["doc_recall"] is not None
            and r["rerank"]["doc_recall"] - r["final"]["doc_recall"] > 0.3
        ],
        key=lambda x: (x[1] - x[2]),
        reverse=True,
    )

    # Chẩn đoán: retrieval đã tệ ngay từ đầu -> nghi ngờ embedding/BM25/corpus
    low_retrieval = sorted(
        [(r["id"], r["retrieval"]["doc_recall"]) for r in records if r["retrieval"]["doc_recall"] is not None],
        key=lambda x: x[1],
    )[:10]

    lines = []
    lines.append(f"# Báo cáo chẩn đoán Pipeline — {n} câu hỏi\n")
    lines.append("## 1. Retrieval (Hybrid Dense+Sparse+RRF)")
    lines.append(f"- Doc recall trung bình: {fmt(avg(retr_doc_recalls))}")
    lines.append(f"- Article recall trung bình: {fmt(avg(retr_art_recalls))}\n")

    lines.append("## 2. Reranking (Cross-Encoder)")
    lines.append(f"- Doc recall trung bình: {fmt(avg(rerank_doc_recalls))}")
    lines.append(f"- Article recall trung bình: {fmt(avg(rerank_art_recalls))}")
    lines.append(f"- Số câu kích hoạt fallback (threshold lọc quá chặt): {fallback_count}/{n}\n")

    lines.append("## 3. Self-Verification")
    lines.append(f"- Pass ngay lần đầu: {verify_first_pass}/{n}")
    lines.append(f"- Phải regenerate (lần 2): {verify_regenerated}/{n}")
    lines.append(f"- Fail cả 2 lần (giữ kết quả tốt nhất): {verify_failed_all}/{n}\n")

    lines.append("## 4. Final Result vs Ground Truth (macro-average — đúng cách chấm BTC)")
    lines.append(
        f"- ARTICLES: precision={fmt(macro_art_p)} | recall={fmt(macro_art_r)} "
        f"| **F2={fmt(f2(macro_art_p, macro_art_r))}**"
    )
    lines.append(
        f"- DOCS:     precision={fmt(macro_doc_p)} | recall={fmt(macro_doc_r)} "
        f"| **F2={fmt(f2(macro_doc_p, macro_doc_r))}**"
    )
    lines.append(f"- Doc F1 trung bình: {fmt(avg(final_doc_f1s))} | Article F1 trung bình: {fmt(avg(final_art_f1s))}\n")

    lines.append("## 5. Chẩn đoán đứt gãy độ chính xác")
    lines.append(
        f"- {len(rerank_drop)} câu có doc_recall giảm >0.3 từ Retrieval -> Rerank "
        "(nghi ngờ RERANKER_THRESHOLD quá chặt hoặc model rerank chấm sai):"
    )
    for qid, before, after in rerank_drop[:10]:
        lines.append(f"  - Q{qid}: retrieval_recall={before:.2f} -> rerank_recall={after:.2f}")

    lines.append(
        f"\n- {len(final_drop)} câu có doc_recall giảm >0.3 từ Rerank -> Final "
        "(nghi ngờ LLM không trích dẫn đúng / extraction logic ở generator/post_processor):"
    )
    for qid, before, after in final_drop[:10]:
        lines.append(f"  - Q{qid}: rerank_recall={before:.2f} -> final_recall={after:.2f}")

    lines.append(
        f"\n- Top 10 câu có doc_recall@Retrieval thấp nhất "
        "(nghi ngờ embedding/BM25/corpus thiếu dữ liệu hoặc câu hỏi diễn đạt khó match):"
    )
    for qid, val in low_retrieval:
        lines.append(f"  - Q{qid}: retrieval_recall={fmt(val)}")

    return "\n".join(lines)


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Debug pipeline — in/lưu kết quả từng giai đoạn")
    parser.add_argument("--input", type=str, default="data/R2AIStage1DATA.json")
    parser.add_argument("--ground-truth", type=str, default="")
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="debug_output")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("[INIT] Đang khởi tạo các module (giống main.py)...")
    corpus = pipeline_main.load_corpus()
    bm25_builder = BM25IndexBuilder()
    if os.path.exists(Settings.BM25_INDEX_PATH) and corpus:
        bm25_builder.load()
    elif corpus:
        bm25_builder.build(corpus).save()
    else:
        from rank_bm25 import BM25Okapi
        bm25_builder.documents = []
        bm25_builder.bm25 = BM25Okapi([[]])

    embed_fn = pipeline_main.build_embedding_fn()
    generator = AnswerGenerator()
    retriever = HybridRetriever(bm25_builder=bm25_builder, embedding_fn=embed_fn, llm_pipeline=generator.pipe)
    reranker = LegalReranker()
    verifier = SelfVerifier.from_manifest_file()
    post_processor = PostProcessor.from_manifest_file()
    logger.info("✅ Tất cả module đã sẵn sàng!\n")

    questions = load_questions(args.input, args.num_questions)
    ground_truth = load_ground_truth(args.ground_truth)

    records: List[Dict] = []
    start = time.time()

    for i, q in enumerate(questions):
        item_id = str(q.get("id", q.get("question_id", "")))
        try:
            record = debug_process_question(
                question_item=q,
                gt_item=ground_truth.get(item_id),
                retriever=retriever,
                reranker=reranker,
                generator=generator,
                verifier=verifier,
                post_processor=post_processor,
            )
        except Exception as e:
            logger.error(f"[Q-{item_id}] Lỗi xử lý: {e}", exc_info=args.debug)
            record = {"id": item_id, "question": q.get("question", ""), "error": str(e)}

        records.append(record)

        # Flush ra file sau MỖI câu để không mất dữ liệu nếu crash giữa chừng
        _dump_stage_files(records, args.output_dir)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            logger.info(f"[Progress] {i + 1}/{len(questions)} | {elapsed:.1f}s")

    valid_records = [r for r in records if "error" not in r]
    summary_md = build_summary(valid_records) if valid_records else "Không có câu hỏi nào xử lý thành công."
    summary_path = os.path.join(args.output_dir, "summary_report.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    logger.info("\n" + "=" * 60)
    logger.info(f"🎉 HOÀN THÀNH DEBUG {len(records)} CÂU HỎI")
    logger.info(f"   Kết quả lưu tại: {args.output_dir}/")
    logger.info(f"   Báo cáo tổng hợp: {summary_path}")
    logger.info("=" * 60)
    print("\n" + summary_md)


def _dump_stage_files(records: List[Dict], output_dir: str):
    stage_files = {
        "01_retrieval.json": [{"id": r["id"], "question": r.get("question"), **r.get("retrieval", {})} for r in records if "retrieval" in r],
        "02_rerank.json": [{"id": r["id"], "question": r.get("question"), **r.get("rerank", {})} for r in records if "rerank" in r],
        "03_llm_generation.json": [{"id": r["id"], "question": r.get("question"), **r.get("generation", {})} for r in records if "generation" in r],
        "04_self_verification.json": [{"id": r["id"], "question": r.get("question"), **r.get("verification", {})} for r in records if "verification" in r],
        "05_final_result.json": [{"id": r["id"], "question": r.get("question"), **r.get("final", {})} for r in records if "final" in r],
    }
    for filename, data in stage_files.items():
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
