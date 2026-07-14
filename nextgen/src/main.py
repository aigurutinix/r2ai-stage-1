"""
main.py
Pipeline Orchestrator — Chạy toàn bộ 2000 câu hỏi end-to-end

Giai đoạn 2: Hybrid Retrieval (BM25 + Qdrant + RRF + Multi-Query)
Giai đoạn 3: Reranking (BGE Cross-Encoder)
Giai đoạn 4: Answer Generation (DeepSeek-R1-14B)
Giai đoạn 5: Self-Verification (5 Quy tắc vàng)
Giai đoạn 6: Post-Processing + Submission Packaging

Tối ưu cho Kaggle GPU (T4/P100).
Chạy: python main.py --input R2AIStage1DATA.json [--resume] [--debug]
"""

import os
import re
import sys
import json
import time
import logging
import argparse
import pickle
from typing import List, Dict, Optional

# =========================================================
# SETUP LOGGING
# =========================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("main")

from config.settings import Settings
from src.index_bm25 import BM25IndexBuilder
from src.hybrid_retriever import HybridRetriever
from src.reranker import LegalReranker
from src.answer_generator import AnswerGenerator
from src.self_verifier import SelfVerifier
from src.post_processor import PostProcessor
from src.evaluator import PipelineEvaluator


# =========================================================
# EMBEDDING FUNCTION (AITeamVN/Vietnamese_Embedding)
# =========================================================
def build_embedding_fn():
    """
    Khởi tạo hàm embedding dùng AITeamVN/Vietnamese_Embedding.
    Load model một lần duy nhất để tránh overhead.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Embedding] Loading AITeamVN/Vietnamese_Embedding trên {device.upper()}...")
    embed_model = SentenceTransformer(Settings.EMBEDDING_MODEL, device=device)
    logger.info("✅ Embedding model loaded.")

    def embed_fn(text: str) -> List[float]:
        vector = embed_model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    return embed_fn


# =========================================================
# LOAD CORPUS CHO BM25
# =========================================================
def load_corpus() -> List[Dict]:
    """
    Load corpus văn bản pháp lý đã chuẩn hóa từ Giai đoạn 0.
    Tìm lần lượt: law_corpus_clean.json → bm25_corpus.pkl
    """
    corpus_json = os.path.join(Settings.DATA_DIR, "law_corpus_clean.json")
    if os.path.exists(corpus_json):
        logger.info(f"[Corpus] Load từ {corpus_json}")
        with open(corpus_json, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.warning(
        "[Corpus] Không tìm thấy law_corpus_clean.json. "
        "BM25 sẽ chạy với corpus rỗng — chỉ dùng Dense Search."
    )
    return []


# =========================================================
# LOAD QUESTIONS
# =========================================================
def load_questions(input_path: str) -> List[Dict]:
    """
    Load 2000 câu hỏi từ file JSON của BTC.
    Hỗ trợ các cấu trúc: List[Dict] hoặc {"data": List[Dict]}
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Xử lý các cấu trúc JSON khác nhau
    if isinstance(raw, list):
        questions = raw
    elif isinstance(raw, dict):
        questions = raw.get("data", raw.get("questions", raw.get("items", [])))
    else:
        raise ValueError(f"Cấu trúc JSON không nhận dạng được: {type(raw)}")

    logger.info(f"[Load] Đã load {len(questions)} câu hỏi từ {input_path}")
    return questions


# =========================================================
# LOAD CHECKPOINT (Resume)
# =========================================================
def load_checkpoint(checkpoint_path: str) -> Dict[str, Dict]:
    """Load kết quả đã xử lý từ lần chạy trước."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"[Resume] Load {len(data)} kết quả từ checkpoint {checkpoint_path}")
        return {str(item["id"]): item for item in data}
    return {}


def save_checkpoint(results: List[Dict], checkpoint_path: str):
    """Lưu checkpoint sau mỗi batch."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def merge_unique_preserve_order(primary: List[str], secondary: List[str]) -> List[str]:
    """Hợp nhất 2 danh sách, giữ nguyên thứ tự, loại trùng lặp. `primary` luôn đứng trước."""
    merged = list(primary)
    seen = set(primary)
    for item in secondary:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


# =========================================================
# PROCESS SINGLE QUESTION
# =========================================================
def process_question(
    question_item: Dict,
    retriever: HybridRetriever,
    reranker: LegalReranker,
    generator: AnswerGenerator,
    verifier: SelfVerifier,
    post_processor: PostProcessor,
    evaluator: PipelineEvaluator,
) -> Dict:
    """
    Xử lý một câu hỏi qua toàn bộ pipeline:
    Retrieval → Rerank → Generate → Verify → Post-Process
    """
    item_id = str(question_item.get("id", question_item.get("question_id", "")))
    query = question_item.get("question", question_item.get("query", ""))

    logger.debug(f"[Q-{item_id}] {query[:80]}...")

    # --------- GIAI ĐOẠN 2: HYBRID RETRIEVAL ---------
    raw_candidates = retriever.retrieve(query)

    # --------- GIAI ĐOẠN 3: RERANKING ---------
    final_contexts = reranker.rerank(query, raw_candidates)

    # --------- GIAI ĐOẠN 4: ANSWER GENERATION ---------
    llm_output = {}
    verify_passed = False
    verify_violations = []
    was_regenerated = False

    for attempt in range(Settings.LLM_MAX_RETRIES + 1):
        temp = Settings.LLM_TEMPERATURE if attempt == 0 else Settings.LLM_REGEN_TEMPERATURE

        # BÂY GIỜ: generator.generate() trả về một DICT {"answer": ..., "relevant_docs": ..., "relevant_articles": ...}
        llm_output = generator.generate(query, final_contexts, temperature=temp)
        answer_text = llm_output.get("answer", "")

        # --------- GIAI ĐOẠN 5: SELF-VERIFICATION ---------
        verify_result = verifier.verify(answer_text, query, final_contexts)
        verify_violations = verify_result.violations

        if verify_result.passed:
            verify_passed = True
            if attempt > 0:
                was_regenerated = True
            break
        else:
            if attempt < Settings.LLM_MAX_RETRIES:
                logger.warning(
                    f"[Q-{item_id}] Attempt {attempt+1} thất bại. "
                    f"Regenerate với temp={Settings.LLM_REGEN_TEMPERATURE}..."
                )
            else:
                logger.error(
                    f"[Q-{item_id}] Đã thử {Settings.LLM_MAX_RETRIES+1} lần, "
                    f"vẫn fail verification. Giữ kết quả tốt nhất."
                )
                was_regenerated = True

    # --------- GIAI ĐOẠN 6: POST-PROCESSING ---------
    # Gửi text thuần vào post_processor để xử lý hậu kỳ (nếu cần)
    result = post_processor.process_single(
        item_id=item_id,
        query=query,
        answer=llm_output.get("answer", ""),
        context_docs=final_contexts
    )

    # ĐẢM BẢO CHUẨN ĐỊNH DẠNG BÀI THI (precision-first để tối ưu F2):
    # Generator đã chọn lọc relevant_docs/relevant_articles theo hướng top-N + GIAO với các
    # Điều mà LLM thực sự trích dẫn (xem AnswerGenerator._extract_references). Vì vậy KHÔNG
    # merge toàn bộ tập Regex của PostProcessor (sẽ làm phình precision → tụt F2). Chỉ dùng
    # PostProcessor làm FALLBACK khi generator không chọn được Điều/văn bản nào.
    result["relevant_docs"] = (
        llm_output.get("relevant_docs") or result.get("relevant_docs", [])
    )
    result["relevant_articles"] = (
        llm_output.get("relevant_articles") or result.get("relevant_articles", [])
    )

    # Log metrics hệ thống
    evaluator.log_item(
        item_id=item_id,
        query=query,
        num_retrieved=len(raw_candidates),
        num_after_rerank=len(final_contexts),
        verify_passed=verify_passed,
        verify_violations=verify_violations,
        was_regenerated=was_regenerated,
        answer_length=len(llm_output.get("answer", "")),
        relevant_docs_count=len(result["relevant_docs"])
    )

    return result
# =========================================================
# MAIN
# =========================================================
def main(args):
    logger.info("=" * 60)
    logger.info("🚀 BẮT ĐẦU PIPELINE RAG PHÁP LÝ VIỆT NAM")
    logger.info(f"   Input: {args.input}")
    logger.info(f"   Resume: {args.resume}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info("=" * 60)

    os.makedirs(Settings.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Settings.LOG_DIR, exist_ok=True)

    checkpoint_path = os.path.join(Settings.OUTPUT_DIR, "checkpoint.json")

    # --------- KHỞI TẠO CÁC MODULE ---------
    logger.info("\n[INIT] Đang khởi tạo các module...")

    # Corpus & BM25
    corpus = load_corpus()
    bm25_builder = BM25IndexBuilder()
    if os.path.exists(Settings.BM25_INDEX_PATH) and corpus:
        bm25_builder.load()
    elif corpus:
        bm25_builder.build(corpus).save()
    else:
        # Corpus rỗng → BM25 dummy
        bm25_builder.documents = []
        from rank_bm25 import BM25Okapi
        bm25_builder.bm25 = BM25Okapi([[]])

    # Embedding function
    embed_fn = build_embedding_fn()

    # Khởi tạo LLM trước (để dùng cho Multi-Query expansion)
    logger.info("[INIT] Loading LLM (có thể mất 3-5 phút lần đầu)...")
    generator = AnswerGenerator()

    # Hybrid Retriever (dùng LLM pipeline cho Multi-Query)
    retriever = HybridRetriever(
        bm25_builder=bm25_builder,
        embedding_fn=embed_fn,
        llm_pipeline=generator.pipe  # Tái dùng pipeline đã load
    )

    # Reranker
    reranker = LegalReranker()

    # Verifier & Post-Processor
    verifier = SelfVerifier.from_manifest_file()
    post_processor = PostProcessor.from_manifest_file()
    evaluator = PipelineEvaluator()

    logger.info("✅ Tất cả module đã sẵn sàng!\n")

    # --------- LOAD QUESTIONS ---------
    questions = load_questions(args.input)

    # --------- RESUME LOGIC ---------
    processed_results: Dict[str, Dict] = {}
    if args.resume:
        processed_results = load_checkpoint(checkpoint_path)
        logger.info(f"[Resume] Bỏ qua {len(processed_results)} câu đã xử lý.")

    # --------- PROCESSING LOOP ---------
    results_list: List[Dict] = list(processed_results.values())
    start_time = time.time()

    pending = [q for q in questions if str(q.get("id", q.get("question_id", ""))) not in processed_results]
    logger.info(f"[Loop] Cần xử lý {len(pending)} câu hỏi...")

    for i, question_item in enumerate(pending):
        try:
            result = process_question(
                question_item=question_item,
                retriever=retriever,
                reranker=reranker,
                generator=generator,
                verifier=verifier,
                post_processor=post_processor,
                evaluator=evaluator,
            )
            results_list.append(result)

        except Exception as e:
            item_id = str(question_item.get("id", f"unknown_{i}"))
            query = question_item.get("question", "")
            logger.error(f"[Q-{item_id}] Pipeline lỗi: {e}", exc_info=args.debug)

            # Fallback: Tạo bản ghi rỗng để không mất ID
            results_list.append({
                "id": item_id,
                "question": query,
                "answer": "Không thể xử lý câu hỏi này do lỗi hệ thống.",
                "relevant_docs": [],
                "relevant_articles": [],
            })

        # Checkpoint mỗi batch
        if (i + 1) % args.batch_size == 0:
            save_checkpoint(results_list, checkpoint_path)
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed * 60  # câu/phút
            eta = (len(pending) - i - 1) / (speed / 60) / 60  # giờ
            logger.info(
                f"[Progress] {len(results_list)}/{len(questions)} | "
                f"Speed: {speed:.1f} câu/phút | ETA: {eta:.1f} giờ"
            )

    # --------- VALIDATION ---------
    logger.info("\n[Validation] Kiểm tra kết quả cuối...")
    validation_report = post_processor.validate_results(results_list)
    logger.info(f"Validation: {'✅ PASS' if validation_report['is_valid'] else '❌ FAIL'}")
    if validation_report["errors"]:
        for err in validation_report["errors"][:10]:
            logger.error(f"  {err}")

    # --------- ĐÓNG GÓI ---------
    json_path, zip_path = post_processor.package_submission(results_list)

    # --------- BÁO CÁO ---------
    report = evaluator.generate_report()
    evaluator.save_detailed_log()

    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("🎉 PIPELINE HOÀN THÀNH!")
    logger.info(f"   Tổng thời gian: {total_time/3600:.1f} giờ")
    logger.info(f"   Kết quả JSON: {json_path}")
    logger.info(f"   File nộp:     {zip_path}")
    logger.info(f"   Báo cáo:      {Settings.EVAL_REPORT_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal RAG Pipeline")
    parser.add_argument(
        "--input", type=str, default="R2AIStage1DATA.json",
        help="Path tới file 2000 câu hỏi của BTC"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Tiếp tục từ checkpoint (bỏ qua câu đã xử lý)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Lưu checkpoint sau mỗi N câu (default: 50)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Bật chế độ debug (verbose logging + traceback)"
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    main(args)
