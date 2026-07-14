"""
fast_retrieval.py  (LUỒNG RIÊNG — KHÔNG ảnh hưởng main.py / debug_pipeline.py)

Pipeline NHANH chỉ gồm:  Hybrid Retrieval  ->  Reranking  ->  trích top-N relevant_docs/articles.
BỎ HẲN LLM Generation và Self-Verification để tối ưu thời gian (answer="").

Mục đích: tối ưu/leo điểm TRUY HỒI (Precision/Recall/F2) — đường điểm này KHÔNG cần `answer`.
Tốc độ: ~1-2 giây/câu (so với ~29 giây/câu khi có LLM) → 2000 câu chỉ ~30-60 phút,
đủ để nộp nhiều lần/ngày. (Đánh đổi: mất điểm 5 tiêu chí QA vì answer rỗng.)

Chạy:
    python fast_retrieval.py --input data/R2AIStage1DATA.json --output output/results.json
    python fast_retrieval.py --input data/R2AIStage1DATA.json --num-questions 50   # thử nhanh
"""
import os
import sys
import json
import time
import logging
import argparse
from typing import List, Dict

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/fast_retrieval.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("fast_retrieval")

from config.settings import Settings
from src.index_bm25 import BM25IndexBuilder
from src.hybrid_retriever import HybridRetriever
from src.reranker import LegalReranker
from src.reference_extractor import load_manifest, extract_references_topn, extract_references_all
from src.llm_selector import select_candidates
from src.answer_intersect import intersect_select


# =========================================================
# HELPERS (độc lập, không import main.py để khỏi kéo theo module LLM)
# =========================================================
def build_embedding_fn():
    """Embedding dùng AITeamVN/Vietnamese_Embedding (KHÔNG phải LLM)."""
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Embedding] Loading {Settings.EMBEDDING_MODEL} trên {device.upper()}...")
    embed_model = SentenceTransformer(Settings.EMBEDDING_MODEL, device=device)
    logger.info("✅ Embedding model loaded.")

    def embed_fn(text: str) -> List[float]:
        return embed_model.encode(text, normalize_embeddings=True).tolist()

    return embed_fn


def load_corpus() -> List[Dict]:
    corpus_json = os.path.join(Settings.DATA_DIR, "law_corpus_clean.json")
    if os.path.exists(corpus_json):
        logger.info(f"[Corpus] Load từ {corpus_json}")
        with open(corpus_json, "r", encoding="utf-8") as f:
            return json.load(f)
    logger.warning("[Corpus] Không thấy law_corpus_clean.json — BM25 rỗng, chỉ Dense Search.")
    return []


def load_questions(path: str, num: int = 0) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    qs = raw if isinstance(raw, list) else raw.get("data", raw.get("questions", raw.get("items", [])))
    return qs[:num] if num and num > 0 else qs


def norm_id(v):
    """Giữ id dạng SỐ NGUYÊN nếu có thể ("1" -> 1), ngược lại giữ nguyên."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    s = str(v).strip()
    return int(s) if s.lstrip("-").isdigit() else s


def save_results(results: List[Dict], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# =========================================================
# MAIN
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Fast retrieval-only pipeline (KHÔNG LLM)")
    ap.add_argument("--input", required=True, help="File câu hỏi JSON")
    ap.add_argument("--output", default=os.path.join(Settings.OUTPUT_DIR, "results.json"))
    ap.add_argument("--num-questions", type=int, default=0, help="0 = tất cả")
    ap.add_argument("--flush-every", type=int, default=200, help="Ghi tạm results.json sau mỗi N câu")
    ap.add_argument("--llm-select", action="store_true",
                    help="Bật LLM chọn lọc từ pool rerank (số lượng biến thiên, làm nguồn trích dẫn chính)")
    ap.add_argument("--pool-k", type=int, default=8,
                    help="Số ứng viên rerank đưa cho LLM chọn (chỉ dùng khi --llm-select)")
    ap.add_argument("--max-select", type=int, default=5,
                    help="Số điều TỐI ĐA LLM được chọn (số lượng thực tế biến thiên 1..max-select)")
    ap.add_argument("--llm-answer", action="store_true",
                    help="Sinh answer THẬT rồi lấy GIAO citation∩pool rerank (answer≠'', còn ăn điểm QA)")
    args = ap.parse_args()

    if args.llm_answer:
        mode = "RETRIEVAL + RERANK + LLM-ANSWER (giao citation∩pool)"
    elif args.llm_select:
        mode = "RETRIEVAL + RERANK + LLM-SELECT"
    else:
        mode = "RETRIEVAL + RERANK (KHÔNG LLM)"
    logger.info(f"[INIT] Khởi tạo {mode}...")
    corpus = load_corpus()
    bm25 = BM25IndexBuilder()
    if os.path.exists(Settings.BM25_INDEX_PATH) and corpus:
        bm25.load()
    elif corpus:
        bm25.build(corpus).save()
    else:
        from rank_bm25 import BM25Okapi
        bm25.documents = []
        bm25.bm25 = BM25Okapi([[]])

    embed_fn = build_embedding_fn()
    retriever = HybridRetriever(bm25_builder=bm25, embedding_fn=embed_fn, llm_pipeline=None)
    reranker = LegalReranker()
    manifest = load_manifest()

    # Chỉ nạp LLM khi cần (tái dùng AnswerGenerator để khỏi sửa code cũ)
    llm_pipe = llm_tokenizer = None
    generator = None
    if args.llm_answer:
        logger.info("[INIT] --llm-answer: đang nạp LLM để sinh answer...")
        from src.answer_generator import AnswerGenerator
        generator = AnswerGenerator()
    elif args.llm_select:
        logger.info("[INIT] --llm-select: đang nạp LLM làm bộ chọn...")
        from src.answer_generator import AnswerGenerator
        _gen = AnswerGenerator()
        llm_pipe, llm_tokenizer = _gen.pipe, _gen.tokenizer
    logger.info(f"✅ Sẵn sàng ({mode}).")

    questions = load_questions(args.input, args.num_questions)
    logger.info(f"[Load] {len(questions)} câu hỏi từ {args.input}")

    results: List[Dict] = []
    start = time.time()

    for i, q in enumerate(questions):
        qid = q.get("id", q.get("question_id", ""))
        question = q.get("question", "")
        answer_text = ""   # mặc định rỗng (chỉ mode --llm-answer mới điền)
        try:
            raw_candidates = retriever.retrieve(question)
            if args.llm_answer:
                # rerank pool lớn -> LLM SINH answer -> GIAO citation∩pool -> derive docs từ điều chọn
                ranked = reranker.rerank(question, raw_candidates, top_k=max(args.pool_k, Settings.TOP_K_FINAL))
                out = generator.generate(question, ranked)
                answer_text = out.get("answer", "")
                kept = intersect_select(answer_text, ranked, pool_k=args.pool_k, max_out=args.max_select)
                rel_docs, rel_articles = extract_references_all(kept, manifest)
            elif args.llm_select:
                # rerank lấy pool lớn hơn rồi để LLM chọn lọc (số lượng biến thiên)
                ranked = reranker.rerank(question, raw_candidates, top_k=max(args.pool_k, Settings.TOP_K_FINAL))
                chosen = select_candidates(
                    question, ranked, llm_pipe, llm_tokenizer,
                    pool_k=args.pool_k, max_select=args.max_select,
                )
                # LLM là nguồn chính: lấy TẤT CẢ điều/văn bản LLM đã chọn (không cap cố định)
                rel_docs, rel_articles = extract_references_all(chosen, manifest)
            else:
                ranked = reranker.rerank(question, raw_candidates)
                rel_docs, rel_articles = extract_references_topn(ranked, manifest)
        except Exception as e:
            logger.error(f"[Q-{qid}] Lỗi: {e}")
            rel_docs, rel_articles = [], []

        results.append({
            "id": norm_id(qid),
            "question": question,
            "answer": answer_text,
            "relevant_docs": rel_docs,
            "relevant_articles": rel_articles,
        })

        if (i + 1) % args.flush_every == 0:
            save_results(results, args.output)  # ghi tạm phòng crash
        if (i + 1) % 50 == 0:
            el = time.time() - start
            logger.info(f"[Progress] {i + 1}/{len(questions)} | {el:.1f}s | {el / (i + 1):.2f}s/câu")

    save_results(results, args.output)
    el = time.time() - start
    logger.info(
        f"🎉 Xong {len(results)} câu trong {el:.1f}s "
        f"({el / max(1, len(results)):.2f}s/câu). → {args.output}"
    )


if __name__ == "__main__":
    main()
