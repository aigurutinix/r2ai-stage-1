"""
app/service.py

RagService — singleton nạp MỘT LẦN các thành phần nặng (corpus/BM25, embedding, reranker, Qdrant,
manifest, map URL) rồi phục vụ nhiều request của app demo. Tái sử dụng đúng các module đường thi:
HybridRetriever, LegalReranker, reference_extractor. Phần sinh câu trả lời gọi API (ApiAnswerGenerator).

Luồng 1 câu hỏi:
    retrieve() -> rerank(top_k=DEMO_RETRIEVE_POOL) -> build_citations() [trả về NGAY cho UI]
    -> ApiAnswerGenerator.stream_raw() [stream token]
"""
import os
import re
import json
import logging
from typing import List, Dict, Iterator, Tuple

from config.settings import Settings
from src.index_bm25 import BM25IndexBuilder
from src.hybrid_retriever import HybridRetriever
from src.reranker import LegalReranker
from src.reference_extractor import load_manifest, extract_references_topn
from src.api_answer_generator import ApiAnswerGenerator

logger = logging.getLogger("app.service")

# Khớp với scripts/build_url_map.py: số/năm/loại | số/loại | số-loại.
DOC_NUMBER_RE = re.compile(
    r"\d{1,4}/\d{4}/[A-Za-zĐđ0-9/\-]+"
    r"|\d{1,4}/[A-Za-zĐđ][A-Za-zĐđ0-9/\-]*"
    r"|\d{1,4}-[A-Za-zĐđ]{1,6}"
)


def _load_corpus() -> List[Dict]:
    """Giống load_corpus của fast_retrieval.py — dùng law_corpus_clean.json."""
    corpus_json = os.path.join(Settings.DATA_DIR, "law_corpus_clean.json")
    if os.path.exists(corpus_json):
        logger.info(f"[Corpus] Load từ {corpus_json}")
        with open(corpus_json, "r", encoding="utf-8") as f:
            return json.load(f)
    logger.warning("[Corpus] Không thấy law_corpus_clean.json — BM25 rỗng, chỉ Dense Search.")
    return []


def _build_embedding_fn():
    """Embedding query bằng ĐÚNG AITeamVN/Vietnamese_Embedding để khớp index Qdrant law_2026."""
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Embedding] Loading {Settings.EMBEDDING_MODEL} trên {device.upper()}...")
    model = SentenceTransformer(Settings.EMBEDDING_MODEL, device=device)
    logger.info("✅ Embedding model loaded.")

    def embed_fn(text: str) -> List[float]:
        return model.encode(text, normalize_embeddings=True).tolist()

    return embed_fn


def _load_url_map() -> Dict[str, str]:
    path = Settings.DOC_URL_MAP_PATH
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        logger.info(f"[URLMap] Load {len(m)} số hiệu → URL từ {path}")
        return m
    logger.warning(
        f"[URLMap] Không thấy {path} — citations sẽ không có link. "
        f"Chạy scripts/build_url_map.py để tạo."
    )
    return {}


def _norm_key(doc_number: str) -> str:
    """Chuẩn hóa số hiệu về cùng khóa với build_url_map.normalize_doc_number."""
    if not doc_number:
        return ""
    m = DOC_NUMBER_RE.search(doc_number)
    if not m:
        return ""
    return m.group(0).strip().upper().replace(" ", "")


class RagService:
    def __init__(self):
        logger.info("[RagService] Khởi tạo các thành phần nặng (một lần)...")
        bm25 = BM25IndexBuilder()
        if os.path.exists(Settings.BM25_INDEX_PATH):
            # Đã có index → nạp thẳng, KHÔNG cần nạp corpus 137MB (tiết kiệm RAM + cold-start trên CPU/HF Spaces).
            bm25.load()
        else:
            corpus = _load_corpus()
            if corpus:
                bm25.build(corpus).save()
            else:
                from rank_bm25 import BM25Okapi
                bm25.documents = []
                bm25.bm25 = BM25Okapi([[]])

        embed_fn = _build_embedding_fn()
        self.retriever = HybridRetriever(bm25_builder=bm25, embedding_fn=embed_fn, llm_pipeline=None)
        self.reranker = LegalReranker()
        self.manifest = load_manifest()
        self.url_map = _load_url_map()
        self.generator = ApiAnswerGenerator()
        logger.info("✅ [RagService] Sẵn sàng phục vụ.")

    # ----------------------------------------------------------------
    def retrieve_and_rerank(self, question: str) -> List[Dict]:
        """Trả về danh sách context đã rerank (điểm cao → thấp), giới hạn DEMO_RETRIEVE_POOL."""
        raw = self.retriever.retrieve(question)
        ranked = self.reranker.rerank(question, raw, top_k=Settings.DEMO_RETRIEVE_POOL)
        return ranked

    def build_citations(self, ranked: List[Dict]) -> Tuple[List[Dict], List[str], List[str]]:
        """
        Từ context đã rerank → danh sách thẻ citation cho UI + (relevant_docs, relevant_articles)
        theo đúng logic F2 top-N (extract_references_topn) để hiển thị nhất quán.

        Mỗi thẻ: { ten_van_ban, so_hieu, dieu, doan_trich, url, score }.
        """
        citations: List[Dict] = []
        seen = set()
        for doc in ranked:
            meta = doc.get("metadata", {})
            doc_number = (meta.get("doc_number") or meta.get("doc_id") or "").strip()
            if not doc_number:
                continue
            article = (meta.get("article_id") or "").strip()
            title = (meta.get("title") or "").strip()

            # Loại trùng theo (số hiệu, điều)
            dedup_key = f"{doc_number}|{article}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Tra URL + tên dự phòng từ map (giá trị có thể là dict {url,name} hoặc chuỗi url cũ).
            entry = self.url_map.get(_norm_key(doc_number))
            if isinstance(entry, dict):
                url, url_name = entry.get("url"), entry.get("name")
            else:
                url, url_name = entry, None

            # Ưu tiên tên: manifest (sạch, có dấu) → tên suy từ URL → title metadata → số hiệu.
            manifest_entry = self.manifest.get(doc_number)
            if isinstance(manifest_entry, dict) and manifest_entry.get("btc_standard_string"):
                ten_van_ban = manifest_entry["btc_standard_string"].split("|", 1)[-1]
            else:
                ten_van_ban = url_name or title or f"Văn bản {doc_number}"

            snippet = re.sub(r"\s+", " ", doc.get("text", "")).strip()
            if len(snippet) > Settings.DEMO_SNIPPET_CHARS:
                snippet = snippet[: Settings.DEMO_SNIPPET_CHARS].rstrip() + "…"

            citations.append({
                "ten_van_ban": ten_van_ban,
                "so_hieu": doc_number,
                "dieu": article,
                "doan_trich": snippet,
                "url": url,
                "score": round(float(doc.get("rerank_score", 0.0)), 4),
            })

        rel_docs, rel_articles = extract_references_topn(ranked, self.manifest)
        return citations, rel_docs, rel_articles

    def stream_answer(self, question: str, ranked: List[Dict]) -> Iterator[str]:
        """Stream các mảnh text thô của câu trả lời từ API LLM."""
        return self.generator.stream_raw(question, ranked)
