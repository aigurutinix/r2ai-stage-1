#!/usr/bin/env python3
"""RAG pipeline: retrieve chunks from Qdrant and answer R2AIStage1 questions."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bm25_retrieval import hybrid_retrieve_one, load_or_build_bm25_index
from qdrant_config import (
    add_qdrant_args,
    chunk_text,
    get_collection_name,
    get_embed_model_path,
    get_rerank_model_path,
    get_sparse_vector_name,
    init_qdrant_from_args,
    make_qdrant_client,
    query_dense,
    query_sparse_bm25,
)
from rerank_retrieval import (
    DEFAULT_PRIMARY_WEIGHT,
    DEFAULT_SUB_WEIGHT,
    dense_hits_to_chunks,
    load_reranker,
    rerank_chunks_hybrid,
)
from query_decompose import (
    DEFAULT_THRESHOLD_1,
    DEFAULT_THRESHOLD_2,
    batch_decompose_questions,
    load_embed_tokenizer,
    plan_queries,
    resolve_queries,
)
from subquery_loader import (
    SubquerySpec,
    load_subquery_index,
    rerank_primary_and_subs,
    retrieval_queries,
    subquery_map_from_index,
)

DEFAULT_QUESTIONS = ROOT / "test" / "R2AIStage1DATA.json"
DEFAULT_EMBED_MODEL = get_embed_model_path()
DEFAULT_LLM_MODEL = ROOT / "models" / "Vi-Qwen2-1.5B-RAG"
DEFAULT_RERANK_MODEL = get_rerank_model_path()
DEFAULT_OUTPUT = ROOT / "test" / "R2AIStage1_answers.json"
DEFAULT_RETRIEVED = ROOT / "test" / "R2AIStage1_retrieved.json"
DEFAULT_SUBQUERIES = ROOT / "test" / "R2AIStage1_subqueries (1).json"
DEFAULT_BM25_CACHE = ROOT / "output" / "bm25_corpus.pkl"

VI_QWEN_SYSTEM = (
    "Bạn là một trợ lí Tiệng Việt nhiệt tình và trung thực. "
    "Hãy luôn trả lời một cách hữu ích nhất có thể."
)
VI_QWEN_RAG_USER = """Chú ý các yêu cầu sau:
- Câu trả lời phải chính xác và đầy đủ nếu ngữ cảnh có câu trả lời.
- Chỉ sử dụng các thông tin có trong ngữ cảnh được cung cấp.
- Viết một đoạn văn liền mạch, trả lời trực tiếp câu hỏi, nêu đủ điều kiện, mức hỗ trợ, thời hạn, trách nhiệm nếu có.
- Không thêm tiêu đề, không thêm mục Kết luận/Phân tích, không liệt kê bullet.
Hãy trả lời câu hỏi dựa trên ngữ cảnh:
### Ngữ cảnh :
{context}

### Câu hỏi :
{question}

### Trả lời :"""

_ARTICLE_RE = re.compile(r"Điều\s+(\d+[a-zA-Z]?)", re.IGNORECASE)
_CODE_RE = re.compile(
    r"\b(\d{1,3}/\d{4}/(?:QH\d+|NĐ-CP|TT-[A-Z]+|QĐ-[A-Z]+|NQ-HĐND))\b",
    re.IGNORECASE,
)
_CODE_ALT_PATTERNS = (
    re.compile(r"\b(\d+/QĐ-[A-Z]+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+/VBHN-[A-Z]+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+/NQ-[A-Z]+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+/TT-[A-Z]+)\b", re.IGNORECASE),
)
_NON_DIEU_ARTICLE_LABELS = frozenset({"phụ lục", "phu luc", "chương", "mục"})


@dataclass
class FilterChunkStats:
    examined: int = 0
    accepted: int = 0
    rejected_no_code: int = 0
    rejected_no_title: int = 0
    rejected_bad_title: int = 0

    def record_rejection(self, reason: str) -> None:
        if reason == "no_code":
            self.rejected_no_code += 1
        elif reason == "no_title":
            self.rejected_no_title += 1
        elif reason == "bad_title":
            self.rejected_bad_title += 1

    @property
    def rejected(self) -> int:
        return self.examined - self.accepted

    def log_summary(self, prefix: str = "") -> None:
        if self.examined == 0:
            return
        pct = 100.0 * self.rejected / self.examined
        print(
            f"{prefix}filter_valid: {self.accepted}/{self.examined} kept "
            f"({pct:.1f}% rejected — "
            f"no_code={self.rejected_no_code}, "
            f"no_title={self.rejected_no_title}, "
            f"bad_title={self.rejected_bad_title})"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--retrieved-cache", type=Path, default=DEFAULT_RETRIEVED)
    p.add_argument("--embed-model", type=Path, default=DEFAULT_EMBED_MODEL)
    p.add_argument("--llm-model", type=Path, default=DEFAULT_LLM_MODEL)
    add_qdrant_args(p)
    p.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Final chunks after rerank",
    )
    p.add_argument(
        "--llm-top-k",
        type=int,
        default=4,
        help="Chunks for LLM context and relevant_articles submission",
    )
    p.add_argument("--retrieve-batch", type=int, default=4)
    p.add_argument("--gen-batch", type=int, default=2, help="LLM generation batch size")
    p.add_argument("--max-context-chars", type=int, default=3500)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-id", type=int, default=1)
    p.add_argument("--skip-retrieve", action="store_true")
    p.add_argument("--skip-answered", action="store_true", help="Skip question ids already in output file")
    p.add_argument("--retrieve-only", action="store_true")
    p.add_argument(
        "--citations-only",
        action="store_true",
        help="Chỉ retrieve và xuất id, relevant_docs, relevant_articles (không dùng LLM)",
    )
    p.add_argument("--device-embed", default="cuda")
    p.add_argument("--device-llm", default="cuda")
    p.add_argument(
        "--include-chunks",
        action="store_true",
        help="Include retrieved_chunks in output JSON (debug)",
    )
    p.add_argument(
        "--no-bm25",
        action="store_true",
        help="Disable hybrid retrieval (dense + BM25 + weightRRF); default is hybrid on",
    )
    p.add_argument(
        "--bm25-cache",
        type=Path,
        default=DEFAULT_BM25_CACHE,
        help="Cache file for BM25 corpus index",
    )
    p.add_argument(
        "--retrieve-pool",
        type=int,
        default=10,
        help="Top chunks per method before RRF (dense=10, BM25=10)",
    )
    p.add_argument(
        "--rrf-top-k",
        type=int,
        default=15,
        help="Chunks kept after weightRRF fusion (0.4 dense + 0.6 BM25), before rerank",
    )
    p.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="weightRRF constant k for hybrid score fusion (dense=0.4, BM25=0.6)",
    )
    p.add_argument(
        "--rerank-model",
        type=Path,
        default=DEFAULT_RERANK_MODEL,
        help="Cross-encoder reranker model path",
    )
    p.add_argument("--device-rerank", default="cuda")
    p.add_argument("--rerank-batch", type=int, default=8)
    p.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking",
    )
    p.add_argument(
        "--no-subquery",
        action="store_true",
        help="Disable sub-query decomposition for long questions",
    )
    p.add_argument(
        "--use-llm-subquery",
        action="store_true",
        help="Use LLM for sub-query decomposition (legacy; default is rule-based)",
    )
    p.add_argument(
        "--subquery-cache",
        type=Path,
        default=DEFAULT_SUBQUERIES,
        help="Precomputed sub-query JSON (R2AIStage1_subqueries (1).json)",
    )
    p.add_argument(
        "--hybrid-rerank",
        action="store_true",
        help="Rerank with queries[0]*primary_weight + mean(sub)*sub_weight",
    )
    p.add_argument(
        "--primary-weight",
        type=float,
        default=DEFAULT_PRIMARY_WEIGHT,
        help="Weight for first query in hybrid rerank (default: 0.7)",
    )
    p.add_argument(
        "--sub-weight",
        type=float,
        default=DEFAULT_SUB_WEIGHT,
        help="Weight for mean of sub-query scores in hybrid rerank (default: 0.3)",
    )
    p.add_argument(
        "--token-threshold-1",
        type=int,
        default=DEFAULT_THRESHOLD_1,
        help="Token count below this uses 1 query (default: 30)",
    )
    p.add_argument(
        "--token-threshold-2",
        type=int,
        default=DEFAULT_THRESHOLD_2,
        help="Token count below this uses 2 sub-queries; >= uses 3 (default: 58)",
    )
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization for LLM (uses fp16 on CUDA)",
    )
    return p.parse_args()


def _looks_bad_title(title: str) -> bool:
    if not title:
        return True
    stripped = title.strip()
    if not stripped or all(c in "-_ " for c in stripped):
        return True
    lowered = stripped.lower()
    return lowered.startswith("căn cứ") or lowered.startswith("theo ")


def _title_from_file_name(file_name: str, law_code: str) -> str:
    if not file_name:
        return ""
    stem = Path(file_name).stem
    code_token = law_code.replace("/", "_") if law_code else ""
    if code_token and code_token in stem:
        rest = stem.split(code_token, 1)[-1].lstrip("_")
        if rest:
            return rest.replace("_", " ").strip()
    parts = stem.split("_", 2)
    if len(parts) >= 3:
        return parts[2].replace("_", " ").strip()
    return stem.replace("_", " ").strip()


def _title_from_text(text: str, law_code: str) -> str:
    if not text:
        return ""
    head = text[:500]
    if law_code:
        pattern = re.compile(
            rf"{re.escape(law_code)}\s+(.+?)(?:\n|$)",
            re.IGNORECASE,
        )
        match = pattern.search(head)
        if match:
            return match.group(1).strip(" -")
    first_line = head.split("\n", 1)[0].strip()
    if law_code and law_code in first_line:
        return first_line.split(law_code, 1)[-1].strip(" -")
    return first_line


def _code_from_text(text: str) -> str:
    if not text:
        return ""
    head = text[:600]
    match = _CODE_RE.search(head)
    if match:
        return match.group(1)
    for pattern in _CODE_ALT_PATTERNS:
        match = pattern.search(head)
        if match:
            return match.group(1)
    return ""


def resolve_law_code(chunk: dict) -> str:
    for key in ("document_number", "law_code"):
        code = str(chunk.get(key) or "").strip()
        if code:
            return code
    return _code_from_text(chunk_text(chunk))


def _title_from_law_type_prefix(text: str, law_code: str) -> str:
    if not text:
        return ""
    head = text[:600]
    if law_code:
        pattern = re.compile(
            rf"{re.escape(law_code)}\s+((?:Luật|Bộ luật|Nghị định|Thông tư|Quyết định|Nghị quyết|Văn bản hợp nhất)[^\n.]{{3,200}})",
            re.IGNORECASE,
        )
        match = pattern.search(head)
        if match:
            return match.group(1).strip(" .")
    prefix_match = re.match(
        r"^((?:Luật|Bộ luật|Nghị định|Thông tư|Quyết định|Nghị quyết|Văn bản hợp nhất)[^\n]{3,200})",
        head.strip(),
        re.IGNORECASE,
    )
    if prefix_match:
        return prefix_match.group(1).strip(" .")
    return ""


def resolve_law_title(chunk: dict) -> str:
    law_code = resolve_law_code(chunk)
    for key in ("document_title", "law_title"):
        title = str(chunk.get(key) or "").strip()
        if not _looks_bad_title(title):
            return title
    title = _title_from_file_name(chunk.get("file_name") or "", law_code)
    if not _looks_bad_title(title):
        return title
    text = chunk_text(chunk)
    title = _title_from_text(text, law_code)
    if not _looks_bad_title(title):
        return title
    title = _title_from_law_type_prefix(text, law_code)
    if not _looks_bad_title(title):
        return title
    return title


def chunk_reject_reason(chunk: dict) -> str | None:
    law_code = resolve_law_code(chunk)
    if not law_code:
        return "no_code"
    law_title = resolve_law_title(chunk)
    if not law_title:
        return "no_title"
    if _looks_bad_title(law_title):
        return "bad_title"
    return None


def is_valid_chunk(chunk: dict) -> bool:
    return chunk_reject_reason(chunk) is None


def filter_valid_chunks(
    chunks: list[dict],
    limit: int | None = None,
    stats: FilterChunkStats | None = None,
) -> list[dict]:
    valid: list[dict] = []
    for chunk in chunks:
        reason = chunk_reject_reason(chunk)
        if stats is not None:
            stats.examined += 1
            if reason:
                stats.record_rejection(reason)
            else:
                stats.accepted += 1
        if reason is None:
            valid.append(chunk)
    if limit is not None:
        valid = valid[:limit]
    for rank, chunk in enumerate(valid, start=1):
        chunk["rank"] = rank
    return valid


def _normalize_article_label(raw: str) -> str | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    lower = raw.lower()
    if lower.startswith("điều "):
        return raw
    if lower in _NON_DIEU_ARTICLE_LABELS:
        return raw[0].upper() + raw[1:] if raw else raw
    return f"Điều {raw}"


def _article_parts(raw: str) -> list[str]:
    raw = str(raw or "").strip()
    if not raw:
        return []
    if raw.lower().startswith("điều "):
        raw = raw[5:].strip()
    if "_" in raw:
        return [part for part in raw.split("_") if part.strip()]
    return [raw]


def expand_article_labels(chunk: dict) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    def add_raw(raw: str) -> None:
        for part in _article_parts(raw):
            label = _normalize_article_label(part)
            if label and label not in seen:
                seen.add(label)
                labels.append(label)

    article = str(chunk.get("article_no") or chunk.get("node_label") or chunk.get("article_number") or "").strip()
    if article:
        add_raw(article)

    list_article = chunk.get("source_article_no_candidates") or chunk.get("list_article")
    if list_article:
        items = list_article if isinstance(list_article, list) else [list_article]
        for item in items:
            if item:
                add_raw(str(item))

    if not labels:
        text = chunk_text(chunk)
        for match in _ARTICLE_RE.finditer(text[:800]):
            add_raw(match.group(1))

    return labels


def resolve_article_label(chunk: dict) -> str | None:
    labels = expand_article_labels(chunk)
    return labels[0] if labels else None


def chunk_to_doc_ref(chunk: dict) -> str | None:
    if not is_valid_chunk(chunk):
        return None
    law_code = resolve_law_code(chunk)
    law_title = resolve_law_title(chunk)
    return f"{law_code}|{law_title}"


def chunk_to_article_refs(chunk: dict) -> list[str]:
    if not is_valid_chunk(chunk):
        return []
    law_code = resolve_law_code(chunk)
    law_title = resolve_law_title(chunk)
    return [f"{law_code}|{law_title}|{article}" for article in expand_article_labels(chunk)]


def chunk_to_article_ref(chunk: dict) -> str | None:
    refs = chunk_to_article_refs(chunk)
    return refs[0] if refs else None


def extract_citations(chunks: list[dict]) -> tuple[list[str], list[str]]:
    relevant_docs: list[str] = []
    relevant_articles: list[str] = []
    seen_docs: set[str] = set()
    seen_articles: set[str] = set()
    for chunk in filter_valid_chunks(chunks):
        doc_ref = chunk_to_doc_ref(chunk)
        if doc_ref and doc_ref not in seen_docs:
            seen_docs.add(doc_ref)
            relevant_docs.append(doc_ref)
        for article_ref in chunk_to_article_refs(chunk):
            if article_ref not in seen_articles:
                seen_articles.add(article_ref)
                relevant_articles.append(article_ref)
    return relevant_docs, relevant_articles


def build_citation_results(items: list[dict], llm_top_k: int) -> list[dict]:
    results: list[dict] = []
    for item in items:
        relevant_docs, relevant_articles = extract_citations(item["chunks"][:llm_top_k])
        results.append({
            "id": item["id"],
            "question": item.get("question", ""),
            "answer": "",
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
        })
    return results


def load_questions(path: Path, start_id: int, limit: int | None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = [q for q in data if q.get("id", 0) >= start_id]
    if limit:
        items = items[:limit]
    return items


def load_subquery_cache(path: Path | None) -> dict[int, list[str]]:
    """Backward-compatible: id -> retrieval query list."""
    return subquery_map_from_index(load_subquery_index(path))


class RetrievalEngine:
    """Embedding + BM25 + rerank retrieval; supports GPU offload between batches."""

    def __init__(
        self,
        *,
        embed_model_path: Path,
        qdrant_url: str | None,
        collection: str | None,
        top_k: int,
        device: str,
        qdrant_api_key: str | None = None,
        vector_name: str | None = None,
        sparse_vector_name: str | None = None,
        use_bm25: bool = False,
        bm25_cache: Path | None = None,
        retrieve_pool: int = 50,
        rrf_top_k: int = 50,
        rrf_k: int = 60,
        use_rerank: bool = True,
        rerank_model_path: Path | None = None,
        device_rerank: str = "cuda",
        rerank_batch: int = 32,
        enable_subquery: bool = True,
        sub_query_map: dict[int, list[str]] | None = None,
        subquery_index: dict[int, SubquerySpec] | None = None,
        token_threshold_1: int = DEFAULT_THRESHOLD_1,
        token_threshold_2: int = DEFAULT_THRESHOLD_2,
        embed_tokenizer=None,
        hybrid_rerank: bool = False,
        primary_weight: float = DEFAULT_PRIMARY_WEIGHT,
        sub_weight: float = DEFAULT_SUB_WEIGHT,
    ) -> None:
        self.embed_model_path = embed_model_path
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.vector_name = vector_name
        self.sparse_vector_name = get_sparse_vector_name(sparse_vector_name)
        self.use_qdrant_bm25 = bool(use_bm25 and self.sparse_vector_name)
        self.collection = get_collection_name(collection)
        self.top_k = top_k
        self.device = device
        self.use_bm25 = use_bm25
        self.bm25_cache = bm25_cache
        self.pool_size = retrieve_pool
        self.rrf_top_k = rrf_top_k
        self.rrf_k = rrf_k
        self.use_rerank = use_rerank
        self.rerank_model_path = rerank_model_path
        self.device_rerank = device_rerank
        self.rerank_batch = rerank_batch
        self.enable_subquery = enable_subquery
        self.sub_query_map = sub_query_map or {}
        self.subquery_index = subquery_index or {}
        self.token_threshold_1 = token_threshold_1
        self.token_threshold_2 = token_threshold_2
        self.embed_tokenizer = embed_tokenizer
        self.hybrid_rerank = hybrid_rerank
        self.primary_weight = primary_weight
        self.sub_weight = sub_weight

        self.fusion_top_k = rrf_top_k if (use_bm25 and use_rerank) else top_k
        self.ann_limit = retrieve_pool if (use_bm25 or use_rerank) else top_k

        self.client: QdrantClient | None = None
        self.embed_model: SentenceTransformer | None = None
        self.reranker = None
        self.bm25 = None
        self.corpus = None
        self._on_gpu = False
        self.filter_stats = FilterChunkStats()

    def load(self) -> None:
        if self.client is not None:
            return
        self.client = make_qdrant_client(self.qdrant_url, self.qdrant_api_key)
        if self.use_bm25:
            if self.use_qdrant_bm25:
                bm25_label = f"Qdrant BM25 ({self.sparse_vector_name})"
            else:
                self.bm25, self.corpus = load_or_build_bm25_index(
                    self.client, self.collection, self.bm25_cache
                )
                bm25_label = "local BM25"
            if self.use_rerank:
                print(
                    f"  Hybrid retrieve: dense + {bm25_label} + rerank "
                    f"(pool={self.pool_size}, rrf_top_k={self.rrf_top_k}, "
                    f"top_k={self.top_k}, weightRRF k={self.rrf_k}, "
                    f"dense=0.4, bm25=0.6)"
                )
            else:
                print(
                    f"  Hybrid retrieve: dense + {bm25_label} "
                    f"(pool={self.pool_size}, top_k={self.top_k}, "
                    f"weightRRF k={self.rrf_k}, dense=0.4, bm25=0.6)"
                )
        elif self.use_rerank:
            print(f"  Dense retrieve + rerank (pool={self.pool_size}, top_k={self.top_k})")

        self.ensure_on_gpu()

    def ensure_on_gpu(self) -> None:
        if self.embed_model is None:
            self.embed_model = SentenceTransformer(str(self.embed_model_path), device=self.device)
            self.embed_model.max_seq_length = 2048
        elif not self._on_gpu and self.device == "cuda":
            self.embed_model.to(self.device)

        if self.use_rerank and self.reranker is None:
            self.reranker = load_reranker(
                self.rerank_model_path or DEFAULT_RERANK_MODEL,
                self.device_rerank,
            )
        elif (
            self.use_rerank
            and self.reranker is not None
            and not self._on_gpu
            and self.device_rerank == "cuda"
        ):
            self.reranker.model.to(self.device_rerank)

        self._on_gpu = self.device == "cuda" or self.device_rerank == "cuda"

    def offload_gpu(self) -> None:
        if self.embed_model is not None and self.device == "cuda":
            self.embed_model.to("cpu")
        if self.reranker is not None and self.device_rerank == "cuda":
            self.reranker.model.to("cpu")
        self._on_gpu = False
        if self.device == "cuda" or self.device_rerank == "cuda":
            torch.cuda.empty_cache()

    def unload(self) -> None:
        self.offload_gpu()
        self.embed_model = None
        self.reranker = None
        self.client = None
        self.bm25 = None
        self.corpus = None

    def _retrieval_query(self, qid: int, question: str) -> str:
        """Single query for 10 dense + 10 BM25 + RRF 15 (sub-queries only at rerank)."""
        spec = self.subquery_index.get(qid)
        if spec is not None and spec.original_question:
            return spec.original_question
        return question

    def _rerank_queries(self, qid: int, question: str) -> tuple[str, list[str]]:
        spec = self.subquery_index.get(qid)
        if spec is not None:
            return rerank_primary_and_subs(spec, question)
        if self.enable_subquery and qid in self.sub_query_map:
            subs = self.sub_query_map[qid]
            if len(subs) > 1:
                return question, list(subs)
        if self.enable_subquery:
            queries = resolve_queries(
                question,
                self.embed_tokenizer,
                threshold_1=self.token_threshold_1,
                threshold_2=self.token_threshold_2,
            )
            if len(queries) > 1:
                return question, list(queries)
        return question, []

    def retrieve_one(
        self,
        q_item: dict,
        *,
        query_vectors: dict[str, list[float]] | None = None,
    ) -> dict:
        assert self.client is not None and self.embed_model is not None

        question = q_item["question"]
        qid = q_item["id"]
        query = self._retrieval_query(qid, question)

        if query_vectors and query in query_vectors:
            vec_list = query_vectors[query]
        else:
            encoded = self.embed_model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )[0]
            vec_list = encoded.tolist()

        hits = query_dense(
            self.client,
            self.collection,
            vec_list,
            limit=self.ann_limit,
            vector_name=self.vector_name,
        )
        if self.use_bm25:
            sparse_hits = None
            if self.use_qdrant_bm25:
                sparse_res = query_sparse_bm25(
                    self.client,
                    self.collection,
                    query,
                    limit=self.pool_size,
                    sparse_vector_name=self.sparse_vector_name,
                )
                sparse_hits = sparse_res.points
            chunks = hybrid_retrieve_one(
                query,
                dense_hits=hits.points,
                bm25=self.bm25,
                corpus=self.corpus,
                sparse_hits=sparse_hits,
                top_k=self.fusion_top_k,
                pool_size=self.pool_size,
                rrf_k=self.rrf_k,
            )
        else:
            chunks = dense_hits_to_chunks(hits.points)[: self.fusion_top_k]

        if self.use_rerank and self.reranker is not None:
            primary, subs = self._rerank_queries(qid, question)
            chunks = rerank_chunks_hybrid(
                primary,
                subs,
                chunks,
                self.reranker,
                top_k=self.top_k,
                batch_size=self.rerank_batch,
                primary_weight=self.primary_weight,
                sub_weight=self.sub_weight,
            )
        chunks = filter_valid_chunks(
            chunks, limit=self.top_k, stats=self.filter_stats
        )

        spec = self.subquery_index.get(qid)
        _, subs = self._rerank_queries(qid, question)
        sub_for_output = subs or None
        return {
            "id": qid,
            "question": question,
            "chunks": chunks,
            "sub_queries": sub_for_output,
        }

    def retrieve_batch(self, questions: list[dict]) -> list[dict]:
        if not questions:
            return []
        self.load()
        self.ensure_on_gpu()
        assert self.client is not None and self.embed_model is not None

        all_queries: list[str] = []
        query_to_encode: set[str] = set()
        for q_item in questions:
            qid = q_item["id"]
            query = self._retrieval_query(qid, q_item["question"])
            all_queries.append(query)
            query_to_encode.add(query)

        vectors_map: dict[str, list[float]] = {}
        if query_to_encode:
            unique_queries = list(query_to_encode)
            vectors = self.embed_model.encode(
                unique_queries, normalize_embeddings=True, show_progress_bar=False
            )
            for query, vec in zip(unique_queries, vectors):
                vectors_map[query] = vec.tolist()

        results: list[dict] = []
        for q_item in questions:
            query = self._retrieval_query(q_item["id"], q_item["question"])
            per_query_vectors = {query: vectors_map[query]} if query in vectors_map else None
            result = self.retrieve_one(q_item, query_vectors=per_query_vectors)
            results.append(result)
        self.filter_stats.log_summary("  ")
        return results


def build_context(chunks: list[dict], max_chars: int, max_chunks: int | None = None) -> str:
    chunks = filter_valid_chunks(chunks, limit=max_chunks)
    parts = []
    total = 0
    for c in chunks:
        law_code = resolve_law_code(c)
        law_title = resolve_law_title(c)
        article = resolve_article_label(c) or ""
        header = f"[{law_code}|{law_title}|{article}]".rstrip("|")
        block = f"{header}\n{chunk_text(c)}"
        if total + len(block) > max_chars:
            remain = max_chars - total
            if remain > 200:
                parts.append(block[:remain])
            break
        parts.append(block)
        total += len(block) + 4
    return "\n\n---\n\n".join(parts)


def apply_chat_template_safe(tokenizer, messages: list[dict], **kwargs) -> str:
    kwargs.setdefault("tokenize", False)
    kwargs.setdefault("add_generation_prompt", True)
    try:
        return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    user = VI_QWEN_RAG_USER.format(context=context, question=question)
    return [{"role": "system", "content": VI_QWEN_SYSTEM}, {"role": "user", "content": user}]


def build_prompt(tokenizer, question: str, context: str) -> str:
    messages = build_messages(question, context)
    return apply_chat_template_safe(tokenizer, messages)


def clean_answer(text: str) -> str:
    text = text.strip()
    think_close = "</" + "redacted_thinking>"
    think_open = "<" + "redacted_thinking>"
    if think_close in text:
        text = text.split(think_close, 1)[-1].strip()
    text = re.sub(rf"{re.escape(think_open)}.*?{re.escape(think_close)}", "", text, flags=re.DOTALL).strip()
    for marker in ("Câu hỏi:", "### Câu hỏi", "assistant", "Ngữ cảnh pháp luật:", "### Ngữ cảnh"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    text = re.sub(r"\s+", " ", text).strip()
    lines = [line.strip() for line in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if line in {"Có", "Không"} and cleaned and cleaned[-1] == line:
            continue
        cleaned.append(line)
    return " ".join(cleaned).strip()


def generate_batch_answers(
    model,
    tokenizer,
    batch_items: list[dict],
    *,
    max_context_chars: int,
    max_new_tokens: int,
    eos_ids: list[int],
    llm_top_k: int,
) -> list[dict]:
    prompts = [
        build_prompt(
            tokenizer,
            item["question"],
            build_context(item["chunks"], max_context_chars, max_chunks=llm_top_k),
        )
        for item in batch_items
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=6144,
    ).to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    results: list[dict] = []
    for item, seq in zip(batch_items, generated):
        answer = clean_answer(
            tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        )
        relevant_docs, relevant_articles = extract_citations(item["chunks"][:llm_top_k])
        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
        })
    return results


@dataclass
class AdaptiveGenBatch:
    """Gen batch size; tự giảm một nửa khi OOM (8→4→2→1)."""

    size: int

    def try_reduce(self) -> bool:
        if self.size <= 1:
            return False
        new_size = max(1, self.size // 2)
        print(f"  OOM gen_batch {self.size} → giảm xuống {new_size}")
        self.size = new_size
        return True


def generate_single_answer_safe(
    model,
    tokenizer,
    item: dict,
    *,
    max_context_chars: int,
    max_new_tokens: int,
    eos_ids: list[int],
    llm_top_k: int,
    device: str,
) -> dict:
    batch_items = [item]
    try:
        return generate_batch_answers(
            model,
            tokenizer,
            batch_items,
            max_context_chars=max_context_chars,
            max_new_tokens=max_new_tokens,
            eos_ids=eos_ids,
            llm_top_k=llm_top_k,
        )[0]
    except torch.cuda.OutOfMemoryError:
        if device == "cuda":
            torch.cuda.empty_cache()
        if llm_top_k > 5:
            reduced_k = max(5, llm_top_k // 2)
            reduced_chars = max(1500, max_context_chars // 2)
            print(
                f"  OOM câu id={item['id']} → "
                f"llm_top_k={reduced_k}, max_context_chars={reduced_chars}"
            )
            return generate_single_answer_safe(
                model,
                tokenizer,
                item,
                max_context_chars=reduced_chars,
                max_new_tokens=max_new_tokens,
                eos_ids=eos_ids,
                llm_top_k=reduced_k,
                device=device,
            )
        if max_new_tokens > 128:
            reduced_tokens = max(128, max_new_tokens // 2)
            print(f"  OOM câu id={item['id']} → max_new_tokens={reduced_tokens}")
            return generate_single_answer_safe(
                model,
                tokenizer,
                item,
                max_context_chars=max_context_chars,
                max_new_tokens=reduced_tokens,
                eos_ids=eos_ids,
                llm_top_k=llm_top_k,
                device=device,
            )
        raise


def generate_items_adaptive(
    model,
    tokenizer,
    items: list[dict],
    *,
    gen_batch: AdaptiveGenBatch,
    max_context_chars: int,
    max_new_tokens: int,
    eos_ids: list[int],
    llm_top_k: int,
    device: str,
) -> list[dict]:
    """Generate answers; giảm gen_batch khi OOM và giữ size mới cho các batch sau."""
    results: list[dict] = []
    idx = 0
    while idx < len(items):
        chunk = items[idx : idx + gen_batch.size]
        try:
            batch_results = generate_batch_answers(
                model,
                tokenizer,
                chunk,
                max_context_chars=max_context_chars,
                max_new_tokens=max_new_tokens,
                eos_ids=eos_ids,
                llm_top_k=llm_top_k,
            )
            results.extend(batch_results)
            idx += len(chunk)
        except torch.cuda.OutOfMemoryError:
            if device == "cuda":
                torch.cuda.empty_cache()
            if len(chunk) > 1 and gen_batch.try_reduce():
                continue
            for item in chunk:
                results.append(
                    generate_single_answer_safe(
                        model,
                        tokenizer,
                        item,
                        max_context_chars=max_context_chars,
                        max_new_tokens=max_new_tokens,
                        eos_ids=eos_ids,
                        llm_top_k=llm_top_k,
                        device=device,
                    )
                )
            idx += len(chunk)
    return results


def load_llm(llm_model_path: Path, device: str, *, load_in_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(str(llm_model_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict = {}
    use_4bit = device == "cuda" and load_in_4bit
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["device_map"] = "auto"
        except ImportError:
            use_4bit = False

    if device == "cuda" and not use_4bit:
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"
    elif device != "cuda":
        model_kwargs["torch_dtype"] = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(llm_model_path),
            **model_kwargs,
        )
        if use_4bit:
            print("  LLM: 4-bit quantization enabled")
        elif device == "cuda":
            print("  LLM: bfloat16 on CUDA")
    except (RuntimeError, OSError) as exc:
        if not use_4bit:
            raise
        print(f"  LLM: 4-bit failed ({exc}), falling back to bfloat16")
        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        model = AutoModelForCausalLM.from_pretrained(
            str(llm_model_path),
            **model_kwargs,
        )

    if device != "cuda" or "device_map" not in model_kwargs:
        model = model.to(device)

    eos_ids = [tokenizer.eos_token_id]
    for token in ("", "<|endoftext|>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id and tid not in eos_ids:
            eos_ids.append(tid)
    return model, tokenizer, eos_ids


def offload_llm(model, device: str) -> None:
    """Move LLM off GPU without reloading from disk."""
    if device == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()


def unload_llm(model, device: str) -> None:
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def ensure_llm_on_device(model, device: str) -> None:
    if device == "cuda" and next(model.parameters()).device.type != "cuda":
        model.to(device)


def append_retrieved_cache(cache_path: Path, new_items: list[dict]) -> None:
    if not new_items:
        return
    save_json(cache_path, merge_existing(cache_path, new_items))


def _needs_decompose(questions: list[dict], embed_tokenizer, args) -> bool:
    """Legacy LLM path — only when --use-llm-subquery is set."""
    if args.no_subquery or not getattr(args, "use_llm_subquery", False):
        return False
    for q in questions:
        n, preset = plan_queries(
            q["question"],
            embed_tokenizer,
            threshold_1=args.token_threshold_1,
            threshold_2=args.token_threshold_2,
            use_token_budget=True,
        )
        if preset is None:
            return True
    return False


def decompose_sub_queries(
    questions: list[dict],
    *,
    args: argparse.Namespace,
    embed_model_path: Path,
) -> dict[int, list[str]]:
    embed_tokenizer = load_embed_tokenizer(embed_model_path)
    use_llm = getattr(args, "use_llm_subquery", False)

    if use_llm and _needs_decompose(questions, embed_tokenizer, args):
        print("=== Decompose sub-queries (LLM) ===")
        load_in_4bit = not args.no_4bit
        model, tokenizer, eos_ids = load_llm(args.llm_model, args.device_llm, load_in_4bit=load_in_4bit)
        try:
            sub_map = batch_decompose_questions(
                questions,
                model,
                tokenizer,
                eos_ids,
                embed_tokenizer,
                threshold_1=args.token_threshold_1,
                threshold_2=args.token_threshold_2,
                use_token_budget=True,
                use_llm=True,
            )
        finally:
            unload_llm(model, args.device_llm)
        return sub_map

    sub_map: dict[int, list[str]] = {}
    for q in questions:
        queries = resolve_queries(
            q["question"],
            embed_tokenizer,
            threshold_1=args.token_threshold_1,
            threshold_2=args.token_threshold_2,
        )
        if len(queries) > 1:
            sub_map[q["id"]] = queries
    return sub_map


def make_retrieval_engine(
    args: argparse.Namespace,
    *,
    sub_query_map: dict[int, list[str]] | None = None,
    subquery_index: dict[int, SubquerySpec] | None = None,
) -> RetrievalEngine:
    embed_tokenizer = None
    if not args.no_subquery and not subquery_index:
        embed_tokenizer = load_embed_tokenizer(args.embed_model)

    if subquery_index is None:
        subquery_index = load_subquery_index(args.subquery_cache)
    if sub_query_map is None and subquery_index:
        sub_query_map = subquery_map_from_index(subquery_index)

    return RetrievalEngine(
        embed_model_path=args.embed_model,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        vector_name=args.vector_name,
        sparse_vector_name=getattr(args, "sparse_vector_name", None),
        collection=args.collection,
        top_k=args.top_k,
        device=args.device_embed,
        use_bm25=not args.no_bm25,
        bm25_cache=args.bm25_cache,
        retrieve_pool=args.retrieve_pool,
        rrf_top_k=args.rrf_top_k,
        rrf_k=args.rrf_k,
        use_rerank=not args.no_rerank,
        rerank_model_path=args.rerank_model,
        device_rerank=args.device_rerank,
        rerank_batch=args.rerank_batch,
        enable_subquery=not args.no_subquery,
        sub_query_map=sub_query_map,
        subquery_index=subquery_index,
        token_threshold_1=args.token_threshold_1,
        token_threshold_2=args.token_threshold_2,
        embed_tokenizer=embed_tokenizer,
        hybrid_rerank=True,
        primary_weight=args.primary_weight,
        sub_weight=args.sub_weight,
    )


def resolve_sub_query_map(
    questions: list[dict],
    *,
    args: argparse.Namespace,
) -> dict[int, list[str]]:
    index = load_subquery_index(args.subquery_cache)
    if index:
        return {
            q["id"]: retrieval_queries(index[q["id"]], q["question"])
            for q in questions
            if q["id"] in index
        }
    if args.no_subquery:
        return {}
    return decompose_sub_queries(
        questions,
        args=args,
        embed_model_path=args.embed_model,
    )


def run_citations_pipeline(
    questions: list[dict],
    *,
    args: argparse.Namespace,
) -> int:
    """Retrieve chunks and export citations only (no LLM)."""
    total = len(questions)
    pipeline_batch = args.retrieve_batch
    engine = make_retrieval_engine(args)

    citation_count = 0
    print(
        f"=== Citations only: retrieve + extract (batch={pipeline_batch}, "
        f"llm_top_k={args.llm_top_k}, hybrid={not args.no_bm25}) ==="
    )
    if (
        args.output is not None
        and not args.skip_answered
        and args.start_id <= 1
        and args.limit is None
    ):
        save_json(args.output, [])
    t0 = time.time()

    try:
        for start in range(0, total, pipeline_batch):
            batch_qs = questions[start : start + pipeline_batch]
            if not args.no_subquery and not engine.subquery_index:
                engine.sub_query_map = resolve_sub_query_map(
                    batch_qs,
                    args=args,
                )
            retrieved_batch = engine.retrieve_batch(batch_qs)
            done_retrieve = min(start + pipeline_batch, total)
            print(f"  Retrieved {done_retrieve}/{total}")

            append_retrieved_cache(args.retrieved_cache, retrieved_batch)

            batch_results = build_citation_results(retrieved_batch, args.llm_top_k)
            citation_count += len(batch_results)

            if args.output is not None and batch_results:
                save_json(args.output, merge_existing(args.output, batch_results))

            done = min(start + pipeline_batch, total)
            print(f"  Done {done}/{total} (retrieve + citations)")
            del retrieved_batch
    finally:
        engine.filter_stats.log_summary()
        engine.unload()

    print(f"Citations xong: {citation_count} câu ({time.time() - t0:.1f}s)")
    return citation_count


def run_interleaved_pipeline(
    questions: list[dict],
    *,
    args: argparse.Namespace,
) -> int:
    """Retrieve and generate per batch; only one pipeline batch of chunks in RAM."""
    total = len(questions)
    pipeline_batch = args.retrieve_batch
    engine = make_retrieval_engine(args)
    load_in_4bit = not args.no_4bit

    llm_model = None
    tokenizer = None
    eos_ids: list[int] = []
    answer_count = 0
    gen_batch = AdaptiveGenBatch(args.gen_batch)

    print(
        f"=== Pipeline: retrieve + generate (batch={pipeline_batch}, "
        f"gen_batch={gen_batch.size}, hybrid={not args.no_bm25}) ==="
    )
    if args.output is not None and not args.skip_answered:
        save_json(args.output, [])
    t0 = time.time()

    try:
        for start in range(0, total, pipeline_batch):
            batch_qs = questions[start : start + pipeline_batch]
            if not args.no_subquery and not engine.subquery_index:
                engine.sub_query_map = resolve_sub_query_map(
                    batch_qs,
                    args=args,
                )
            retrieved_batch = engine.retrieve_batch(batch_qs)
            done_retrieve = min(start + pipeline_batch, total)
            print(f"  Retrieved {done_retrieve}/{total}")

            append_retrieved_cache(args.retrieved_cache, retrieved_batch)

            engine.offload_gpu()
            if llm_model is None:
                llm_model, tokenizer, eos_ids = load_llm(
                    args.llm_model, args.device_llm, load_in_4bit=load_in_4bit
                )
            else:
                ensure_llm_on_device(llm_model, args.device_llm)

            gen_start = 0
            while gen_start < len(retrieved_batch):
                gen_items = retrieved_batch[gen_start : gen_start + gen_batch.size]
                batch_results = generate_items_adaptive(
                    llm_model,
                    tokenizer,
                    gen_items,
                    gen_batch=gen_batch,
                    max_context_chars=args.max_context_chars,
                    max_new_tokens=args.max_new_tokens,
                    eos_ids=eos_ids,
                    llm_top_k=args.llm_top_k,
                    device=args.device_llm,
                )
                if args.device_llm == "cuda":
                    torch.cuda.empty_cache()

                for result in batch_results:
                    if args.include_chunks:
                        chunks = next(
                            x["chunks"] for x in gen_items if x["id"] == result["id"]
                        )
                        result["retrieved_chunks"] = chunks
                    answer_count += 1

                if args.output is not None and batch_results:
                    save_json(args.output, merge_existing(args.output, batch_results))
                gen_start += len(gen_items)

            done = min(start + pipeline_batch, total)
            print(f"  Done {done}/{total} (retrieve + generate)")
            del retrieved_batch

            if llm_model is not None:
                offload_llm(llm_model, args.device_llm)
    finally:
        engine.filter_stats.log_summary()
        engine.unload()
        if llm_model is not None:
            unload_llm(llm_model, args.device_llm)

    print(f"Pipeline xong: {answer_count} câu ({time.time() - t0:.1f}s)")
    return answer_count


def generate_answers(
    retrieved: list[dict],
    *,
    llm_model_path: Path,
    max_context_chars: int,
    max_new_tokens: int,
    llm_top_k: int,
    device: str,
    gen_batch: int,
    output_path: Path | None = None,
    include_chunks: bool = False,
    fresh_output: bool = False,
    load_in_4bit: bool = True,
) -> list[dict]:
    model, tokenizer, eos_ids = load_llm(llm_model_path, device, load_in_4bit=load_in_4bit)

    outputs: list[dict] = []
    total = len(retrieved)
    adaptive_batch = AdaptiveGenBatch(gen_batch)
    if fresh_output and output_path is not None:
        save_json(output_path, [])

    start = 0
    while start < total:
        batch_items = retrieved[start : start + adaptive_batch.size]
        batch_results = generate_items_adaptive(
            model,
            tokenizer,
            batch_items,
            gen_batch=adaptive_batch,
            max_context_chars=max_context_chars,
            max_new_tokens=max_new_tokens,
            eos_ids=eos_ids,
            llm_top_k=llm_top_k,
            device=device,
        )
        if device == "cuda":
            torch.cuda.empty_cache()
        for result in batch_results:
            if include_chunks:
                chunks = next(x["chunks"] for x in batch_items if x["id"] == result["id"])
                result["retrieved_chunks"] = chunks
            outputs.append(result)

        done = min(start + len(batch_items), total)
        print(f"  Generated {done}/{total}")
        if output_path is not None:
            if fresh_output:
                existing = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else []
                existing.extend(batch_results)
                save_json(output_path, existing)
            else:
                save_json(output_path, merge_existing(output_path, outputs))
        start += len(batch_items)

    unload_llm(model, device)
    return outputs


def save_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_existing(output_path: Path, new_items: list[dict]) -> list[dict]:
    if not output_path.exists():
        return new_items
    existing = json.loads(output_path.read_text(encoding="utf-8"))
    by_id = {x["id"]: x for x in existing}
    for item in new_items:
        by_id[item["id"]] = item
    return [by_id[k] for k in sorted(by_id)]


def main() -> int:
    args = parse_args()
    init_qdrant_from_args(args)
    args.embed_model = get_embed_model_path(args.embed_model)
    args.rerank_model = get_rerank_model_path(args.rerank_model)

    if not args.questions.exists():
        print(f"Không tìm thấy: {args.questions}", file=sys.stderr)
        return 1

    questions = load_questions(args.questions, args.start_id, args.limit)
    if args.skip_answered and args.output.exists():
        answered_ids = {
            item["id"] for item in json.loads(args.output.read_text(encoding="utf-8"))
        }
        before = len(questions)
        questions = [q for q in questions if q["id"] not in answered_ids]
        print(f"Bỏ qua {before - len(questions)} câu đã trả lời, còn {len(questions)}")
    print(f"Câu hỏi: {len(questions)} (start_id={args.start_id})")
    if not questions:
        print("Không còn câu hỏi cần xử lý.")
        return 0

    if args.citations_only and args.skip_retrieve and args.retrieved_cache.exists():
        retrieved = json.loads(args.retrieved_cache.read_text(encoding="utf-8"))
        ids = {q["id"] for q in questions}
        retrieved = [r for r in retrieved if r["id"] in ids]
        print(f"Dùng cache retrieve: {len(retrieved)} câu")
        print("=== Extract citations (from cache, no LLM) ===")
        t1 = time.time()
        citations = build_citation_results(retrieved, args.llm_top_k)
        del retrieved
        if args.skip_answered:
            merged = merge_existing(args.output, citations)
            save_json(args.output, merged)
            print(f"Đã lưu: {args.output} ({len(merged)} câu, {time.time()-t1:.1f}s)")
        else:
            save_json(args.output, citations)
            print(f"Đã lưu: {args.output} ({len(citations)} câu, {time.time()-t1:.1f}s)")
        return 0

    if args.citations_only:
        citation_count = run_citations_pipeline(questions, args=args)
        print(f"Đã lưu: {args.output} ({citation_count} câu)")
        return 0

    if args.retrieve_only:
        print("=== Retrieve only (incremental cache) ===")
        t0 = time.time()
        engine = make_retrieval_engine(args)
        try:
            total = len(questions)
            for start in range(0, total, args.retrieve_batch):
                batch_qs = questions[start : start + args.retrieve_batch]
            if not args.no_subquery and not engine.subquery_index:
                engine.sub_query_map = resolve_sub_query_map(
                    batch_qs,
                    args=args,
                )
                retrieved_batch = engine.retrieve_batch(batch_qs)
                append_retrieved_cache(args.retrieved_cache, retrieved_batch)
                done = min(start + args.retrieve_batch, total)
                print(f"  Retrieved {done}/{total}")
                del retrieved_batch
        finally:
            engine.unload()
        print(f"Đã lưu retrieve cache: {args.retrieved_cache} ({time.time()-t0:.1f}s)")
        return 0

    if args.skip_retrieve and args.retrieved_cache.exists():
        retrieved = json.loads(args.retrieved_cache.read_text(encoding="utf-8"))
        ids = {q["id"] for q in questions}
        retrieved = [r for r in retrieved if r["id"] in ids]
        print(f"Dùng cache retrieve: {len(retrieved)} câu")
        print("=== Generate answers (from cache) ===")
        t1 = time.time()
        answers = generate_answers(
            retrieved,
            llm_model_path=args.llm_model,
            max_context_chars=args.max_context_chars,
            max_new_tokens=args.max_new_tokens,
            llm_top_k=args.llm_top_k,
            device=args.device_llm,
            gen_batch=args.gen_batch,
            output_path=args.output,
            include_chunks=args.include_chunks,
            fresh_output=not args.skip_answered,
            load_in_4bit=not args.no_4bit,
        )
        del retrieved
        if args.skip_answered:
            merged = merge_existing(args.output, answers)
            save_json(args.output, merged)
            print(f"Đã lưu: {args.output} ({len(merged)} câu, {time.time()-t1:.1f}s)")
        else:
            print(f"Đã lưu: {args.output} ({len(answers)} câu, {time.time()-t1:.1f}s)")
        return 0

    answer_count = run_interleaved_pipeline(questions, args=args)
    print(f"Đã lưu: {args.output} ({answer_count} câu)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
