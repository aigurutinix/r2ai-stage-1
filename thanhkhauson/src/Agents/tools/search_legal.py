from typing import List, Dict, Any, Tuple
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import torch
import os
import threading
import json
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import py_vncorenlp
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from Agents.logs.agent_logger import logger
from Agents.llm_client import get_llm

from Agents.config import (
    RERANKER_PATH, 
    ES_HOST, QDRANT_HOST, QDRANT_PORT, INDEX_NAME, 
    COMPRESS_LLM_TEMPERATURE, COMPRESS_LLM_TOP_P, COMPRESS_LLM_TOP_K, COMPRESS_LLM_ENABLE_THINKING,
    RETRIEVER_TOP_K,
    RERANKER_TOP_K,
    SEARCH_LOCAL_DOCS,
    EMBEDDING_MAX_CONCURRENT,
    RERANKER_MAX_CONCURRENT,
    RERANKER_BATCH_SIZE,
    VNCORENLP_MAX_CONCURRENT,
    EMBEDDING_MODEL_PATH, VNCORENLP_DIR
)

reranker_tokenizer = None
reranker_model = None
embedding_model = None
vncorenlp_rdrsegmenter = None
es_client = None
qdrant_client = None
_resource_init_lock = threading.Lock()
_embedding_queue = threading.BoundedSemaphore(EMBEDDING_MAX_CONCURRENT)
_reranker_queue = threading.BoundedSemaphore(RERANKER_MAX_CONCURRENT)
_vncorenlp_queue = threading.BoundedSemaphore(VNCORENLP_MAX_CONCURRENT)
_filter_stats_log_lock = threading.Lock()
device = "cuda" if torch.cuda.is_available() else "cpu"

def _acquire_queue(queue: threading.BoundedSemaphore, name: str) -> None:
    if not queue.acquire(blocking=False):
        logger.info(f"[*] {name} đang bận, request sẽ chờ trong queue...")
        queue.acquire()

def _release_queue(queue: threading.BoundedSemaphore) -> None:
    queue.release()

def get_reranker():
    """Trả về reranker đã được nạp sẵn."""
    global reranker_tokenizer, reranker_model
    if reranker_model is None or reranker_tokenizer is None:
        with _resource_init_lock:
            if reranker_model is None or reranker_tokenizer is None:
                logger.info(f"[*] Đang nạp mô hình ViRanker từ: {RERANKER_PATH}")
                reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_PATH)
                reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_PATH).to(device)
                reranker_model.eval()
                logger.info("[*] Nạp ViRanker thành công!")
    return reranker_tokenizer, reranker_model

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        with _resource_init_lock:
            if embedding_model is None:
                logger.info(f"[*] Đang nạp mô hình Embedding từ: {EMBEDDING_MODEL_PATH}")
                embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH, model_kwargs={"torch_dtype": torch.float32})
                embedding_model.max_seq_length = 512
                logger.info("[*] Nạp Embedding Model thành công!")
    return embedding_model

def get_vncorenlp():
    global vncorenlp_rdrsegmenter
    if vncorenlp_rdrsegmenter is None:
        with _resource_init_lock:
            if vncorenlp_rdrsegmenter is None:
                logger.info("[*] Đang khởi tạo VnCoreNLP...")
                os.makedirs(VNCORENLP_DIR, exist_ok=True)
                if not os.path.exists(os.path.join(VNCORENLP_DIR, "models")):
                    py_vncorenlp.download_model(save_dir=VNCORENLP_DIR)
                vncorenlp_rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=VNCORENLP_DIR)
                logger.info("[*] Nạp VnCoreNLP thành công!")
    return vncorenlp_rdrsegmenter

def get_es_client():
    global es_client
    if es_client is None:
        with _resource_init_lock:
            if es_client is None:
                logger.info(f"[*] Đang kết nối Elasticsearch tại: {ES_HOST}")
                es_client = Elasticsearch(ES_HOST)
    return es_client

def get_qdrant_client():
    global qdrant_client
    if qdrant_client is None:
        with _resource_init_lock:
            if qdrant_client is None:
                logger.info(f"[*] Đang kết nối Qdrant Server tại: {QDRANT_HOST}:{QDRANT_PORT}")
                qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return qdrant_client

def initialize_resources():
    """Nạp toàn bộ resource dùng chung ngay lúc khởi động process."""
    logger.info("[*] Khởi tạo sẵn các model và client truy hồi...")
    get_vncorenlp()
    get_embedding_model()
    get_reranker()
    get_es_client()
    get_qdrant_client()
    logger.info("[*] Hoàn tất khởi tạo resource.")

def tokenize_text(text: str) -> str:
    if not text: return ""
    rdrsegmenter = get_vncorenlp()
    _acquire_queue(_vncorenlp_queue, "VnCoreNLP")
    try:
        sentences = rdrsegmenter.word_segment(text)
        return " ".join(sentences)
    finally:
        _release_queue(_vncorenlp_queue)

def perform_batch_hybrid_search(bm25_queries: List[str], dense_queries: List[str], top_k: int = RETRIEVER_TOP_K) -> List[List[Dict[str, Any]]]:
    """Chạy hybrid search cho nhiều query cùng lúc."""
    es = get_es_client()
    qdrant = get_qdrant_client()
    embed_model = get_embedding_model()
    
    logger.info(f"[*] Batch truy vấn Database - {len(dense_queries)} queries")
    
    instruct_queries = [f"Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: {q}" for q in dense_queries]
    _acquire_queue(_embedding_queue, "Embedding model")
    try:
        query_vectors = embed_model.encode(instruct_queries, batch_size=len(instruct_queries), normalize_embeddings=True, show_progress_bar=False)
    finally:
        _release_queue(_embedding_queue)
    
    all_final_docs = []
    
    es_body = []
    for bm25_query in bm25_queries:
        tokenized_query = tokenize_text(bm25_query)
        es_body.append({"index": INDEX_NAME})
        match_query = {"match": {"content_search": {"query": tokenized_query}}}
        if not SEARCH_LOCAL_DOCS:
            final_query = {
                "bool": {
                    "must": [match_query],
                    "filter": [{"term": {"is_local": False}}]
                }
            }
        else:
            final_query = match_query
            
        es_body.append({"query": final_query, "size": top_k, "_source": False})
        
    try:
        es_res = es.msearch(body=es_body)
        es_responses = [resp.get("hits", {}).get("hits", []) for resp in es_res.get("responses", [])]
    except Exception as e:
        logger.error(f"[!] Lỗi truy vấn Elasticsearch msearch: {e}")
        es_responses = [[] for _ in bm25_queries]

    from qdrant_client.models import QueryRequest, Filter, FieldCondition, MatchValue
    
    qdrant_filter = None
    if not SEARCH_LOCAL_DOCS:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="is_local",
                    match=MatchValue(value=False)
                )
            ]
        )

    qdrant_requests = [
        QueryRequest(
            query=query_vectors[idx].tolist(), 
            limit=top_k,
            filter=qdrant_filter
        )
        for idx in range(len(query_vectors))
    ]
    try:
        qdrant_res_batch = qdrant.query_batch_points(
            collection_name=INDEX_NAME,
            requests=qdrant_requests
        )
        qdrant_responses = [resp.points for resp in qdrant_res_batch]
    except Exception as e:
        logger.error(f"[!] Lỗi truy vấn Qdrant batch: {e}")
        qdrant_responses = [[] for _ in dense_queries]
    
    ranked_chunk_ids_by_query = []
    rrf_scores_by_query = []
    all_chunk_ids = []
    seen_chunk_ids = set()

    for idx, (bm25_query, dense_query) in enumerate(zip(bm25_queries, dense_queries)):
        es_hits = es_responses[idx] if idx < len(es_responses) else []
        qdrant_res = qdrant_responses[idx] if idx < len(qdrant_responses) else []
        
        k = 60
        rrf_scores = {}
        for rank, hit in enumerate(es_hits):
            chunk_id = hit["_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, point in enumerate(qdrant_res):
            chunk_id = point.id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        ranked_chunk_ids_by_query.append(sorted_chunk_ids)
        rrf_scores_by_query.append(rrf_scores)

        for chunk_id in sorted_chunk_ids:
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                all_chunk_ids.append(chunk_id)

    docs_by_chunk_id = {}
    if all_chunk_ids:
        try:
            mget_res = es.mget(index=INDEX_NAME, body={"ids": all_chunk_ids})
            for requested_id, doc in zip(all_chunk_ids, mget_res.get("docs", [])):
                if not doc.get("found"):
                    continue

                src = doc.get("_source", {}) or {}
                chunk_id = src.get("chunk_id") or doc.get("_id") or requested_id
                if not chunk_id:
                    continue

                law_title = str(src.get('raw_title') or '').strip()
                article_no = str(src.get('article_no') or '').strip()
                article_title = str(src.get('article_title') or '').strip()
                raw_content = str(src.get('raw_content') or '').strip()

                prefix = f"{law_title}\n" if law_title else ""
                if article_no: prefix += f"{article_no}: {article_title}\n" if article_title else f"{article_no}\n"
                elif article_title: prefix += f"{article_title}\n"

                prepared_doc = {
                    "id": chunk_id,
                    "text": prefix + raw_content,
                }
                lookup_ids = {requested_id, str(requested_id), chunk_id, str(chunk_id)}
                es_doc_id = doc.get("_id")
                if es_doc_id:
                    lookup_ids.update({es_doc_id, str(es_doc_id)})
                for lookup_id in lookup_ids:
                    docs_by_chunk_id[lookup_id] = prepared_doc
        except Exception as e:
            logger.error(f"[!] Lỗi lấy nội dung Elasticsearch mget: {e}")

    for sorted_chunk_ids, rrf_scores in zip(ranked_chunk_ids_by_query, rrf_scores_by_query):
        final_docs = []
        for chunk_id in sorted_chunk_ids:
            doc = docs_by_chunk_id.get(chunk_id)
            if not doc:
                continue
            final_doc = dict(doc)
            final_doc["rrf_score"] = rrf_scores.get(chunk_id, 0.0)
            final_docs.append(final_doc)

        final_docs = sorted(final_docs, key=lambda x: x.get("rrf_score", 0), reverse=True)
        all_final_docs.append(final_docs)
        
    return all_final_docs

def perform_batch_local_rerank(queries: List[str], raw_docs_list: List[List[Dict[str, Any]]], top_k: int = RERANKER_TOP_K) -> List[List[Dict[str, Any]]]:
    """Rerank batch tất cả tài liệu của tất cả queries."""
    tokenizer, model = get_reranker()
    
    flat_pairs = []
    doc_indices = []
    for q_idx, (query, raw_docs) in enumerate(zip(queries, raw_docs_list)):
        for d_idx, doc in enumerate(raw_docs):
            flat_pairs.append([query, doc.get("text", "")])
            doc_indices.append((q_idx, d_idx))
            
    if not flat_pairs:
        return [[] for _ in queries]
        
    logger.info(f"[*] Đang Batch Rerank {len(flat_pairs)} cặp câu hỏi-tài liệu")
    batch_size = RERANKER_BATCH_SIZE
    all_scores = []
    
    _acquire_queue(_reranker_queue, "Reranker model")
    try:
        with torch.inference_mode():
            for i in range(0, len(flat_pairs), batch_size):
                batch_pairs = flat_pairs[i:i+batch_size]
                inputs = tokenizer(batch_pairs, padding=True, truncation=True, return_tensors='pt', max_length=1024).to(model.device)
                scores = model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().numpy()
                all_scores.extend(scores.tolist())
    finally:
        _release_queue(_reranker_queue)
            
    for (q_idx, d_idx), score in zip(doc_indices, all_scores):
        raw_docs_list[q_idx][d_idx]["rerank_score"] = float(score)
        
    reranked_docs_list = []
    for raw_docs in raw_docs_list:
        sorted_docs = sorted(raw_docs, key=lambda x: x.get("rerank_score", 0), reverse=True)
        reranked_docs_list.append(sorted_docs[:top_k])
        
    return reranked_docs_list

class QueryPair(BaseModel):
    bm25_query: str = Field(description="Từ khóa tìm kiếm cho BM25 (chứa các keyword quan trọng, loại bỏ stopwords).")
    dense_query: str = Field(description="Câu hỏi hoặc văn bản ngữ nghĩa cho Dense/Vector search (giữ nguyên ngữ cảnh).")

class SearchInput(BaseModel):
    queries: List[QueryPair] = Field(description="Danh sách các cặp truy vấn để tìm kiếm chung một mục đích cho mỗi cặp. Mỗi mục cần có 1 bm25_query và 1 dense_query tương ứng.")

class ToolCompressOutput(BaseModel):
    relevant_chunk_ids: List[str] = Field(description="Danh sách ID (Ví dụ: DOC_0, DOC_1) của các tài liệu chứa thông tin liên quan đến mục tiêu tìm kiếm. Bỏ qua các ID vô giá trị.")

def _log_tool_filter_stats(total: int, kept: int, **extra: Any) -> None:
    total = max(0, int(total or 0))
    kept = max(0, int(kept or 0))
    removed = max(0, total - kept)
    removed_pct = (removed / total * 100.0) if total else 0.0
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "node": "Tool Compress Stats",
        "input": total,
        "kept": kept,
        "removed": removed,
        "removed_pct": round(removed_pct, 3),
        **extra,
    }
    logger.info(
        "[Tool Compress Stats] input=%s kept=%s removed=%s removed_pct=%.1f%% %s",
        total,
        kept,
        removed,
        removed_pct,
        " ".join(f"{key}={value}" for key, value in extra.items()),
    )

    log_path = os.getenv("FILTER_STATS_LOG_PATH", "").strip()
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with _filter_stats_log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Không ghi được tool filter stats log: {e}")

def _normalize_query_pairs(queries: List[QueryPair]) -> List[QueryPair]:
    normalized_queries = []
    for query in queries or []:
        if isinstance(query, QueryPair):
            normalized_queries.append(query)
        elif isinstance(query, dict):
            normalized_queries.append(QueryPair(**query))
    return normalized_queries

def _invoke_structured_with_retries(chain, payload: Dict[str, Any], label: str, max_attempts: int = 3):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = chain.invoke(payload)
            if result is None:
                raise ValueError("Structured output returned None")
            return result
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                logger.warning(f"{label} structured output lỗi lần {attempt}/{max_attempts}, retry: {e}")
            else:
                logger.error(f"{label} structured output lỗi sau {max_attempts} lần: {e}")
    raise RuntimeError(f"{label} structured output failed after {max_attempts} attempts") from last_error

def _required_structured_tool(llm, schema):
    from langchain_core.output_parsers.openai_tools import PydanticToolsParser

    return llm.bind_tools([schema], tool_choice="required") | PydanticToolsParser(
        tools=[schema],
        first_tool_only=True,
    )

def run_hybrid_search_with_ids(queries: List[QueryPair]) -> Tuple[str, List[str]]:
    "Chạy logic search thật và trả cả text cho LLM lẫn chunk IDs để cập nhật state."
    queries = _normalize_query_pairs(queries)
    if not queries:
        _log_tool_filter_stats(0, 0, mode="no_queries", query_count=0)
        return "Không có truy vấn tìm kiếm hợp lệ.", []

    bm25_queries = [q.bm25_query for q in queries]
    dense_queries = [q.dense_query for q in queries]
    logger.info(f"[*] Chạy truy vấn bổ sung với {len(queries)} cặp query.")

    from langchain_core.prompts import ChatPromptTemplate

    raw_docs_list = perform_batch_hybrid_search(bm25_queries, dense_queries, top_k=RETRIEVER_TOP_K)
    reranked_docs_list = perform_batch_local_rerank(dense_queries, raw_docs_list, top_k=RERANKER_TOP_K)

    seen_ids = set()
    merged_docs = []
    for docs in reranked_docs_list:
        for doc in docs:
            doc_id = doc.get("id")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged_docs.append(doc)

    if not merged_docs:
        _log_tool_filter_stats(0, 0, mode="no_docs", query_count=len(queries))
        return "Không tìm thấy kết quả pháp lý nào.", []

    logger.info(f"[*] Đang chạy LLM Filter chung cho {len(merged_docs)} tài liệu...")
    context_text = ""
    doc_mapping = {}
    for idx, doc in enumerate(merged_docs):
        temp_id = f"DOC_{idx}"
        doc_mapping[temp_id] = doc
        context_text += f">> Tài liệu: {temp_id} <<\nNội dung:\n{doc.get('text', '')}\n\n"

    combined_query = "\n- ".join([""] + dense_queries)

    llm = get_llm(
        temperature=COMPRESS_LLM_TEMPERATURE,
        top_p=COMPRESS_LLM_TOP_P,
        top_k=COMPRESS_LLM_TOP_K,
        enable_thinking=COMPRESS_LLM_ENABLE_THINKING
    )
    structured_llm = _required_structured_tool(llm, ToolCompressOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một Thẩm định viên pháp lý chuyên nghiệp. Nhiệm vụ của bạn là LỌC dữ liệu đầu vào để tìm ra căn cứ pháp lý chính xác nhất.
Quy tắc cực kỳ nghiêm ngặt:
1. Bạn sẽ nhận được các truy vấn/từ khóa tìm kiếm bổ sung và các tài liệu được hệ thống tìm về.
2. Đối chiếu từng tài liệu với mục đích của truy vấn bổ sung.
3. Lưu ý kiểm tra kỹ các metadata đi kèm trong nội dung tài liệu như legal_type, document_number, article_no, effect_date để xác định đúng căn cứ.
4. NẾU tài liệu KHÔNG LIÊN QUAN hoặc không giải quyết được mục đích tìm kiếm -> BỎ QUA ID ĐÓ.
5. NẾU tài liệu LIÊN QUAN TRỰC TIẾP và đáp ứng đúng mục đích tìm kiếm -> Đưa ID (DOC_X) của nó vào danh sách trả về.
6. Không giữ tài liệu chỉ vì có chung vài từ khóa bề mặt nhưng không trả lời được vấn đề pháp lý cần tìm.

Mục đích của bạn là cung cấp bộ chứng cứ sạch, chính xác và có giá trị pháp lý cao nhất.
BẮT BUỘC trả về kết quả định dạng JSON với duy nhất một key `relevant_chunk_ids` chứa mảng các ID."""),
        ("human", "Các truy vấn/từ khóa tìm kiếm bổ sung:\n{query}\n\nCác tài liệu:\n{context}")
    ])
    chain = prompt | structured_llm

    try:
        result = _invoke_structured_with_retries(
            chain,
            {"query": combined_query, "context": context_text},
            "Tool Compress",
        )
        valid_temp_ids = result.relevant_chunk_ids
        unique_docs = [doc_mapping[tid] for tid in valid_temp_ids if tid in doc_mapping]
        if not unique_docs:
            unique_docs = merged_docs[:1]
            _log_tool_filter_stats(
                len(merged_docs),
                len(unique_docs),
                mode="fallback_empty_filter",
                query_count=len(queries),
                selected_temp_ids=len(valid_temp_ids),
            )
        else:
            _log_tool_filter_stats(
                len(merged_docs),
                len(unique_docs),
                mode="structured",
                query_count=len(queries),
                selected_temp_ids=len(valid_temp_ids),
            )
    except Exception as e:
        logger.error(f"[!] Lỗi LLM nén chung: {e}")
        unique_docs = merged_docs[:1]
        _log_tool_filter_stats(
            len(merged_docs),
            len(unique_docs),
            mode="fallback_error",
            query_count=len(queries),
        )

    if not unique_docs:
        return "Không tìm thấy kết quả pháp lý nào.", []

    result_text = "\n\n".join([f"--- Nguồn (ID: {d.get('id')}) ---\n{d.get('text', '')}" for d in unique_docs])
    chunk_ids = [str(d.get("id")) for d in unique_docs if d.get("id")]
    return result_text, chunk_ids

@tool(args_schema=SearchInput)
def hybrid_search_tool(queries: List[QueryPair]) -> str:
    "Sử dụng công cụ này để tìm kiếm thêm các điều luật và văn bản pháp luật trên cơ sở dữ liệu."
    result_text, _ = run_hybrid_search_with_ids(queries)
    return result_text


initialize_resources()
