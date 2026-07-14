import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from google import genai as google_genai
from google.genai import types as genai_types
import chromadb

# Load env variables
load_dotenv()

# Lazily-initialized google-genai client (re-used across embedding calls)
_genai_client: Optional[google_genai.Client] = None


def _get_genai_client() -> Optional[google_genai.Client]:
    """Return a cached google-genai Client, or None if GEMINI_API_KEY is not set."""
    global _genai_client
    if _genai_client is not None:
        return _genai_client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    _genai_client = google_genai.Client(api_key=api_key)
    return _genai_client

class LegalRAGEngine:
    def __init__(self, chroma_path: str = None):
        if chroma_path is None:
            # Look up path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            chroma_path = os.path.join(current_dir, "..", "data", "chroma_db")
        
        self.chroma_path = os.path.abspath(chroma_path)
        print(f"Connecting to ChromaDB at: {self.chroma_path}")
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="vietnamese_laws",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Keep legal_db as a dummy empty list for backward compatibility
        self.legal_db = []
        
        # Dialect and Terminology Translation Map
        self.slang_map = {
            "đuổi việc": ["sa thải", "đơn phương chấm dứt hợp đồng"],
            "cho nghỉ": ["sa thải", "đơn phương chấm dứt hợp đồng", "chấm dứt hợp đồng lao động"],
            "bắt đền": ["bồi thường thiệt hại", "trách nhiệm vật chất"],
            "đền tiền": ["bồi thường thiệt hại", "hoàn trả"],
            "lương lậu": ["tiền lương", "tiền công", "thù lao"],
            "tiền lời": ["tiền hoa hồng", "lợi nhuận"],
            "làm thử": ["thử việc"],
            "nghỉ ngang": ["tự ý bỏ việc", "đơn phương chấm dứt trái pháp luật"],
            "bỏ việc": ["tự ý bỏ việc"],
            "mở công ty con": ["chi nhánh", "văn phòng đại diện", "đăng ký doanh nghiệp"],
            "lập văn phòng": ["văn phòng đại diện", "địa điểm kinh doanh"]
        }
        
    def _generate_embedding(self, text: str, is_query: bool = False) -> Optional[List[float]]:
        """Calls Gemini API (google-genai SDK) to generate real semantic vector embeddings."""
        client = _get_genai_client()
        if client is None:
            return None
        try:
            task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=[text],
                config=genai_types.EmbedContentConfig(task_type=task_type),
            )
            # New SDK returns result.embeddings: list[ContentEmbedding] each with .values
            if result.embeddings:
                return list(result.embeddings[0].values)
            return None
        except Exception as e:
            print(f"Error generating embedding via Gemini API: {e}")
            return None

    def _load_db(self) -> List[Dict[str, Any]]:
        """Dummy method for backward compatibility."""
        return []

    def translate_slang(self, query: str) -> str:
        """Translates informal terms and slang to formal legal terms to enhance search."""
        normalized_query = query.lower()
        for slang, formal_terms in self.slang_map.items():
            if slang in normalized_query:
                # Append formal terms to the query to boost sparse retrieval
                normalized_query += " " + " ".join(formal_terms)
        return normalized_query

    def _is_effective_at(self, doc: Dict[str, Any], event_date_str: Optional[str]) -> bool:
        """Checks if a document is effective at the time of the specified event date."""
        if not event_date_str:
            # If no event date specified, default to active documents as of today
            event_date = datetime.now()
        else:
            try:
                event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
            except ValueError:
                event_date = datetime.now()

        try:
            effective_date = datetime.strptime(doc.get("effective_date", "1970-01-01"), "%Y-%m-%d")
        except Exception:
            effective_date = datetime.min

        expiration_date = None
        if doc.get("expiration_date"):
            try:
                expiration_date = datetime.strptime(doc["expiration_date"], "%Y-%m-%d")
            except Exception:
                pass

        is_effective = effective_date <= event_date
        is_not_expired = expiration_date is None or expiration_date > event_date
        return is_effective and is_not_expired and doc.get("status") == "active"

    def filter_by_time(self, documents: List[Dict[str, Any]], event_date_str: Optional[str]) -> List[Dict[str, Any]]:
        """Filters documents based on status and historical event date to prevent retroactive application."""
        return [doc for doc in documents if self._is_effective_at(doc, event_date_str)]

    def resolve_conflicts(self, matched_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolves overlapping or conflicting legal norms based on hierarchical precedence and lex posterior."""
        # legislative hierarchy rank (higher is more authoritative)
        def get_rank(law_name: str) -> int:
            law_name_upper = law_name.upper()
            if "HIẾN PHÁP" in law_name_upper:
                return 4
            elif "LUẬT" in law_name_upper or "BỘ LUẬT" in law_name_upper or "BLLĐ" in law_name_upper or "LDN" in law_name_upper:
                return 3
            elif "NGHỊ ĐỊNH" in law_name_upper or "NĐ-CP" in law_name_upper or "ND" in law_name_upper:
                return 2
            elif "THÔNG TƯ" in law_name_upper or "TT-" in law_name_upper:
                return 1
            return 1
        
        # Sort by hierarchy rank descending, then by effective date descending (lex posterior)
        def sort_key(doc):
            law_name = doc.get("law_name", "")
            rank = get_rank(law_name)
            eff_date = doc.get("effective_date", "1970-01-01")
            return (rank, eff_date)
            
        return sorted(matched_docs, key=sort_key, reverse=True)

    def hybrid_search(self, query: str, event_date_str: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs a highly optimized Dense Vector + Sparse Keyword Hybrid Search over ChromaDB."""
        # 1. Generate query embedding
        query_vector = self._generate_embedding(query, is_query=True)
        if not query_vector:
            # If embeddings failed, retrieve a fallback list using empty metadata search
            print("Warning: Embedding generation failed. Falling back to simple retrieve.")
            results = self.collection.get(limit=150)
            distances = [1.0] * len(results["ids"])
        else:
            # Dense Vector Query top 150 matches from ChromaDB
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=150
            )
            # results keys: 'ids', 'distances', 'metadatas', 'embeddings', 'documents'
            # Note: query returns list of lists (since batch queries are supported)
            results = {k: v[0] if v is not None else None for k, v in results.items()}
            distances = results.get("distances") or []

        metadatas = results.get("metadatas") or []
        if not metadatas:
            return []

        # 2. Sparse Keyword Search Setup & Scoring
        translated_query = self.translate_slang(query)
        query_words = set(translated_query.lower().split())

        scored_docs = []
        for idx, doc in enumerate(metadatas):
            # Calculate dense score: cosine distance in ChromaDB is 1 - cosine_similarity
            # Similarity = 1 - distance
            distance = distances[idx] if idx < len(distances) else 0.5
            dense_score = max(0.0, 1.0 - distance)

            # Sparse score based on word overlap
            sparse_score = 0.0
            doc_text = (doc.get("title", "") + " " + doc.get("text", "")).lower()
            
            for word in query_words:
                if word in doc_text:
                    sparse_score += 2.0
                    
            # Slang Synonym Boost
            for slang, formal_terms in self.slang_map.items():
                if slang in query.lower():
                    for term in formal_terms:
                        if term in doc_text:
                            sparse_score += 1.5

            # Combine scores with weightings
            combined_score = (10.0 * dense_score) + (1.0 * sparse_score)
            scored_docs.append((combined_score, doc))

        # 3. Timeline filtering
        available_docs = []
        for score, doc in scored_docs:
            if self._is_effective_at(doc, event_date_str):
                available_docs.append((score, doc))

        if not available_docs:
            return []

        # 4. Sort and select top candidates
        available_docs.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [doc for score, doc in available_docs[:top_k * 2]]
        
        # 5. Resolve conflicts using legislative rank precedence
        resolved_candidates = self.resolve_conflicts(top_candidates)
        
        # 6. Parent-Child retrieval integration
        final_results = []
        seen_parent_ids = set()
        
        for candidate in resolved_candidates:
            if len(final_results) >= top_k:
                break
                
            if candidate.get("type") == "child":
                parent_id = candidate.get("parent_id")
                if parent_id and parent_id not in seen_parent_ids:
                    # Query parent document directly from ChromaDB by ID
                    parent_results = self.collection.get(ids=[parent_id])
                    if parent_results and parent_results["metadatas"]:
                        parent_doc = parent_results["metadatas"][0]
                        seen_parent_ids.add(parent_id)
                        merged_doc = parent_doc.copy()
                        merged_doc["matched_child_clause"] = candidate.get("text", "")
                        merged_doc["matched_child_title"] = candidate.get("title", "")
                        final_results.append(merged_doc)
            else:
                parent_id = candidate.get("id")
                if parent_id and parent_id not in seen_parent_ids:
                    seen_parent_ids.add(parent_id)
                    final_results.append(candidate)
                    
        return final_results

# Quick test if run directly
if __name__ == "__main__":
    engine = LegalRAGEngine()
    import time
    start = time.perf_counter()
    results = engine.hybrid_search("Nhân viên nghỉ ngang tự ý bỏ việc 5 ngày có sa thải được không")
    duration = time.perf_counter() - start
    print(f"Query completed in {duration*1000:.3f} ms. Total results: {len(results)}")
    for r in results:
        print(f"\n- Title: {r['title']}\n- Law: {r['law_name']}\n- URL: {r.get('source_url')}")
