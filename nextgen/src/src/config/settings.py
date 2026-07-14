"""
config/settings.py
Cấu hình tập trung cho toàn bộ pipeline RAG pháp lý.
Tối ưu cho môi trường Kaggle GPU (T4/P100) + DeepSeek-R1-Distill-Qwen-14B
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # =========================================================
    # QDRANT CLOUD
    # =========================================================
    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME = "law_2026"
    EMBEDDING_DIM = 1024  # Chiều vector của AITeamVN/Vietnamese_Embedding

    # =========================================================
    # EMBEDDING MODEL
    # =========================================================
    EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"

    # =========================================================
    # BM25 & HYBRID RETRIEVAL
    # =========================================================
    RRF_K = 60          # Hằng số RRF chuẩn
    TOP_K_RAW = 30      # Số văn bản thô lấy từ mỗi nhánh (Dense + Sparse)

    # =========================================================
    # MULTI-QUERY EXPANSION
    # =========================================================
    NUM_QUERY_VARIANTS = 3   # Số biến thể câu hỏi sinh ra để mở rộng retrieval

    # =========================================================
    # RERANKER (Cross-Encoder)
    # Chạy local trên Kaggle GPU
    # =========================================================
    RERANKER_MODEL = "BAAI/bge-reranker-large"
    RERANKER_THRESHOLD = 0.30  # Ngưỡng tối thiểu để giữ văn bản (giảm xuống 0.30 để an toàn hơn)
    TOP_K_FINAL = 6            # Số văn bản đưa vào LLM (5 để có đủ context)

    # =========================================================
    # GIỚI HẠN OUTPUT NỘP BÀI (relevant_docs / relevant_articles)
    # F2 phạt precision nặng khi ground-truth mỗi câu chỉ 1-3 điều
    # → KHÔNG dump cả TOP_K_FINAL chunk ra field nộp bài.
    # =========================================================
    # top-2 rerank cho F2 cao nhất (mô phỏng trên 50 câu GT: article F2≈0.521, doc F2≈0.612).
    RELEVANT_ARTICLES_MAX = 2   # Số Điều emit ra relevant_articles mỗi câu (top-N theo rerank)
    RELEVANT_DOCS_MAX = 2       # Số văn bản emit ra relevant_docs mỗi câu (top-N theo rerank)
    RELEVANT_FALLBACK_K = 2     # (DEPRECATED — không còn dùng sau khi bỏ logic giao citation LLM)

    # =========================================================
    # LLM - DeepSeek-R1-Distill-Qwen-14B trên Kaggle
    # =========================================================
    LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    LLM_MAX_NEW_TOKENS = 1024
    LLM_TEMPERATURE = 0.1      # Thấp → tránh ảo giác pháp lý
    LLM_TOP_P = 0.85
    LLM_REPETITION_PENALTY = 1.1

    # Tham số Regenerate khi Self-Verification thất bại
    LLM_REGEN_TEMPERATURE = 0.05   # Thấp hơn nữa khi sinh lại
    LLM_MAX_RETRIES = 1             # Số lần thử lại tối đa

    # =========================================================
    # PATHS
    # =========================================================
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    LAW_MANIFEST_PATH = os.path.join(DATA_DIR, "law_manifest.json")
    LEGAL_CATALOG_PATH = os.path.join(DATA_DIR, "legal_documents_catalog.json")
    BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_corpus.pkl")

    LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
    RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")
    EVAL_REPORT_PATH = os.path.join(LOG_DIR, "evaluation_report.md")
    DOC_URL_MAP_PATH = os.path.join(DATA_DIR, "doc_url_map.json")

    # =========================================================
    # DEMO WEB APP (app/) — LUỒNG RIÊNG, KHÔNG ảnh hưởng main.py / fast_retrieval.py
    # LLM sinh câu trả lời gọi qua API hosted (OpenAI-compatible: Together/DeepInfra/Novita...)
    # vì máy demo chỉ có CPU. Embedding + Reranker vẫn chạy local (GPU cloud).
    # =========================================================
    LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://api.together.xyz/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_API_MODEL = os.getenv("LLM_API_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo")

    # Số ứng viên (sau rerank) hiển thị làm citation trong app demo.
    DEMO_RETRIEVE_POOL = 5
    # Độ dài đoạn trích văn bản gốc hiển thị trong mỗi thẻ citation (ký tự).
    DEMO_SNIPPET_CHARS = 400

    # =========================================================
    # SUPABASE (đăng nhập Google + lưu lịch sử hội thoại)
    # anon key là khóa CÔNG KHAI (an toàn nhúng frontend); frontend lấy qua endpoint /config.
    # =========================================================
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

    # =========================================================
    # SELF-VERIFICATION RULES
    # =========================================================
    MIN_ARTICLE_REFS = 1      # Bắt buộc câu trả lời phải có ít nhất 1 tham chiếu Điều X
    MAX_CONTEXT_CHARS = 12000  # Giới hạn ký tự context nạp vào LLM (Qwen2.5-7B context 32k token,
                               # 12000 ký tự ~ 5-6k token, đủ chứa trọn TOP_K_FINAL chunk không bị cắt)
