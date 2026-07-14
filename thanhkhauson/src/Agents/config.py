import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _positive_int_env(name: str, default: int) -> int:
    return max(1, int(os.getenv(name, str(default))))

def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "legal-rag")

MAIN_LLM_ENABLE_THINKING = _bool_env("MAIN_LLM_ENABLE_THINKING", True)
MAIN_LLM_TEMPERATURE = float(os.getenv("MAIN_LLM_TEMPERATURE", "0.5"))
MAIN_LLM_TOP_P = float(os.getenv("MAIN_LLM_TOP_P", "0.9"))
MAIN_LLM_TOP_K = _positive_int_env("MAIN_LLM_TOP_K", 20)

PLANNER_THINKING_TOKEN_BUDGET = _positive_int_env("PLANNER_THINKING_TOKEN_BUDGET", 1024)
REASONING_THINKING_TOKEN_BUDGET = _positive_int_env("REASONING_THINKING_TOKEN_BUDGET", 8192)
REASONING_ENABLE_TOOLS = _bool_env("REASONING_ENABLE_TOOLS", True)
REASONING_MAX_TOOL_CALLS = _positive_int_env("REASONING_MAX_TOOL_CALLS", 5)

COMPRESS_LLM_ENABLE_THINKING = _bool_env("COMPRESS_LLM_ENABLE_THINKING", True)
COMPRESS_LLM_TEMPERATURE = float(os.getenv("COMPRESS_LLM_TEMPERATURE", "0.3"))
COMPRESS_LLM_TOP_P = float(os.getenv("COMPRESS_LLM_TOP_P", "0.80"))
COMPRESS_LLM_TOP_K = _positive_int_env("COMPRESS_LLM_TOP_K", 20)

RETRIEVER_TOP_K = 15
RERANKER_TOP_K = 4

EMBEDDING_MAX_CONCURRENT = _positive_int_env("EMBEDDING_MAX_CONCURRENT", 1)
RERANKER_MAX_CONCURRENT = _positive_int_env("RERANKER_MAX_CONCURRENT", 1)
VNCORENLP_MAX_CONCURRENT = _positive_int_env("VNCORENLP_MAX_CONCURRENT", 4)
RERANKER_BATCH_SIZE = _positive_int_env("RERANKER_BATCH_SIZE", 32)

SEARCH_LOCAL_DOCS = False
RERANKER_PATH = os.path.join(BASE_DIR, "model_cache/ViRanker")
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "model_cache/vietlegal-harrier-0.6b")

LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5-9b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "65536"))
LLM_REASONING_FORMAT = os.getenv("LLM_REASONING_FORMAT", "deepseek")

ES_HOST = "http://localhost:9201"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
INDEX_NAME = "legal_chunks"

CHECKPOINTER_BACKEND = os.getenv("CHECKPOINTER_BACKEND", "postgres").strip().lower()
CHECKPOINT_POSTGRES_DSN = os.getenv(
    "CHECKPOINT_POSTGRES_DSN",
    f"postgresql://{os.getenv('PGUSER') or os.getenv('USER') or 'postgres'}@localhost:5432/legal_assistant?sslmode=disable",
)
CHECKPOINT_POSTGRES_POOL_SIZE = int(os.getenv("CHECKPOINT_POSTGRES_POOL_SIZE", "10"))
CHECKPOINT_SQLITE_PATH = os.getenv(
    "CHECKPOINT_SQLITE_PATH",
    os.path.join(BASE_DIR, "Agents", "mem", "chat_history.db"),
)

VNCORENLP_DIR = os.path.join(BASE_DIR, "model_cache/vncorenlp")
