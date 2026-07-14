"""
app/main.py

Backend FastAPI cho app demo Legal-RAG. Chạy trên máy GPU cloud (đã load embedding + reranker),
gọi LLM qua API + Qdrant Cloud. Expose ra ngoài bằng tunnel (cloudflared/ngrok).

Endpoints:
  GET  /            → trang demo (app/static/index.html)
  GET  /health      → trạng thái service
  POST /ask         → SSE stream: event `citations` (ngay) → nhiều event `token` → `done`/`error`

Chạy:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import os
import json
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.settings import Settings
from app.service import RagService
from src.api_answer_generator import clean_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Legal-RAG Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo: cho phép mọi origin (qua tunnel)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nạp service MỘT LẦN lúc khởi động (không dùng biến toàn cục lười để lỗi sớm nếu thiếu cấu hình).
_service: Optional[RagService] = None


@app.on_event("startup")
def _startup():
    global _service
    _service = RagService()


class AskRequest(BaseModel):
    question: str


def _sse(event: str, data) -> str:
    """
    Đóng gói một sự kiện SSE. LUÔN JSON-encode data (kể cả chuỗi token) để:
      - phía client chỉ cần JSON.parse đồng nhất cho mọi sự kiện;
      - ký tự xuống dòng trong token được escape (\\n) nên không phá khung SSE.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
def health():
    return {"status": "ok", "ready": _service is not None}


@app.get("/config")
def config():
    """Cấu hình công khai cho frontend (Supabase URL + anon key). Không chứa bí mật server."""
    return {
        "supabaseUrl": Settings.SUPABASE_URL,
        "supabaseAnonKey": Settings.SUPABASE_ANON_KEY,
    }


@app.post("/ask")
def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        return JSONResponse({"error": "Câu hỏi trống."}, status_code=400)
    if _service is None:
        return JSONResponse({"error": "Service chưa sẵn sàng."}, status_code=503)

    def event_stream():
        try:
            # --- Giai đoạn nhanh: retrieval + rerank → gửi citations NGAY ---
            ranked = _service.retrieve_and_rerank(question)
            citations, rel_docs, rel_articles = _service.build_citations(ranked)
            yield _sse("citations", {
                "citations": citations,
                "relevant_docs": rel_docs,
                "relevant_articles": rel_articles,
            })

            if not ranked:
                yield _sse("token", "Xin lỗi, tôi không tìm thấy văn bản pháp luật liên quan đến câu hỏi này.")
                yield _sse("done", {"answer": ""})
                return

            # --- Giai đoạn chậm: stream câu trả lời từ LLM API ---
            buffer = []
            for piece in _service.stream_answer(question, ranked):
                buffer.append(piece)
                yield _sse("token", piece)

            yield _sse("done", {"answer": clean_answer("".join(buffer))})
        except Exception as e:  # noqa: BLE001
            logger.exception("Lỗi khi xử lý /ask")
            yield _sse("error", {"message": f"Lỗi hệ thống: {e}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Phục vụ React build (Vite build -> app/static). Mount ở "/" với html=True để trả index.html
# cho "/" và mọi route SPA; các route API (/ask, /health) đã khai báo phía trên nên được ưu tiên.
# Đặt CUỐI FILE để không nuốt các route API.
if os.path.isdir(STATIC_DIR) and os.path.exists(os.path.join(STATIC_DIR, "index.html")):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="spa")
else:
    @app.get("/")
    def _no_build():
        return JSONResponse(
            {"message": "Chưa có React build. Chạy `npm run build` trong web/ (hoặc build qua Docker)."},
            status_code=200,
        )
