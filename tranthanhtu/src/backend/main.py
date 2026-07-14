import os
import re
import uuid
import shutil
import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from agent import create_legal_agent, LegalAgentOutput
from ingestion import LegalDocumentIngestion

app = FastAPI(
    title="AI Legal Assistant for Vietnamese SMEs",
    description="Hệ thống trợ lý pháp lý AI hỗ trợ doanh nghiệp SME Việt Nam tra cứu và giải quyết tình huống pháp lý bằng Google Antigravity SDK.",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to store stateful Agent instances in memory per user session
# session_id -> Agent instance
active_agents: Dict[str, Any] = {}

# Ensure static directories exist
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
generated_docs_dir = os.path.join(static_dir, "generated_docs")
os.makedirs(generated_docs_dir, exist_ok=True)

@app.get("/api/documents")
async def get_all_documents():
    """Returns metadata list of all active uploaded legal documents from registry."""
    registry_path = os.path.join(current_dir, "..", "data", "document_registry.json")
    if not os.path.exists(registry_path):
        return []
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc hồ sơ tài liệu: {str(e)}")

@app.delete("/api/documents/{doc_id}")
async def delete_document_endpoint(doc_id: str):
    """Admin endpoint to delete a raw legal document, remove its registry metadata, and purge its RAG indices."""
    try:
        ingestor = LegalDocumentIngestion()
        success = ingestor.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu trong hồ sơ để xóa.")
            
        # ChromaDB handles database updates dynamically in real-time
        
        return {"status": "success", "message": f"Đã xóa thành công tài liệu và làm sạch cơ sở pháp lý RAG."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi thực hiện xóa tài liệu: {str(e)}")

@app.get("/api/documents/download/{filename}")
async def download_raw_document(filename: str):
    """Serves the raw official legal .docx documents for download."""
    uploaded_dir = os.path.join(static_dir, "uploaded_documents")
    file_path = os.path.join(uploaded_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file tài liệu thô gốc.")
        
    return FileResponse(
        path=file_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename
    )

@app.post("/api/admin/upload")
async def upload_docx(
    file: UploadFile = File(...),
    law_name: Optional[str] = None,
    source_url: Optional[str] = None
):
    """Admin endpoint to upload and dynamically ingest raw Vietnamese legal DOCX publications."""
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận tệp tin định dạng Microsoft Word (.docx).")
    
    # Save the file temporarily
    temp_filename = f"temp_upload_{uuid.uuid4().hex}.docx"
    temp_path = os.path.join(generated_docs_dir, temp_filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run Ingestion
        ingestor = LegalDocumentIngestion()
        total_items = ingestor.ingest_document(temp_path, law_name_override=law_name, source_url=source_url)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if total_items == 0:
            raise ValueError("Không thể bóc tách được điều khoản nào từ văn bản thô. Vui lòng kiểm tra lại cấu trúc file Word.")
            
        # ChromaDB handles database updates dynamically in real-time
        
        return {
            "status": "success",
            "message": f"Nạp dữ liệu thành công! Đã bóc tách {total_items} điều khoản mới.",
            "law_name": file.filename.replace(".docx", "")
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Lỗi phân tách tài liệu: {str(e)}")

@app.post("/api/admin/import-vbpl")
async def import_vbpl_endpoint(
    doc_id: str = Body(..., embed=True)
):
    """Admin endpoint to pull multiple laws from vbpl.vn Gateway, parse them, compile physical .docx, and RAG-ingest."""
    import re
    import urllib.request
    import json
    
    # Extract all valid ID tokens (alphanumeric, hyphens, underscores)
    doc_ids = re.findall(r'[a-zA-Z0-9_\-]+', doc_id)
    if not doc_ids:
        raise HTTPException(status_code=400, detail="Không tìm thấy ID văn bản hợp lệ. Vui lòng kiểm tra lại định dạng nhập.")
        
    success_count = 0
    imported_laws = []
    total_new_items = 0
    errors = []
    
    ingestor = LegalDocumentIngestion()
    
    for d_id in doc_ids:
        url = f"https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{d_id}"
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                status_code = response.getcode()
                if status_code != 200:
                    errors.append(f"{d_id} (Lỗi kết nối Cổng HTTP {status_code})")
                    continue
                
                payload_data = response.read().decode('utf-8')
                payload = json.loads(payload_data)
                
                if not payload.get("success") or "data" not in payload:
                    message = payload.get("message", "Tài liệu không tồn tại.")
                    errors.append(f"{d_id} ({message})")
                    continue
                    
                # Run Ingestion
                total_items = ingestor.ingest_vbpl_payload(d_id, payload)
                success_count += 1
                total_new_items += total_items
                
                data = payload.get("data", {})
                title = data.get("title", f"Văn bản VBPL {d_id}")
                imported_laws.append(title)
                
        except Exception as e:
            errors.append(f"{d_id} ({str(e)})")
            
    if success_count == 0:
        error_msg = "; ".join(errors)
        raise HTTPException(status_code=500, detail=f"Không thể đồng bộ bất kỳ văn bản nào: {error_msg}")
        
    # ChromaDB handles database updates dynamically in real-time
    
    laws_joined = ", ".join(imported_laws)
    message = f"Nạp thành công {success_count} văn bản VBPL! Tổng cộng bóc tách {total_new_items} điều khoản mới."
    if errors:
        message += f" (Gặp lỗi {len(errors)} ID: {', '.join(errors)})"
      
    return {
        "status": "success",
        "message": message,
        "law_name": laws_joined,
        "doc_id": f"VBPL_BULK_{success_count}"
    }

@app.post("/api/chat", response_model=LegalAgentOutput)
async def chat_endpoint(
    session_id: str = Body(..., embed=True),
    query: str = Body(..., embed=True)
):
    """Processes chat queries using a stateful Google Antigravity SDK agent instance per session."""
    # 1. Retrieve or create stateful agent session
    if session_id not in active_agents:
        try:
            agent = create_legal_agent()
            # Enter the context manager asynchronously to initialize connections and hooks
            await agent.__aenter__()
            active_agents[session_id] = agent
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Không thể khởi tạo AI Agent: {str(e)}. Hãy chắc chắn bạn đã cấu hình GEMINI_API_KEY."
            )
            
    agent = active_agents[session_id]
    
    # 2. Query the agent and fetch structured output
    try:
        # Preprocess the query (PII Redaction + Slang Translation)
        # a. PII Redaction
        sanitized = query
        sanitized = re.sub(r'\b(0[35789]\d{8})\b', '[SỐ_ĐIỆN_THOẠI_ĐÃ_ẨN]', sanitized)
        sanitized = re.sub(r'\b(\d{12})\b', '[SỐ_CCCD_ĐÃ_ẨN]', sanitized)
        sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_ĐÃ_ẨN]', sanitized)
        
        # b. Slang Translation
        from agent import rag_engine
        processed_query = rag_engine.translate_slang(sanitized)
        
        response = await agent.chat(processed_query)
        raw_text = await response.text()
        
        # Self-healing structured output parsing
        structured_data = None
        
        # 1. Regex JSON block extraction (extremely robust against Markdown or conversational text wrapper)
        match = re.search(r"(\{[\s\S]*\})", raw_text)
        if match:
            try:
                from agent import LegalAgentOutput
                parsed_obj = LegalAgentOutput.model_validate_json(match.group(1))
                structured_data = parsed_obj.model_dump()
            except Exception as parse_err:
                print(f"Lỗi phân tích cú pháp JSON bằng Regex: {parse_err}")
                
        # 2. SDK structured_output fallback (only if current turn matches structured output state)
        if not structured_data:
            sdk_data = await response.structured_output()
            if sdk_data:
                # Only use SDK data if raw_text is "Finished" or empty (indicating current turn success)
                cleaned_text = raw_text.strip().replace("`", "").replace("json", "")
                if cleaned_text == "Finished" or not cleaned_text:
                    structured_data = sdk_data
                else:
                    # If model returned text, the SDK's structured_output is likely a stale previous turn
                    print("Bỏ qua dữ liệu SDK của lượt chat cũ.")
        
        if not structured_data:
            raise ValueError("Không nhận được dữ liệu cấu trúc hợp lệ từ Agent.")
            
        # structured_data is a dictionary matching LegalAgentOutput
        return structured_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Check if the error is due to an invalid API key
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "API key not found" in error_msg:
            raise HTTPException(
                status_code=401,
                detail="Khóa API Gemini (GEMINI_API_KEY) không hợp lệ hoặc thiếu. Vui lòng kiểm tra lại cấu hình."
            )
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi hệ thống trong luồng xử lý của Agent: {error_msg}"
        )

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Serves the generated .docx templates for download."""
    file_path = os.path.join(generated_docs_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy tập tin biểu mẫu yêu cầu.")
        
    return FileResponse(
        path=file_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename
    )

# Cleanup agents on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleans up all active agent connections to conserve server resources."""
    print("Shutting down API server. Cleaning up active agent sessions...")
    for session_id, agent in list(active_agents.items()):
        try:
            await agent.__aexit__(None, None, None)
        except Exception as e:
            print(f"Error closing agent session {session_id}: {e}")
    active_agents.clear()

# Mount frontend files at root (served after api routes)
frontend_dir = os.path.join(current_dir, "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"Warning: Frontend directory not found at {frontend_dir}. API server will only run backend endpoints.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8089, reload=False)
