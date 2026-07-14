import os
import json
import re
import pydantic
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from google.antigravity import Agent, LocalAgentConfig, ToolContext, types
from google.antigravity.hooks import hooks, policy

from rag_engine import LegalRAGEngine
from document_generator import LegalDocumentGenerator

# Load environment variables
load_dotenv()

# Initialize RAG engine and Doc generator
rag_engine = LegalRAGEngine()
doc_generator = LegalDocumentGenerator()

# 1. Pydantic Schemas for Structured Output
class Citation(pydantic.BaseModel):
    law_name: str = ""
    article: str = ""
    clause: str = ""
    point: str = ""
    exact_text: str = ""
    source_url: str = ""

class LegalAgentOutput(pydantic.BaseModel):
    output_type: str = "advice"       # "advice" | "clarification" | "document"
    # Fields for "advice"
    summary_answer: str = ""
    detailed_analysis: str = ""
    citations: List[Citation] = []
    # Fields for "clarification"
    clarification_question: str = ""
    clarification_options: List[str] = []
    # Fields for "document"
    download_url: str = ""
    document_message: str = ""
    
    confidence_score: float = 1.0
    disclaimer: str = "Tuyên bố miễn trừ trách nhiệm: Thông tin do Trợ lý AI cung cấp chỉ mang tính chất tham khảo cho doanh nghiệp SME và không thay thế cho dịch vụ tư vấn pháp lý chuyên nghiệp của Luật sư."

# 2. Custom Tools for the Agent
def search_vietnamese_laws(query: str, event_date_str: Optional[str] = None) -> str:
    """Tra cứu các văn bản pháp luật, nghị định, thông tư của Việt Nam liên quan đến câu hỏi.

    Args:
        query: Câu hỏi hoặc thuật ngữ pháp lý cần tra cứu.
        event_date_str: Ngày xảy ra sự việc (định dạng YYYY-MM-DD), dùng để lọc thời gian hiệu lực pháp lý (nếu có).
    """
    results = rag_engine.hybrid_search(query, event_date_str, top_k=3)
    if not results:
        return "Không tìm thấy điều luật hoặc nghị định nào liên quan trực tiếp đến yêu cầu."
    
    # Return formatted string for LLM reasoning
    formatted_results = []
    for r in results:
        item = f"Văn bản: {r['law_name']}\nTiêu đề: {r['title']}\nNội dung: {r['text']}\nHiệu lực từ: {r['effective_date']}\nTrạng thái: {r['status']}\nURL kiểm tra: {r.get('source_url', '')}"
        if "matched_child_clause" in r:
            item += f"\nKhoản khớp chi tiết: {r['matched_child_clause']}"
        formatted_results.append(item)
        
    return "\n===\n".join(formatted_results)

def generate_docx_document(
    template_type: str, 
    employee_name: str, 
    cccd: str, 
    position: str, 
    reason: str, 
    company_name: str
) -> str:
    """Tạo biểu mẫu quyết định hoặc thỏa thuận pháp lý chuẩn bằng file Word .docx và trả về đường dẫn tải xuống.

    Args:
        template_type: Loại biểu mẫu cần tạo. Chọn một trong các giá trị: "sa_thai" (Quyết định sa thải), "nda" (Thỏa thuận bảo mật thông tin).
        employee_name: Họ và tên của người lao động.
        cccd: Số CCCD của người lao động.
        position: Chức vụ hoặc bộ phận làm việc.
        reason: Lý do sa thải hoặc căn cứ thỏa thuận.
        company_name: Tên của công ty/doanh nghiệp SME.
    """
    data = {
        "company_name": company_name,
        "employee_name": employee_name,
        "cccd": cccd,
        "position": position,
        "reason": reason,
        "date_str": "ngày 28 tháng 05 năm 2026",
        "effective_date": "ngày ký quyết định"
    }
    
    try:
        if template_type == "sa_thai":
            fn = doc_generator.generate_dismissal_decision(data)
        elif template_type == "nda":
            fn = doc_generator.generate_nda(data)
        else:
            return "Loại biểu mẫu không hợp lệ. Vui lòng chọn 'sa_thai' hoặc 'nda'."
        
        # Format the relative URL that the frontend can call to download
        download_url = f"/api/download/{fn}"
        return json.dumps({
            "status": "success",
            "download_url": download_url,
            "message": f"Đã khởi tạo thành công biểu mẫu Word (.docx) cho người lao động {employee_name}."
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Không thể tạo tài liệu do lỗi: {str(e)}"
        }, ensure_ascii=False)

# 3. Google Antigravity Hooks
# Pre-Turn Hook: Validate user request (Safe Gate)
@hooks.pre_turn
async def preprocess_turn(prompt_text: str) -> types.HookResult:
    if not prompt_text or not prompt_text.strip():
        return types.HookResult(allow=False, message="Yêu cầu trống không hợp lệ.")
    return types.HookResult(allow=True)

# Helper functions for robust citation matching to prevent false hallucinations
def match_law_name(db_law: str, gen_law: str) -> bool:
    db_law = db_law.lower()
    gen_law = gen_law.lower()
    if gen_law in db_law or db_law in gen_law:
        return True
    subjects = ["hình sự", "lao động", "dân sự", "thương mại", "doanh nghiệp", "thư viện", "giáo dục", "an ninh mạng", "cảnh sát biển", "trồng trọt", "chăn nuôi", "đặc xá", "bảo vệ bí mật", "thể dục, thể thao", "đo đạc", "cạnh tranh", "lâm nghiệp", "trợ giúp pháp lý", "cảnh vệ", "bồi thường", "du lịch", "công nghệ", "ngoại thương", "tiếp cận thông tin", "trưng cầu ý dân", "tạm giữ", "giám sát", "kiểm toán", "tòa án", "hộ tịch", "hôn nhân", "nhập cảnh", "tiếp công dân", "cư trú", "hòa giải", "phòng, chống tham nhũng", "chứng khoán", "viên chức", "thi hành án", "trọng tài", "bưu chính"]
    for sub in subjects:
        if sub in gen_law and sub in db_law:
            return True
    gen_words = set(re.findall(r'\w{4,}', gen_law))
    db_words = set(re.findall(r'\w{4,}', db_law))
    if gen_words.intersection(db_words):
        return True
    return False

def match_article(db_art: str, gen_art: str) -> bool:
    db_art = db_art.lower()
    gen_art = gen_art.lower()
    if gen_art in db_art or db_art in gen_art:
        return True
    db_digits = "".join(re.findall(r'\d+', db_art))
    gen_digits = "".join(re.findall(r'\d+', gen_art))
    if db_digits and gen_digits and db_digits == gen_digits:
        return True
    return False

def match_exact_text(db_text: str, gen_text: str) -> bool:
    db_text = db_text.lower()
    gen_text = gen_text.lower()
    db_clean = re.sub(r'[^\w\d]', '', db_text)
    gen_clean = re.sub(r'[^\w\d]', '', gen_text)
    if not gen_clean:
        return True
    if gen_clean in db_clean or db_clean in gen_clean:
        return True
    if len(gen_clean) > 25:
        for i in range(len(gen_clean) - 24):
            sub = gen_clean[i:i+25]
            if sub in db_clean:
                return True
    gen_words = set(re.findall(r'\w{3,}', gen_text))
    db_words = set(re.findall(r'\w{3,}', db_text))
    if len(gen_words) > 0:
        overlap = gen_words.intersection(db_words)
        if len(overlap) / len(gen_words) >= 0.5:
            return True
    return False

# Post-Turn Hook: Validate citations to prevent Hallucination
@hooks.post_turn
async def postprocess_turn(response_content: str):
    # If the response is in JSON matching our schema, verify the citations
    try:
        data = json.loads(response_content)
        if "citations" in data and data["citations"]:
            for cit in data["citations"]:
                # Check if exact_text is empty or not in the db
                exact_text = cit.get("exact_text") or ""
                law_name = cit.get("law_name") or ""
                article = cit.get("article") or ""
                
                # If fields are missing/empty, ignore this particular citation safely
                if not exact_text or not law_name or not article:
                    continue
                
                # Query ChromaDB by article to find candidate matches
                found = False
                results = rag_engine.collection.get(where={"article": article})
                
                if results and results["metadatas"]:
                    for doc in results["metadatas"]:
                        if match_law_name(doc["law_name"], law_name):
                            if match_exact_text(doc["text"], exact_text):
                                found = True
                                break
                
                if not found:
                    # Halt turn if hallucinated citation detected
                    raise ValueError(f"Phát hiện trích dẫn giả mạo (Hallucination) tại: {article} - {law_name}")
    except json.JSONDecodeError:
        # Non-JSON responses are fine during normal interactive loops
        pass

# 4. Agent Initialization Factory
def create_legal_agent() -> Agent:
    api_key = os.getenv("GEMINI_API_KEY")
    
    config = LocalAgentConfig(
        api_key=api_key,
        model="gemini-3.1-flash-lite",
        response_schema=LegalAgentOutput,
        tools=[search_vietnamese_laws, generate_docx_document],
        hooks=[preprocess_turn, postprocess_turn],
        # policy.confirm_run_command() returns a list[Policy] bundle (deny run_command + allow rest).
        # It is also LocalAgentConfig's default — pass directly (no extra list wrap) or omit.
        policies=policy.confirm_run_command(),
        system_instructions=(
            "Bạn là trợ lý pháp lý AI chuyên sâu (AI Legal Assistant) hỗ trợ doanh nghiệp SME tại Việt Nam.\n"
            "Nhiệm vụ của bạn là tư vấn, tra cứu luật, xử lý tình huống lao động/thương mại và hỗ trợ tạo biểu mẫu.\n\n"
            "QUY TẮC PHẢN HỒI (BẮT BUỘC TUÂN THỦ):\n"
            "1. Bạn PHẢI sử dụng công cụ `search_vietnamese_laws` để tìm kiếm cơ sở pháp lý trước khi đưa ra bất kỳ lời khuyên nào.\n"
            "2. Đầu ra của bạn BẮT BUỘC phải luôn là định dạng JSON khớp hoàn toàn với cấu trúc của schema `LegalAgentOutput`. KHÔNG ĐƯỢC VIẾT TEXT TỰ DO NGOÀI JSON.\n"
            "3. Hướng dẫn chi tiết từng trường hợp của `LegalAgentOutput`:\n"
            "   - Trường hợp A: Quy trình làm rõ thông tin tương tác từng bước (Clarification Loop):\n"
            "     Nếu câu hỏi hoặc dữ kiện của người dùng quá mơ hồ, thiếu các thông tin cốt lõi để đưa ra lời khuyên pháp lý chuẩn xác (ví dụ: 'tôi muốn sa thải nhân viên', 'bị phạt hợp đồng',...):\n"
            "     + Thiết lập `output_type` là 'clarification'.\n"
            "     + Điền duy nhất MỘT câu hỏi làm rõ ngắn gọn vào trường `clarification_question` (không được hỏi dồn nhiều câu hỏi một lúc).\n"
            "     + BẮT BUỘC điền danh sách 3 đến 5 nút lựa chọn trả lời ngắn gọn vào trường `clarification_options` (Ví dụ: ['Hợp đồng thử việc', 'Hợp đồng lao động xác định thời hạn', 'Hợp đồng lao động không xác định thời hạn']). KHÔNG ĐƯỢC ĐỂ TRỐNG.\n"
            "     + BẮT BUỘC đặt tất cả các trường khác về giá trị rỗng/mặc định: `citations = []`, `summary_answer = \"\"`, `detailed_analysis = \"\"`, `download_url = \"\"`, `document_message = \"\"`.\n"
            "   - Trường hợp B: Nếu đã có đủ thông tin để đưa ra lời giải đáp pháp lý:\n"
            "     + Thiết lập `output_type` là 'advice'.\n"
            "     + Điền câu trả lời tóm tắt cực kỳ ngắn gọn vào trường `summary_answer`.\n"
            "     + Điền lập luận chi tiết từng bước vào trường `detailed_analysis` (hãy đan xen các trích dẫn dạng [Trích dẫn 1], [Trích dẫn 2]... để người dùng bấm vào xem luật).\n"
            "     + Điền mảng `citations` chứa chính xác văn bản gốc và thông tin được trả về từ công cụ tra cứu. Đảm bảo mỗi trích dẫn trong mảng có đầy đủ các trường: `law_name`, `article`, `clause`, `point`, `exact_text`, `source_url`.\n"
            "     + BẮT BUỘC đặt tất cả các trường khác về giá trị rỗng/mặc định: `clarification_question = \"\"`, `clarification_options = []`, `download_url = \"\"`, `document_message = \"\"`.\n"
            "   - Trường hợp C: Nếu người dùng yêu cầu tạo quyết định sa thải hoặc thỏa thuận NDA:\n"
            "     + Gọi công cụ `generate_docx_document` với các đối số tương ứng.\n"
            "     + Thiết lập `output_type` là 'document'.\n"
            "     + Điền `download_url` nhận được từ công cụ vào trường `download_url`.\n"
            "     + Điền thông điệp hướng dẫn/chúc mừng vào trường `document_message`.\n"
            "     + BẮT BUỘC đặt tất cả các trường khác về giá trị rỗng/mặc định: `clarification_question = \"\"`, `clarification_options = []`, `summary_answer = \"\"`, `detailed_analysis = \"\"`, `citations = []`.\n"
            "4. Tuyệt đối không tự bịa đặt điều luật hoặc trích dẫn các văn bản pháp luật không tồn tại trong kết quả tra cứu của công cụ. Nếu không tìm thấy luật, hãy ghi rõ và khuyến khích doanh nghiệp tham vấn luật sư chuyên nghiệp."
        )
    )
    return Agent(config)
