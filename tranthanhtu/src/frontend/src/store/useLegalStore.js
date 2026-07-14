import { create } from 'zustand'

const API_BASE = "http://127.0.0.1:8089";

// Generate or retrieve session ID
let initialSessionId = sessionStorage.getItem("legal_assistant_session_id");
if (!initialSessionId) {
  initialSessionId = "session_" + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
  sessionStorage.setItem("legal_assistant_session_id", initialSessionId);
}

// A lightweight Markdown-to-HTML parser function to render bold, italic, lists, and links properly.
function parseMarkdownToHtml(text) {
  if (!text) return "";
  
  let html = text;
  
  // 1. Escape HTML
  html = html
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
    
  // 2. Bold text **text** -> <strong>text</strong>
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  
  // 3. Italic text *text* or _text_ -> <em>text</em>
  html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
  html = html.replace(/_(.*?)_/g, "<em>$1</em>");
  
  // 4. Markdown links: [text](url) -> <a href="url" target="_blank" ...>text</a>
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="md-link">$1 <i class="fa-solid fa-arrow-up-right-from-square" style="font-size: 10px; margin-left: 2px;"></i></a>');

  // 5. Headers
  html = html.replace(/^####\s+(.*)$/gm, '<h5 class="md-h5">$1</h5>');
  html = html.replace(/^###\s+(.*)$/gm, '<h4 class="md-h4">$1</h4>');
  html = html.replace(/^##\s+(.*)$/gm, '<h3 class="md-h3">$1</h3>');

  // 6. Split and process lists and paragraphs
  const lines = html.split('\n');
  const processedLines = lines.map(line => {
    const trimmed = line.trim();
    // Bullet point: - text or * text
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      return `<li class="md-bullet-item">${trimmed.substring(2)}</li>`;
    }
    // Numbered list: 1. text
    const numListMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
    if (numListMatch) {
      return `<li class="md-number-item" value="${numListMatch[1]}">${numListMatch[2]}</li>`;
    }
    // Normal line
    return line;
  });
  
  let finalHtml = "";
  let inList = false;
  let listType = ""; // "ul" or "ol"
  
  processedLines.forEach(line => {
    const isLi = line.startsWith('<li');
    const isOl = line.includes('class="md-number-item"');
    
    if (isLi) {
      const currentType = isOl ? "ol" : "ul";
      if (!inList) {
        finalHtml += `<${currentType} class="md-list">`;
        inList = true;
        listType = currentType;
      } else if (listType !== currentType) {
        finalHtml += `</${listType}><${currentType} class="md-list">`;
        listType = currentType;
      }
      finalHtml += line;
    } else {
      if (inList) {
        finalHtml += `</${listType}>`;
        inList = false;
        listType = "";
      }
      if (line.trim()) {
        if (line.startsWith('<h3') || line.startsWith('<h4') || line.startsWith('<h5')) {
          finalHtml += line;
        } else {
          finalHtml += `<p class="detailed-analysis-paragraph">${line}</p>`;
        }
      }
    }
  });
  
  if (inList) {
    finalHtml += `</${listType}>`;
  }
  
  return finalHtml;
}

export const useLegalStore = create((set, get) => ({
  sessionId: initialSessionId,
  currentRole: "client",
  activeTab: "laws",
  messages: [
    {
      sender: "system",
      iconClass: "fa-solid fa-robot",
      htmlContent: `
        <h2>Xin kính chào Quý Doanh nghiệp! 👋</h2>
        <p>Tôi là <strong>Trợ lý Pháp lý AI</strong>, được tối ưu hóa đặc biệt bằng <strong>Google Antigravity SDK</strong> để hỗ trợ các doanh nghiệp SME tại Việt Nam tra cứu điều luật, phân tích tình huống thực tế và soạn thảo biểu mẫu hành chính nhanh chóng.</p>
        <p class="suggest-title">Bạn có thể hỏi tôi các câu hỏi như:</p>
        <div class="suggestion-chips">
          <button class="chip-btn" data-query="Nhân viên tự ý nghỉ việc 5 ngày liên tục không phép có bị sa thải không?">
            <i class="fa-solid fa-user-minus"></i> Sa thải tự ý nghỉ việc 5 ngày?
          </button>
          <button class="chip-btn" data-query="Tôi muốn sa thải nhân viên thử việc thì có phải đền bù không?">
            <i class="fa-solid fa-user-clock"></i> Sa thải nhân viên thử việc?
          </button>
          <button class="chip-btn" data-query="Hồ sơ và thủ tục thành lập chi nhánh mới cho công ty TNHH gồm những gì?">
            <i class="fa-solid fa-building-circle-check"></i> Thủ tục mở chi nhánh?
          </button>
        </div>
      `
    }
  ],
  isLoading: false,
  activeCitation: null,
  documents: [],
  clarification: null,
  uploadProgress: {
    visible: false,
    percent: 0,
    status: ""
  },
  ingestAlert: {
    visible: false,
    type: "success",
    message: ""
  },

  // Actions
  setRole: (role) => set({ currentRole: role, activeTab: role === "client" ? "laws" : get().activeTab }),
  setTab: (tab) => set({ activeTab: tab }),
  setLoading: (loading) => set({ isLoading: loading }),
  selectCitation: (citation) => set({ activeCitation: citation }),
  clearCitation: () => set({ activeCitation: null }),
  setUploadProgress: (progress) => set({ uploadProgress: { ...get().uploadProgress, ...progress } }),
  setIngestAlert: (alert) => set({ ingestAlert: { ...get().ingestAlert, ...alert } }),

  loadDocuments: async () => {
    try {
      const response = await fetch(`${API_BASE}/api/documents`);
      if (!response.ok) throw new Error("Không thể nạp danh sách hồ sơ.");
      const docs = await response.json();
      set({ documents: docs });
    } catch (e) {
      console.error("Lỗi khi tải hồ sơ luật:", e);
    }
  },

  addMessage: (sender, htmlContent, iconClass) => {
    set((state) => ({
      messages: [...state.messages, { sender, htmlContent, iconClass }]
    }));
  },

  deleteDocument: async (docId) => {
    if (!confirm("CẢNH BÁO NGUY HIỂM:\nBạn có chắc chắn muốn xóa văn bản pháp luật này khỏi hệ thống không?\n\nToàn bộ tệp tin thô vật lý và tất cả điều khoản thuộc văn bản này trong Vector/JSON DB của RAG Specialist sẽ bị xóa vĩnh viễn!")) {
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/api/documents/${docId}`, {
        method: "DELETE"
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Không thể thực hiện xóa.");
      }
      
      const res = await response.json();
      get().addMessage(
        "assistant",
        `<p><i class="fa-solid fa-trash-can" style="color: #ef4444;"></i> <strong>Purge Complete:</strong> Hệ thống RAG vừa xóa toàn bộ điều khoản và tệp thô của tài liệu có mã <strong>${docId}</strong> thành công.</p>
         <p>Cơ sở dữ liệu của Agent đã được thu hẹp về trạng thái cũ thời gian thực.</p>`,
        "fa-solid fa-robot"
      );
      
      get().loadDocuments();
    } catch (e) {
      alert("Lỗi khi thực hiện xóa tài liệu: " + e.message);
    }
  },

  submitQuery: async (queryText) => {
    if (!queryText.trim()) return;

    // 1. Add User message
    get().addMessage("user", `<p>${queryText}</p>`, "fa-solid fa-user");
    set({ isLoading: true, clarification: null });

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          session_id: get().sessionId,
          query: queryText
        })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Có lỗi xảy ra trên hệ thống.");
      }

      const data = await response.json(); // LegalAgentOutput structure
      set({ isLoading: false });

      if (data.output_type === "clarification") {
        const parsedQuestion = parseMarkdownToHtml(data.clarification_question);
        get().addMessage("assistant", parsedQuestion, "fa-solid fa-circle-question");
        set({
          clarification: {
            question: data.clarification_question,
            options: data.clarification_options || []
          }
        });
      } else if (data.output_type === "advice") {
        let htmlContent = `
          <h3 class="summary-answer-title"><i class="fa-solid fa-circle-check text-emerald"></i> Tóm tắt giải đáp:</h3>
          <p class="summary-answer-text">${data.summary_answer}</p>
          
          <h3 class="analysis-title"><i class="fa-solid fa-bars-staggered"></i> Phân tích tình huống chi tiết:</h3>
          <div class="detailed-analysis">
        `;

        // Format citations in text (parsing Markdown structure first)
        let formattedText = parseMarkdownToHtml(data.detailed_analysis);
        if (data.citations && data.citations.length > 0) {
          data.citations.forEach((cit, idx) => {
            const citationId = `cit_key_${idx}_${Date.now()}`;
            // Cache citation global mapping in window context for easy HTML click binding
            window.activeCitations = window.activeCitations || {};
            window.activeCitations[citationId] = cit;

            const markers = [
              `[Trích dẫn ${idx + 1}]`,
              `[trích dẫn ${idx + 1}]`,
              `[${idx + 1}]`
            ];
            markers.forEach(marker => {
              const replacement = `<a class="text-citation" data-citation-id="${citationId}" href="javascript:void(0)"><i class="fa-solid fa-file-shield"></i> ${cit.article}</a>`;
              while (formattedText.includes(marker)) {
                formattedText = formattedText.replace(marker, replacement);
              }
            });
          });
        }
        
        htmlContent += formattedText;
        htmlContent += `</div>`;

        // Citations badges
        if (data.citations && data.citations.length > 0) {
          htmlContent += `<div class="citation-group">`;
          data.citations.forEach((cit, idx) => {
            const citationId = `cit_key_${idx}_${Date.now()}`;
            window.activeCitations = window.activeCitations || {};
            window.activeCitations[citationId] = cit;
            htmlContent += `
              <button class="citation-tag" data-citation-id="${citationId}">
                <i class="fa-solid fa-file-signature"></i> ${cit.law_name} - ${cit.article}
              </button>
            `;
          });
          htmlContent += `</div>`;
        }

        // Disclaimer
        htmlContent += `
          <div class="disclaimer-box">
            <i class="fa-solid fa-circle-info"></i> ${data.disclaimer}
          </div>
        `;

        get().addMessage("assistant", htmlContent, "fa-solid fa-robot");
      } else if (data.output_type === "document") {
        const htmlContent = `
          <p>${data.document_message}</p>
          <div class="doc-card">
            <div class="doc-info">
              <i class="fa-solid fa-file-word"></i>
              <div>
                <div class="doc-name">Biểu mẫu Pháp lý (.docx)</div>
                <div class="doc-desc">Đã điền đầy đủ thông tin chuẩn xác theo nội quy lao động và quy định pháp luật Việt Nam.</div>
              </div>
            </div>
            <a href="${API_BASE}${data.download_url}" class="download-btn">
              <i class="fa-solid fa-download"></i> Tải biểu mẫu ngay
            </a>
          </div>
        `;
        get().addMessage("assistant", htmlContent, "fa-solid fa-robot");
      }

    } catch (e) {
      set({ isLoading: false });
      get().addMessage(
        "assistant",
        `<div class="disclaimer-box" style="border-left-color: #ef4444; background: rgba(239, 68, 68, 0.03); color: #fca5a5;">
          <i class="fa-solid fa-triangle-exclamation"></i> <strong>Lỗi:</strong> ${e.message}
          <br><br>Hãy chắc chắn bạn đã khởi chạy API server FastAPI (\`http://127.0.0.1:8089\`) và cấu hình đúng API Key trong tệp \`backend/.env\`.
        </div>`,
        "fa-solid fa-triangle-exclamation"
      );
    }
  },

  importVbpl: async (docId) => {
    if (!docId.trim()) return;

    set({
      uploadProgress: { visible: true, percent: 20, status: "Đang kết nối Cổng chính phủ vbpl.vn..." },
      ingestAlert: { visible: false, type: "success", message: "" }
    });

    const progressInterval = setInterval(() => {
      const cur = get().uploadProgress.percent;
      if (cur < 90) {
        set({
          uploadProgress: {
            visible: true,
            percent: cur + 10,
            status: cur > 40 ? "Đang tải dữ liệu XML/HTML gốc của văn bản..." : "Đang bóc tách cấu trúc Điều/Khoản/Điểm..."
          }
        });
      }
    }, 400);

    try {
      const response = await fetch(`${API_BASE}/api/admin/import-vbpl`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          doc_id: docId
        })
      });

      clearInterval(progressInterval);
      set({ uploadProgress: { visible: true, percent: 100, status: "Đồng bộ hoàn tất!" } });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Không thể đồng bộ từ VBPL.");
      }

      const resData = await response.json();
      
      setTimeout(() => {
        set({
          uploadProgress: { visible: false, percent: 0, status: "" },
          ingestAlert: { visible: true, type: "success", message: `${resData.message} Hệ thống AI đã học luật thô thành công.` }
        });
        
        get().addMessage(
          "assistant",
          `<p><i class="fa-solid fa-circle-check text-emerald"></i> <strong>Học luật thô thành công từ vbpl.vn!</strong> Tôi vừa tải toàn văn và bóc tách cấu trúc tài liệu của văn bản <strong>${resData.law_name}</strong> (Mã: VBPL_${docId}) thành công.</p>
           <p>Cơ sở dữ liệu RAG đã tự động cập nhật. Bạn có thể đặt bất kỳ câu hỏi nào về văn bản này ngay bây giờ!</p>`,
          "fa-solid fa-robot"
        );
        
        get().loadDocuments();
      }, 600);

    } catch (e) {
      clearInterval(progressInterval);
      set({
        uploadProgress: { visible: false, percent: 0, status: "" },
        ingestAlert: { visible: true, type: "error", message: `Lỗi nạp văn bản từ VBPL: ${e.message}` }
      });
    }
  },

  uploadDocx: async (file, lawName, sourceUrl) => {
    if (!file) return;

    set({
      uploadProgress: { visible: true, percent: 20, status: "Đang tải tệp tin lên server..." },
      ingestAlert: { visible: false, type: "success", message: "" }
    });

    const formData = new FormData();
    formData.append("file", file);
    if (lawName) formData.append("law_name", lawName);
    if (sourceUrl) formData.append("source_url", sourceUrl);

    const progressInterval = setInterval(() => {
      const cur = get().uploadProgress.percent;
      if (cur < 85) {
        set({
          uploadProgress: {
            visible: true,
            percent: cur + 15,
            status: cur > 40 ? "Đang bóc tách cấu trúc Điều/Khoản/Điểm..." : "Đang thiết lập liên kết cha-con..."
          }
        });
      }
    }, 300);

    try {
      const response = await fetch(`${API_BASE}/api/admin/upload`, {
        method: "POST",
        body: formData
      });

      clearInterval(progressInterval);
      set({ uploadProgress: { visible: true, percent: 100, status: "Nạp dữ liệu hoàn tất!" } });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Không thể phân tách dữ liệu.");
      }

      const resData = await response.json();
      
      setTimeout(() => {
        set({
          uploadProgress: { visible: false, percent: 0, status: "" },
          ingestAlert: { visible: true, type: "success", message: `${resData.message} Hệ thống AI đã học luật thô thành công.` }
        });
        
        get().addMessage(
          "assistant",
          `<p><i class="fa-solid fa-circle-check text-emerald"></i> <strong>Học luật thô thành công!</strong> Tôi vừa bóc tách cấu trúc tài liệu Word của văn bản <strong>${resData.law_name}</strong> và tự động cập nhật vào RAG.</p>
           <p>Bạn có thể đặt bất kỳ câu hỏi nào liên quan đến tài liệu này để kiểm chứng khả năng cập nhật thời gian thực của tôi.</p>`,
          "fa-solid fa-robot"
        );
        
        get().loadDocuments();
      }, 600);

    } catch (e) {
      clearInterval(progressInterval);
      set({
        uploadProgress: { visible: false, percent: 0, status: "" },
        ingestAlert: { visible: true, type: "error", message: `Lỗi nạp văn bản: ${e.message}` }
      });
    }
  }
}))
