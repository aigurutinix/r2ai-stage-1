// Base API URL for backend server
const API_BASE = "http://127.0.0.1:8089";

// Generate or retrieve a persistent session ID for the user's tab
let sessionId = sessionStorage.getItem("legal_assistant_session_id");
if (!sessionId) {
    sessionId = "session_" + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
    sessionStorage.setItem("legal_assistant_session_id", sessionId);
}

// DOM Elements
const chatThread = document.getElementById("chat-thread");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const agentLoader = document.getElementById("agent-loader");
const sessionStatus = document.getElementById("session-status");

// Clarification Panel elements
const clarificationPanel = document.getElementById("clarification-panel");
const clarifyQuestionText = document.getElementById("clarify-question-text");
const clarifyOptionsContainer = document.getElementById("clarify-options-container");

// Side Verification Panel elements
const verificationPanel = document.getElementById("verification-panel");
const sidePlaceholder = document.getElementById("side-placeholder");
const lawCard = document.getElementById("law-card");
const lawStatusBadge = document.getElementById("law-status-badge");
const lawTitleText = document.getElementById("law-title-text");
const lawNameText = document.getElementById("law-name-text");
const lawDateText = document.getElementById("law-date-text");
const lawTextContent = document.getElementById("law-text-content");
const clauseHighlightContainer = document.getElementById("clause-highlight-container");
const lawHighlightText = document.getElementById("law-highlight-text");
const lawExternalLink = document.getElementById("law-external-link");
const closePanelBtn = document.getElementById("close-panel-btn");

// Global store to hold retrieved citations for active message session
let activeCitationsMap = {};

// Auto scroll thread to bottom
function scrollToBottom() {
    chatThread.scrollTop = chatThread.scrollHeight;
}

// Format date nicely
function formatDate(dateStr) {
    if (!dateStr) return "Chưa cập nhật";
    try {
        const parts = dateStr.split('-');
        if (parts.length === 3) {
            return `${parts[2]}/${parts[1]}/${parts[0]}`;
        }
        return dateStr;
    } catch (e) {
        return dateStr;
    }
}

// Escape HTML utility to prevent script injection
function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// 1. Send Suggestion (Triggered by welcome chips)
function sendSuggestion(text) {
    userInput.value = text;
    chatForm.dispatchEvent(new Event("submit"));
}

// 2. Open Verification Side Panel
function displayVerificationLaw(citation) {
    // Hide placeholder, show card
    sidePlaceholder.style.display = "none";
    lawCard.style.display = "block";
    
    // Fill text values
    lawNameText.textContent = citation.law_name;
    lawTitleText.textContent = `${citation.article} ${citation.clause ? "- " + citation.clause : ""} ${citation.point ? "- " + citation.point : ""}`;
    lawDateText.textContent = citation.effective_date ? formatDate(citation.effective_date) : "Đang hiệu lực";
    
    // Status text and style
    lawStatusBadge.textContent = citation.status === "expired" ? "Hết hiệu lực" : "Còn hiệu lực";
    if (citation.status === "expired") {
        lawStatusBadge.style.background = "rgba(239, 68, 68, 0.15)";
        lawStatusBadge.style.color = "#ef4444";
        lawStatusBadge.style.borderColor = "rgba(239, 68, 68, 0.3)";
    } else {
        lawStatusBadge.style.background = "rgba(16, 185, 129, 0.15)";
        lawStatusBadge.style.color = "var(--accent-emerald)";
        lawStatusBadge.style.borderColor = "rgba(16, 185, 129, 0.3)";
    }

    // Set full law text (or the exact text matched)
    lawTextContent.innerHTML = `<p>${escapeHtml(citation.exact_text)}</p>`;

    // If there is child clause matching, show it highlighted
    if (citation.matched_child_clause) {
        clauseHighlightContainer.style.display = "block";
        lawHighlightText.textContent = citation.matched_child_clause;
    } else {
        clauseHighlightContainer.style.display = "none";
    }

    // Official external link
    if (citation.source_url) {
        lawExternalLink.style.display = "flex";
        lawExternalLink.href = citation.source_url;
    } else {
        lawExternalLink.style.display = "none";
    }

    // Open side panel drawer if on mobile viewport
    verificationPanel.classList.add("open");
}

// 3. Render Message in Thread
function appendMessage(sender, htmlContent, iconClass) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender === "user" ? "user-msg" : "assistant-msg");
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="${iconClass}"></i>
        </div>
        <div class="message-content">
            ${htmlContent}
        </div>
    `;
    
    chatThread.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

// 4. Parse text citations and turn [Trích dẫn X] into clickable HTML links
function parseCitationsText(text, citations) {
    if (!citations || citations.length === 0) return text;
    
    let formattedText = escapeHtml(text);
    
    // Weave citations inside the text based on matches like [Trích dẫn X] or [1], [2]
    citations.forEach((cit, idx) => {
        const citationId = `cit_key_${idx}_${Date.now()}`;
        activeCitationsMap[citationId] = cit;
        
        // Match markers like [Trích dẫn 1] or [1]
        const markers = [
            `[Trích dẫn ${idx + 1}]`,
            `[trích dẫn ${idx + 1}]`,
            `[${idx + 1}]`
        ];
        
        markers.forEach(marker => {
            const escapedMarker = escapeHtml(marker);
            // Replace marker with clickable custom HTML tag
            const replacement = `<a class="text-citation" data-citation-id="${citationId}" href="javascript:void(0)"><i class="fa-solid fa-file-shield"></i> ${cit.article}</a>`;
            // Keep replacing all occurrences
            while (formattedText.includes(escapedMarker)) {
                formattedText = formattedText.replace(escapedMarker, replacement);
            }
        });
    });

    // Replace newlines with HTML paragraphs for beautiful clean text structure
    formattedText = formattedText.split('\n\n').map(p => `<p class="detailed-analysis-paragraph">${p}</p>`).join('');
    return formattedText;
}

// Handle dynamic click events on embedded citation elements
document.addEventListener("click", function(event) {
    const citationTag = event.target.closest(".text-citation") || event.target.closest(".citation-tag");
    if (citationTag) {
        const citationId = citationTag.getAttribute("data-citation-id");
        const citationData = activeCitationsMap[citationId];
        if (citationData) {
            displayVerificationLaw(citationData);
        }
    }
});

// Close Mobile Side Panel Drawer
closePanelBtn.addEventListener("click", () => {
    verificationPanel.classList.remove("open");
});

// 5. Submit Message to FastAPI Server
chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const query = userInput.value.trim();
    if (!query) return;
    
    // Clear input & disable UI during inference
    userInput.value = "";
    userInput.disabled = true;
    sendBtn.disabled = true;
    
    // Hide clarification panel while waiting
    clarificationPanel.style.display = "none";
    
    // 1. Add User Message
    appendMessage("user", `<p>${escapeHtml(query)}</p>`, "fa-solid fa-user");
    
    // 2. Show agent thinking spinner
    agentLoader.style.display = "flex";
    sessionStatus.textContent = "Agent đang tra cứu...";
    
    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                session_id: sessionId,
                query: query
            })
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Có lỗi xảy ra trên hệ thống.");
        }
        
        const data = await response.json(); // LegalAgentOutput Pydantic response
        
        // Hide spinner
        agentLoader.style.display = "none";
        sessionStatus.textContent = "Agent Sẵn sàng";
        
        // 3. Render Agent Response according to Output Type
        if (data.output_type === "clarification") {
            // A. Clarification Loop Flow
            appendMessage("assistant", `<p>${escapeHtml(data.clarification_question)}</p>`, "fa-solid fa-circle-question");
            
            // Set clarification buttons
            clarifyQuestionText.textContent = "Làm rõ thông tin:";
            clarifyOptionsContainer.innerHTML = "";
            
            data.clarification_options.forEach(opt => {
                const btn = document.createElement("button");
                btn.classList.add("clarify-btn");
                btn.textContent = opt;
                btn.onclick = () => {
                    sendSuggestion(opt);
                };
                clarifyOptionsContainer.appendChild(btn);
            });
            
            clarificationPanel.style.display = "block";
            
        } else if (data.output_type === "advice") {
            // B. Legal Advice Flow
            let htmlContent = `
                <h3 class="summary-answer-title"><i class="fa-solid fa-circle-check text-emerald"></i> Tóm tắt giải đáp:</h3>
                <p class="summary-answer-text">${escapeHtml(data.summary_answer)}</p>
                
                <h3 class="analysis-title"><i class="fa-solid fa-bars-staggered"></i> Phân tích tình huống chi tiết:</h3>
                <div class="detailed-analysis">
                    ${parseCitationsText(data.detailed_analysis, data.citations)}
                </div>
            `;
            
            // Append citations tags at bottom
            if (data.citations && data.citations.length > 0) {
                htmlContent += `<div class="citation-group">`;
                data.citations.forEach((cit, idx) => {
                    const citationId = `cit_key_${idx}_${Date.now()}`;
                    activeCitationsMap[citationId] = cit;
                    htmlContent += `
                        <button class="citation-tag" data-citation-id="${citationId}">
                            <i class="fa-solid fa-file-signature"></i> ${cit.law_name} - ${cit.article}
                        </button>
                    `;
                });
                htmlContent += `</div>`;
            }
            
            // Add disclaimer
            htmlContent += `
                <div class="disclaimer-box">
                    <i class="fa-solid fa-circle-info"></i> ${escapeHtml(data.disclaimer)}
                </div>
            `;
            
            appendMessage("assistant", htmlContent, "fa-solid fa-robot");
            
        } else if (data.output_type === "document") {
            // C. Document Generation Flow
            let htmlContent = `
                <p>${escapeHtml(data.document_message)}</p>
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
            appendMessage("assistant", htmlContent, "fa-solid fa-robot");
        }
        
    } catch (error) {
        agentLoader.style.display = "none";
        sessionStatus.textContent = "Có lỗi xảy ra";
        
        // Append error bubble
        appendMessage(
            "assistant", 
            `<div class="disclaimer-box" style="border-left-color: #ef4444; background: rgba(239, 68, 68, 0.03); color: #fca5a5;">
                <i class="fa-solid fa-triangle-exclamation"></i> <strong>Lỗi:</strong> ${escapeHtml(error.message)}
                <br><br>Hãy chắc chắn bạn đã chạy API server FastAPI (\`python main.py\`) và cấu hình đúng API Key trong tệp \`backend/.env\`.
            </div>`,
            "fa-solid fa-triangle-exclamation"
        );
    } finally {
        // Re-enable input UI
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
        scrollToBottom();
    }
});

// Setup initial focus
userInput.focus();
scrollToBottom();
console.log("AI Legal Assistant frontend started. Session ID:", sessionId);


// ==========================================
// PHASE 5: TABS & DYNAMIC UPLOAD INGESTION
// ==========================================

// 1. Tab Switching Logic
const tabLawsBtn = document.getElementById("tab-laws-btn");
const tabAdminBtn = document.getElementById("tab-admin-btn");
const lawsTabContent = document.getElementById("laws-tab-content");
const adminTabContent = document.getElementById("admin-tab-content");

tabLawsBtn.addEventListener("click", () => {
    tabLawsBtn.classList.add("active");
    tabAdminBtn.classList.remove("active");
    lawsTabContent.style.display = "block";
    adminTabContent.style.display = "none";
});

tabAdminBtn.addEventListener("click", () => {
    tabAdminBtn.classList.add("active");
    tabLawsBtn.classList.remove("active");
    adminTabContent.style.display = "block";
    lawsTabContent.style.display = "none";
});

// 2. Drag & Drop Upload Zone Handlers
const uploadDropzone = document.getElementById("upload-dropzone");
const fileInput = document.getElementById("file-file-input");
const fileNameIndicator = document.getElementById("file-name-indicator");
const dropzoneText = document.getElementById("dropzone-text");

let selectedFile = null;

// Trigger hidden file picker on click
uploadDropzone.addEventListener("click", () => {
    fileInput.click();
});

// Handle file picked manually
fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
    }
});

// Handle Drag Events
uploadDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadDropzone.classList.add("dragover");
});

uploadDropzone.addEventListener("dragleave", () => {
    uploadDropzone.classList.remove("dragover");
});

uploadDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadDropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
    }
});

function handleFileSelection(file) {
    if (!file.name.endsWith(".docx")) {
        showIngestAlert("error", "Chỉ hỗ trợ nạp tệp tin định dạng Microsoft Word (.docx).");
        selectedFile = null;
        fileNameIndicator.textContent = "";
        dropzoneText.style.display = "block";
        return;
    }
    selectedFile = file;
    fileNameIndicator.innerHTML = `<i class="fa-solid fa-file-circle-check"></i> ${escapeHtml(file.name)} (${(file.size / 1024).toFixed(1)} KB)`;
    dropzoneText.style.display = "none";
    hideIngestAlert();
}

// 3. Upload & Ingest Submit Handler
const uploadSubmitBtn = document.getElementById("upload-submit-btn");
const adminLawNameInput = document.getElementById("admin-law-name");
const adminSourceUrlInput = document.getElementById("admin-source-url");
const progressContainer = document.getElementById("upload-progress-container");
const progressBarFill = document.getElementById("progress-bar-fill");
const progressStatusText = document.getElementById("progress-status-text");
const ingestStatusAlert = document.getElementById("ingest-status-alert");

function showIngestAlert(type, message) {
    ingestStatusAlert.className = `ingest-status-alert ${type}`;
    ingestStatusAlert.innerHTML = type === "success" 
        ? `<i class="fa-solid fa-circle-check"></i> ${message}`
        : `<i class="fa-solid fa-triangle-exclamation"></i> ${message}`;
    ingestStatusAlert.style.display = "block";
}

function hideIngestAlert() {
    ingestStatusAlert.style.display = "none";
}

uploadSubmitBtn.addEventListener("click", async () => {
    if (!selectedFile) {
        showIngestAlert("error", "Vui lòng kéo thả hoặc chọn một tệp tin .docx pháp luật trước.");
        return;
    }

    // Prepare FormData
    const formData = new FormData();
    formData.append("file", selectedFile);
    
    const lawNameOverride = adminLawNameInput.value.trim();
    if (lawNameOverride) {
        formData.append("law_name", lawNameOverride);
    }
    
    const sourceUrl = adminSourceUrlInput.value.trim();
    if (sourceUrl) {
        formData.append("source_url", sourceUrl);
    }

    // Disable UI during upload
    uploadSubmitBtn.disabled = true;
    uploadDropzone.style.pointerEvents = "none";
    hideIngestAlert();
    
    // Show and animate progress
    progressContainer.style.display = "block";
    progressBarFill.style.width = "20%";
    progressStatusText.textContent = "Đang tải tệp tin lên server...";
    
    try {
        // Animate fake progress to make UX extremely sleek
        const progressInterval = setInterval(() => {
            let curWidth = parseInt(progressBarFill.style.width);
            if (curWidth < 85) {
                progressBarFill.style.width = (curWidth + 15) + "%";
                if (curWidth > 40) {
                    progressStatusText.textContent = "Đang bóc tách cấu trúc Điều/Khoản/Điểm...";
                }
                if (curWidth > 70) {
                    progressStatusText.textContent = "Đang đánh chỉ mục và thiết lập liên kết cha-con...";
                }
            }
        }, 300);

        const response = await fetch(`${API_BASE}/api/admin/upload`, {
            method: "POST",
            body: formData
        });

        clearInterval(progressInterval);
        progressBarFill.style.width = "100%";
        progressStatusText.textContent = "Nạp dữ liệu hoàn tất!";

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Không thể phân tách dữ liệu.");
        }

        const resData = await response.json();
        
        setTimeout(() => {
            progressContainer.style.display = "none";
            showIngestAlert("success", `${resData.message} Hệ thống AI đã học luật thô thành công. Doanh nghiệp có thể đặt câu hỏi về văn bản này ngay!`);
            
            // Re-enable UI
            uploadSubmitBtn.disabled = false;
            uploadDropzone.style.pointerEvents = "auto";
            
            // Add a clean announcement bubble in the chat thread to guide the user!
            appendMessage(
                "assistant",
                `<p><i class="fa-solid fa-circle-check text-emerald"></i> <strong>Học luật thô thành công!</strong> Tôi vừa bóc tách cấu trúc tài liệu Word của văn bản <strong>${escapeHtml(resData.law_name)}</strong> và tự động cập nhật vào RAG.</p>
                 <p>Bạn có thể đặt bất kỳ câu hỏi nào liên quan đến tài liệu này để kiểm chứng khả năng cập nhật thời gian thực của tôi.</p>`,
                "fa-solid fa-robot"
            );
            
            // Reset upload inputs
            selectedFile = null;
            fileNameIndicator.textContent = "";
            dropzoneText.style.display = "block";
            adminLawNameInput.value = "";
            adminSourceUrlInput.value = "";
            fileInput.value = "";

            // Reload sidebar list
            loadDocuments();
        }, 600);

    } catch (error) {
        progressContainer.style.display = "none";
        showIngestAlert("error", `Lỗi nạp văn bản: ${error.message}`);
        
        uploadSubmitBtn.disabled = false;
        uploadDropzone.style.pointerEvents = "auto";
    }
});

// 3b. VBPL Import Submit Handler
const adminVbplIdInput = document.getElementById("admin-vbpl-id");
const adminVbplSubmitBtn = document.getElementById("admin-vbpl-submit-btn");

adminVbplSubmitBtn.addEventListener("click", async () => {
    const docId = adminVbplIdInput.value.trim();
    if (!docId) {
        showIngestAlert("error", "Vui lòng nhập ID số văn bản pháp lý từ trang vbpl.vn (ví dụ: 139877).");
        return;
    }

    // Disable UI during sync
    adminVbplSubmitBtn.disabled = true;
    hideIngestAlert();
    
    // Show progress
    progressContainer.style.display = "block";
    progressBarFill.style.width = "20%";
    progressStatusText.textContent = "Đang kết nối Cổng chính phủ vbpl.vn...";
    
    try {
        // Animate fake progress to make UX extremely sleek
        const progressInterval = setInterval(() => {
            let curWidth = parseInt(progressBarFill.style.width);
            if (curWidth < 90) {
                progressBarFill.style.width = (curWidth + 10) + "%";
                if (curWidth > 40) {
                    progressStatusText.textContent = "Đang tải dữ liệu XML/HTML gốc của văn bản...";
                }
                if (curWidth > 70) {
                    progressStatusText.textContent = "Đang bóc tách cấu trúc Điều/Khoản/Điểm & tự dựng file Word vật lý...";
                }
            }
        }, 400);

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
        progressBarFill.style.width = "100%";
        progressStatusText.textContent = "Đồng bộ hoàn tất!";

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Không thể đồng bộ dữ liệu từ VBPL.");
        }

        const resData = await response.json();
        
        setTimeout(() => {
            progressContainer.style.display = "none";
            showIngestAlert("success", `${resData.message} Hệ thống AI đã học luật thô thành công. Doanh nghiệp có thể đặt câu hỏi về văn bản này ngay!`);
            
            // Re-enable UI
            adminVbplSubmitBtn.disabled = false;
            
            // Add a clean announcement bubble in the chat thread to guide the user!
            appendMessage(
                "assistant",
                `<p><i class="fa-solid fa-circle-check text-emerald"></i> <strong>Học luật thô thành công từ vbpl.vn!</strong> Tôi vừa tải toàn văn và bóc tách cấu trúc tài liệu của văn bản <strong>${escapeHtml(resData.law_name)}</strong> (Mã: VBPL_${docId}) thành công.</p>
                 <p>Cơ sở dữ liệu RAG đã tự động cập nhật. Bạn có thể đặt bất kỳ câu hỏi nào về văn bản này ngay bây giờ!</p>`,
                "fa-solid fa-robot"
            );
            
            // Reset upload inputs
            adminVbplIdInput.value = "";

            // Reload sidebar list
            loadDocuments();
        }, 600);

    } catch (error) {
        progressContainer.style.display = "none";
        showIngestAlert("error", `Lỗi nạp văn bản từ VBPL: ${error.message}`);
        adminVbplSubmitBtn.disabled = false;
    }
});


// ==========================================
// PHASE 6: ROLE-BASED ACCESS CONTROL (RBAC)
// ==========================================

// 1. Role switcher elements
const roleClientBtn = document.getElementById("role-client-btn");
const roleAdminBtn = document.getElementById("role-admin-btn");
const legalLibrarySection = document.getElementById("legal-library-section");
const clientDocList = document.getElementById("client-doc-list");
const adminDocList = document.getElementById("admin-doc-list");
const backToLibraryBtn = document.getElementById("back-to-library-btn");

let currentRole = "client"; // default role

// Toggle view to Client role
roleClientBtn.addEventListener("click", () => {
    currentRole = "client";
    roleClientBtn.classList.add("active");
    roleAdminBtn.classList.remove("active");

    // Clients cannot see Admin tab button
    tabAdminBtn.style.display = "none";
    // Force view laws tab
    tabLawsBtn.click();

    // Show client library
    legalLibrarySection.style.display = "flex";
    loadDocuments();
});

// Toggle view to Admin role
roleAdminBtn.addEventListener("click", () => {
    currentRole = "admin";
    roleAdminBtn.classList.add("active");
    roleClientBtn.classList.remove("active");

    // Admins can see the Admin upload tab button
    tabAdminBtn.style.display = "flex";

    // Refresh lists
    loadDocuments();
});

// 2. Back to Library Button Click
backToLibraryBtn.addEventListener("click", () => {
    lawCard.style.display = "none";
    sidePlaceholder.style.display = "flex";
    if (currentRole === "client") {
        legalLibrarySection.style.display = "flex";
    }
});

// Hide Document Library when verified law card is showing
const originalDisplayVerificationLaw = displayVerificationLaw;
displayVerificationLaw = function(citation) {
    legalLibrarySection.style.display = "none";
    originalDisplayVerificationLaw(citation);
};

// 3. Dynamic Registry Loader
async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE}/api/documents`);
        if (!response.ok) throw new Error("Không thể nạp danh sách văn bản.");
        
        const docs = await response.json();
        
        // A. Populate Client View (Read-only list with Download buttons)
        clientDocList.innerHTML = "";
        if (docs.length === 0) {
            clientDocList.innerHTML = `<div class="doc-item-name" style="color: var(--text-muted); font-size:12px; padding: 10px;">Thư viện trống. Chưa có văn bản nào được nạp.</div>`;
        } else {
            docs.forEach(doc => {
                const item = document.createElement("div");
                item.classList.add("doc-item");
                item.innerHTML = `
                    <div class="doc-item-left">
                        <i class="fa-solid fa-file-lines"></i>
                        <div class="doc-item-info">
                            <div class="doc-item-name" title="${escapeHtml(doc.law_name)}">${escapeHtml(doc.law_name)}</div>
                            <div class="doc-item-meta">${doc.total_articles} Điều • ${doc.file_size_kb} KB</div>
                        </div>
                    </div>
                    <div class="doc-item-actions">
                        <a href="${API_BASE}/api/documents/download/${doc.filename}" class="doc-action-btn" title="Tải xuống tài liệu gốc .docx">
                            <i class="fa-solid fa-download"></i>
                        </a>
                    </div>
                `;
                clientDocList.appendChild(item);
            });
        }

        // B. Populate Admin View (Manage list with Delete buttons)
        adminDocList.innerHTML = "";
        if (docs.length === 0) {
            adminDocList.innerHTML = `<div class="doc-item-name" style="color: var(--text-muted); font-size:12px; padding: 10px;">Chưa có tài liệu nào trong kho.</div>`;
        } else {
            docs.forEach(doc => {
                const item = document.createElement("div");
                item.classList.add("doc-item");
                item.innerHTML = `
                    <div class="doc-item-left">
                        <i class="fa-solid fa-file-shield"></i>
                        <div class="doc-item-info">
                            <div class="doc-item-name" title="${escapeHtml(doc.law_name)}">${escapeHtml(doc.law_name)}</div>
                            <div class="doc-item-meta">${doc.uploaded_at.substring(0, 10)} • ${doc.file_size_kb} KB</div>
                        </div>
                    </div>
                    <div class="doc-item-actions">
                        <a href="${API_BASE}/api/documents/download/${doc.filename}" class="doc-action-btn" title="Tải file Word gốc">
                            <i class="fa-solid fa-download"></i>
                        </a>
                        <button class="doc-action-btn delete" onclick="deleteDocument('${doc.doc_id}')" title="Xóa tài liệu khỏi hệ thống RAG">
                            <i class="fa-solid fa-trash-can"></i>
                        </button>
                    </div>
                `;
                adminDocList.appendChild(item);
            });
        }

    } catch (e) {
        console.error("Lỗi khi nạp dữ liệu hồ sơ luật:", e);
    }
}

// 4. Delete Document Handler
window.deleteDocument = async function(docId) {
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
        
        // Show success notification in chat thread
        appendMessage(
            "assistant",
            `<p><i class="fa-solid fa-trash-can" style="color: #ef4444;"></i> <strong>Purge Complete:</strong> Hệ thống RAG vừa xóa toàn bộ điều khoản và tệp thô của tài liệu có mã <strong>${escapeHtml(docId)}</strong> thành công.</p>
             <p>Cơ sở dữ liệu của Agent đã được thu hẹp về trạng thái cũ thời gian thực.</p>`,
            "fa-solid fa-robot"
        );

        // Refresh documents list
        loadDocuments();
    } catch (e) {
        alert("Lỗi khi thực hiện xóa tài liệu: " + e.message);
    }
};

// Initial triggers
tabAdminBtn.style.display = "none"; // Hide admin tab initially
loadDocuments(); // Load registry metadata on launch


