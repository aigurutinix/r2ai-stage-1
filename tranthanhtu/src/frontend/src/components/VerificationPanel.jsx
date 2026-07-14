import React, { useState, useEffect } from 'react'
import { 
  Gavel, 
  Wrench, 
  FolderOpen, 
  Download, 
  Trash2, 
  ArrowLeft, 
  ExternalLink, 
  FileText, 
  FileUp, 
  CloudLightning,
  Sparkles,
  Loader2
} from 'lucide-react'
import { useLegalStore } from '../store/useLegalStore'

export default function VerificationPanel() {
  const { 
    currentRole, 
    activeTab, 
    setTab, 
    activeCitation, 
    clearCitation, 
    documents, 
    loadDocuments, 
    deleteDocument,
    uploadProgress,
    ingestAlert,
    uploadDocx,
    importVbpl
  } = useLegalStore()

  // Local Form Inputs state
  const [lawName, setLawName] = useState("")
  const [sourceUrl, setSourceUrl] = useState("")
  const [selectedFile, setSelectedFile] = useState(null)
  const [vbplId, setVbplId] = useState("")
  const [dragOver, setDragOver] = useState(false)

  // Load documents metadata on mount
  useEffect(() => {
    loadDocuments()
  }, [])

  // Format date utility
  const formatDate = (dateStr) => {
    if (!dateStr) return "Chưa cập nhật"
    try {
      const parts = dateStr.split('-')
      if (parts.length === 3) {
        return `${parts[2]}/${parts[1]}/${parts[0]}`
      }
      return dateStr
    } catch (e) {
      return dateStr
    }
  }

  // Handle file drop & selection
  const onFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (file) => {
    if (!file.name.endsWith('.docx')) {
      alert("Chỉ hỗ trợ nạp tệp tin định dạng Microsoft Word (.docx).")
      return
    }
    setSelectedFile(file)
  }

  const handleUploadSubmit = () => {
    if (!selectedFile) {
      alert("Vui lòng kéo thả hoặc chọn một tệp tin .docx pháp luật trước.")
      return
    }
    uploadDocx(selectedFile, lawName, sourceUrl)
    // Reset forms
    setSelectedFile(null)
    setLawName("")
    setSourceUrl("")
  }

  const handleVbplSubmit = () => {
    const trimmedId = vbplId.trim()
    if (!trimmedId) {
      alert("Vui lòng nhập ID số hoặc chuỗi UUID văn bản pháp lý từ trang vbpl.vn.")
      return
    }
    importVbpl(trimmedId)
    setVbplId("")
  }

  return (
    <aside className="verification-panel">
      <header className="verification-header">
        <div className="panel-title flex items-center gap-2">
          <Gavel size={18} className="text-amber-400" />
          <h2>Cơ sở Pháp lý Đối chiếu</h2>
        </div>
      </header>

      {/* Tabs */}
      <div className="panel-tabs">
        <button 
          className={`tab-btn ${activeTab === 'laws' ? 'active' : ''}`}
          onClick={() => setTab('laws')}
        >
          <Gavel size={14} /> Đối chiếu Luật
        </button>
        {currentRole === 'admin' && (
          <button 
            className={`tab-btn ${activeTab === 'admin' ? 'active' : ''}`}
            onClick={() => setTab('admin')}
          >
            <Wrench size={14} /> Học Luật Thô
          </button>
        )}
      </div>

      <div className="verification-body">
        {/* LAWS TAB */}
        {activeTab === 'laws' && (
          <div>
            {!activeCitation ? (
              // Library view
              <div>
                <div className="side-panel-placeholder">
                  <div className="placeholder-icon">
                    <FolderOpen size={40} className="text-slate-500" />
                  </div>
                  <h3>Bảng Tra Cứu Nguyên Văn</h3>
                  <p>Nhấp vào bất kỳ <strong>Thẻ Trích Dẫn</strong> nào trong bong bóng chat để hiển thị nguyên văn văn bản pháp luật gốc tại đây nhằm tự đối chiếu chéo.</p>
                </div>

                {/* Read Only Library list */}
                <div className="legal-library-section">
                  <div className="section-title-box">
                    <FolderOpen size={14} />
                    <span>Thư viện Văn bản Trợ lý sở hữu</span>
                  </div>
                  <div className="doc-list">
                    {documents.length === 0 ? (
                      <div className="doc-item-name" style={{ color: 'var(--text-muted)', fontSize: '12px', padding: '10px' }}>
                        Thư viện trống. Chưa có văn bản nào được nạp.
                      </div>
                    ) : (
                      documents.map((doc) => (
                        <div key={doc.doc_id} className="doc-item">
                          <div className="doc-item-left">
                            <FileText size={20} className="text-emerald-400" />
                            <div className="doc-item-info">
                              <div className="doc-item-name" title={doc.law_name}>{doc.law_name}</div>
                              <div className="doc-item-meta">{doc.total_articles} Điều • {doc.file_size_kb} KB</div>
                            </div>
                          </div>
                          <div className="doc-item-actions">
                            <a href={`http://127.0.0.1:8089/api/documents/download/${doc.filename}`} className="doc-action-btn" title="Tải xuống tài liệu gốc .docx">
                              <Download size={13} />
                            </a>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            ) : (
              // Verified Law card view
              <div className="verified-law-card">
                <div className="flex items-center justify-between w-full border-b border-white/5 pb-2 mb-4">
                  <button className="back-to-library-btn !mb-0" onClick={clearCitation}>
                    <ArrowLeft size={13} /> Quay lại Thư viện
                  </button>
                  <div className={`law-tag !mb-0 ${activeCitation.status === 'expired' ? 'expired-tag' : ''}`}>
                    {activeCitation.status === 'expired' ? 'Hết hiệu lực' : 'Còn hiệu lực'}
                  </div>
                </div>
                <h3 className="law-title">
                  {activeCitation.article} {activeCitation.clause ? `- ${activeCitation.clause}` : ''} {activeCitation.point ? `- ${activeCitation.point}` : ''}
                </h3>
                <div className="law-meta">
                  <div className="meta-item">
                    <span className="meta-label">Văn bản:</span>
                    <span className="meta-value">{activeCitation.law_name}</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Hiệu lực:</span>
                    <span className="meta-value">{activeCitation.effective_date ? formatDate(activeCitation.effective_date) : 'Đang hiệu lực'}</span>
                  </div>
                </div>

                <div className="law-section-divider">Nguyên văn Điều Luật:</div>
                <div className="law-text-content">
                  <p>{activeCitation.exact_text}</p>
                </div>

                {activeCitation.matched_child_clause && (
                  <div>
                    <div className="law-section-divider text-accent">Khoản/Điểm đối chiếu khớp:</div>
                    <div className="law-highlight-box">
                      <p>{activeCitation.matched_child_clause}</p>
                    </div>
                  </div>
                )}

                {activeCitation.source_url && (
                  <a href={activeCitation.source_url} target="_blank" rel="noopener noreferrer" className="verify-external-btn">
                    <ExternalLink size={13} /> Kiểm tra tại vbpl.vn (Cổng Chính Phủ)
                  </a>
                )}
              </div>
            )}
          </div>
        )}

        {/* ADMIN TAB */}
        {activeTab === 'admin' && (
          <div className="admin-dashboard-card">
            <h3>Nạp Tài liệu Pháp lý Thô (.docx)</h3>
            <p className="admin-desc">Hệ thống RAG Động cho phép tải lên trực tiếp văn bản luật thô (.docx) tải từ cổng chính phủ để AI tự động bóc tách cấu trúc và học ngay lập tức.</p>

            {/* Drag Zone */}
            <div 
              className={`upload-dropzone ${dragOver ? 'dragover' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]) }}
              onClick={() => document.getElementById('admin-file-picker').click()}
            >
              <FileUp size={34} className={`dropzone-icon ${dragOver ? 'text-emerald-400' : 'text-slate-500'}`} />
              <p className="dropzone-text">
                {selectedFile ? `Đã chọn: ${selectedFile.name} (${(selectedFile.size / 1024).toFixed(1)} KB)` : 'Kéo thả file .docx tại đây hoặc nhấp chuột để chọn file'}
              </p>
              <input type="file" id="admin-file-picker" accept=".docx" style={{ display: 'none' }} onChange={onFileChange} />
            </div>

            <div className="admin-inputs-group">
              <input 
                type="text" 
                value={lawName}
                onChange={(e) => setLawName(e.target.value)}
                placeholder="Tên văn bản luật (Tùy chọn ghi đè)..."
              />
              <input 
                type="text" 
                value={sourceUrl}
                onChange={(e) => setSourceUrl(e.target.value)}
                placeholder="URL nguồn kiểm chứng thực tế (Tùy chọn)..."
              />
            </div>

            <button className="upload-submit-btn" onClick={handleUploadSubmit}>
              <CloudLightning size={14} /> Bắt đầu Nạp & Trích lọc
            </button>

            <div className="or-divider">
              <span>HOẶC</span>
            </div>

            <h3>Tải & Học từ Cổng chính phủ VBPL.vn</h3>
            <p className="admin-desc">Nhập mã ID hoặc danh sách nhiều ID văn bản pháp lý từ trang <strong>vbpl.vn</strong> chính thức (ví dụ: <code>"139877", "162690", ...</code> hoặc dán thô danh sách ID) để AI kết nối, tải toàn văn và học đồng loạt vào RAG vector.</p>

            <div className="admin-inputs-group" style={{ marginTop: '4px' }}>
              <textarea 
                rows={3}
                value={vbplId}
                onChange={(e) => setVbplId(e.target.value)}
                placeholder='Nhập một hoặc nhiều ID VBPL (ví dụ: "139877", "162690",...)'
              />
            </div>

            <button 
              className="upload-submit-btn" 
              style={{ background: 'linear-gradient(135deg, var(--accent-gold-dark), var(--accent-gold))', color: '#ffffff' }}
              onClick={handleVbplSubmit}
            >
              <Sparkles size={14} /> Kết nối & Đồng bộ VBPL
            </button>

            {/* Ingest Progress bar */}
            {uploadProgress.visible && (
              <div className="upload-progress-container">
                <div className="progress-bar-bg">
                  <div className="progress-bar-fill" style={{ width: `${uploadProgress.percent}%` }}></div>
                </div>
                <span className="progress-status-text flex items-center gap-1.5">
                  <Loader2 size={10} className="animate-spin text-emerald-400" /> {uploadProgress.status}
                </span>
              </div>
            )}

            {/* Ingest Alert Notification */}
            {ingestAlert.visible && (
              <div className={`ingest-status-alert ${ingestAlert.type}`}>
                <p>{ingestAlert.message}</p>
              </div>
            )}

            {/* Manager list with cascade delete red triggers */}
            <div className="legal-library-section">
              <div className="section-title-box text-accent-red">
                <FolderOpen size={14} />
                <span>Quản lý Kho Dữ liệu thô</span>
              </div>
              <div className="doc-list">
                {documents.length === 0 ? (
                  <div className="doc-item-name" style={{ color: 'var(--text-muted)', fontSize: '12px', padding: '10px' }}>
                    Chưa có tài liệu nào trong kho.
                  </div>
                ) : (
                  documents.map((doc) => (
                    <div key={doc.doc_id} className="doc-item">
                      <div className="doc-item-left">
                        <FileText size={20} className="text-amber-400" />
                        <div className="doc-item-info">
                          <div className="doc-item-name" title={doc.law_name}>{doc.law_name}</div>
                          <div className="doc-item-meta">{doc.uploaded_at.substring(0, 10)} • {doc.file_size_kb} KB</div>
                        </div>
                      </div>
                      <div className="doc-item-actions">
                        <a href={`http://127.0.0.1:8089/api/documents/download/${doc.filename}`} className="doc-action-btn" title="Tải file Word gốc">
                          <Download size={13} />
                        </a>
                        <button className="doc-action-btn delete" onClick={() => deleteDocument(doc.doc_id)} title="Xóa tài liệu khỏi hệ thống RAG">
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

          </div>
        )}
      </div>
    </aside>
  )
}
