import React, { useState } from 'react'
import { Send, ShieldAlert, HelpCircle } from 'lucide-react'
import { useLegalStore } from '../store/useLegalStore'

export default function ChatInput() {
  const { clarification, isLoading, submitQuery } = useLegalStore()
  const [value, setValue] = useState("")

  const [customOpt, setCustomOpt] = useState("")

  const handleSubmit = (e) => {
    e?.preventDefault()
    if (!value.trim() || isLoading) return
    submitQuery(value.trim())
    setValue("")
  }

  const handleCustomSubmit = (e) => {
    e?.preventDefault()
    if (!customOpt.trim() || isLoading) return
    submitQuery(customOpt.trim())
    setCustomOpt("")
  }

  return (
    <div className="w-full">
      {/* Clarification panel above input area */}
      {clarification && (
        <div className="clarification-panel">
          <div className="clarification-header">
            <HelpCircle size={15} className="text-amber-400" />
            <span>Làm rõ thông tin: (Chọn phương án hoặc tự nhập câu trả lời của bạn)</span>
          </div>
          <div className="clarification-options">
            {clarification.options && Array.isArray(clarification.options) && clarification.options.map((opt, idx) => (
              <button 
                key={idx}
                className="clarify-btn"
                onClick={() => {
                  submitQuery(opt)
                }}
              >
                {opt}
              </button>
            ))}
            
            {/* Custom input option at the end */}
            <form onSubmit={handleCustomSubmit} className="clarification-custom-input-container">
              <input 
                type="text" 
                value={customOpt}
                onChange={(e) => setCustomOpt(e.target.value)}
                placeholder="Ý kiến khác... (Nhập tình huống chi tiết của riêng bạn tại đây)" 
                className="clarification-custom-input"
                disabled={isLoading}
              />
              <button type="submit" className="clarification-custom-btn" disabled={isLoading || !customOpt.trim()}>
                Gửi câu trả lời
              </button>
            </form>
          </div>
        </div>
      )}

      <footer className="input-area">
        <form onSubmit={handleSubmit} className="chat-form">
          <input 
            type="text" 
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="Nhập tình huống hoặc câu hỏi pháp lý của doanh nghiệp tại đây..." 
            autoComplete="off" 
            disabled={isLoading}
            required
          />
          <button type="submit" className="send-btn flex items-center justify-center" disabled={isLoading}>
            <Send size={16} />
          </button>
        </form>
        <div className="privacy-note">
          <ShieldAlert size={12} className="text-emerald-400" />
          <span>Bảo mật tối đa: Hệ thống tự động khử định danh (PII Redaction) để bảo vệ bí mật kinh doanh của bạn.</span>
        </div>
      </footer>
    </div>
  )
}
