import React, { useEffect, useRef } from 'react'
import { Bot, User, HelpCircle, FileWarning, Loader2 } from 'lucide-react'
import { useLegalStore } from '../store/useLegalStore'

export default function ChatThread() {
  const { messages, isLoading, submitQuery, selectCitation } = useLegalStore()
  const threadEndRef = useRef(null)
  const threadRef = useRef(null)

  // Auto scroll to bottom on new messages or loading state change
  useEffect(() => {
    const timer = setTimeout(() => {
      if (threadEndRef.current) {
        threadEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      }
    }, 100)
    return () => clearTimeout(timer)
  }, [messages, isLoading])

  // Handle welcome chip click
  const handleChipClick = (queryText) => {
    submitQuery(queryText)
  }

  // Handle Event Delegation for dynamic raw HTML citations clicks
  const handleThreadClick = (e) => {
    // Find closest citation tag or badge
    const citationTag = e.target.closest('.text-citation') || e.target.closest('.citation-tag')
    if (citationTag) {
      const citationId = citationTag.getAttribute('data-citation-id')
      if (window.activeCitations && window.activeCitations[citationId]) {
        const citationData = window.activeCitations[citationId]
        selectCitation(citationData)
      }
    }

    // Find chip btn if clicked inside welcome message
    const chipBtn = e.target.closest('.chip-btn')
    if (chipBtn) {
      const query = chipBtn.getAttribute('data-query')
      if (query) {
        handleChipClick(query)
      }
    }
  }

  // Determine avatar icon based on sender type
  const renderAvatar = (sender, iconClass) => {
    if (sender === 'user') {
      return <User className="text-white" size={16} />
    } else if (iconClass === 'fa-solid fa-triangle-exclamation') {
      return <FileWarning className="text-red-300" size={16} />
    } else if (iconClass === 'fa-solid fa-circle-question') {
      return <HelpCircle className="text-amber-300" size={16} />
    }
    return <Bot className="text-white" size={16} />
  }

  return (
    <section 
      className="chat-thread" 
      ref={threadRef}
      onClick={handleThreadClick}
    >
      {messages.map((msg, idx) => (
        <div 
          key={idx} 
          className={`message ${msg.sender === 'user' ? 'user-msg' : msg.sender === 'system' ? 'system-msg' : 'assistant-msg'}`}
        >
          <div className="message-avatar flex items-center justify-center">
            {renderAvatar(msg.sender, msg.iconClass)}
          </div>
          <div 
            className="message-content"
            dangerouslySetInnerHTML={{ __html: msg.htmlContent }}
          />
        </div>
      ))}

      {/* Thinking loader */}
      {isLoading && (
        <div className="agent-loader flex items-center gap-3">
          <div className="loader-icon">
            <Loader2 size={16} className="animate-spin text-emerald-400" />
          </div>
          <div className="loader-text font-medium text-emerald-300/80">Agent đang phân tích lập luận & tra cứu điều luật...</div>
        </div>
      )}

      <div ref={threadEndRef} />
    </section>
  )
}
