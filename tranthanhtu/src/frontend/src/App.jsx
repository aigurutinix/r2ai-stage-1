import React from 'react'
import Header from './components/Header'
import ChatThread from './components/ChatThread'
import ChatInput from './components/ChatInput'
import VerificationPanel from './components/VerificationPanel'

export default function App() {
  return (
    <div className="app-container">
      {/* Main Chat Area (Left Panel) */}
      <main className="chat-panel">
        <Header />
        <ChatThread />
        <ChatInput />
      </main>

      {/* Legal Verification Panel (Right Panel) */}
      <VerificationPanel />
    </div>
  )
}
