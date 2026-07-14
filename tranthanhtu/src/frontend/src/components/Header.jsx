import React, { useState, useEffect } from 'react'
import { Scale, User, ShieldCheck, Sun, Moon } from 'lucide-react'
import { useLegalStore } from '../store/useLegalStore'

export default function Header() {
  const { currentRole, setRole } = useLegalStore()
  
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'dark'
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark')
  }

  return (
    <header className="app-header">
      <div className="logo-area">
        <div className="logo-icon flex items-center justify-center">
          <Scale size={20} className="text-white" />
        </div>
        <div className="logo-text">
          <h1>AI Legal Assistant</h1>
          <span className="sub-logo">Trợ lý Pháp lý Doanh nghiệp SME</span>
        </div>
      </div>
      
      <div className="flex items-center gap-2">
        {/* Theme Toggle Button */}
        <button 
          onClick={toggleTheme} 
          className="theme-toggle-btn"
          title="Chuyển đổi Giao diện Sáng/Tối"
        >
          {theme === 'dark' ? <Sun size={15} className="text-amber-400" /> : <Moon size={15} className="text-indigo-600" />}
        </button>

        {/* Role Selector Toggle */}
        <div className="role-selector">
          <button 
            className={`role-btn ${currentRole === 'client' ? 'active' : ''}`} 
            id="role-client-btn"
            onClick={() => setRole('client')}
          >
            <User size={13} /> SME Client
          </button>
          <button 
            className={`role-btn ${currentRole === 'admin' ? 'active' : ''}`} 
            id="role-admin-btn"
            onClick={() => setRole('admin')}
          >
            <ShieldCheck size={13} /> Admin
          </button>
        </div>
      </div>

      <div className="session-badge">
        <span className="status-dot pulsing"></span>
        <span id="session-status">Agent Sẵn sàng</span>
      </div>
    </header>
  )
}
