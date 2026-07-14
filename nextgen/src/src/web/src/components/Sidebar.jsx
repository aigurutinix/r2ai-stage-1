// Sidebar: Trợ lý NextGen (New chat) · danh sách hội thoại lịch sử · tài khoản/đăng nhập · Settings.
export default function Sidebar({
  open, onClose,
  user, conversations, currentConvId,
  onNewChat, onSelect, onDelete,
  onSignIn, onSignOut, onOpenSettings,
}) {
  return (
    <>
      {open && <div className="sidebar-backdrop" onClick={onClose} />}
      <aside className={"sidebar" + (open ? " open" : "")}>
        <div className="sidebar-brand">
          Next<span className="brand-x">Gen</span>
        </div>

        <button className="new-chat" onClick={onNewChat}>
          <span className="spark-mini">✦</span> Trợ lý NextGen
          <span className="plus">+</span>
        </button>

        <div className="side-section-label">Lịch sử</div>
        <div className="conv-list">
          {!user && (
            <div className="side-hint">Đăng nhập để lưu và xem lại lịch sử hội thoại.</div>
          )}
          {user && conversations.length === 0 && (
            <div className="side-hint">Chưa có hội thoại nào. Hãy đặt câu hỏi đầu tiên!</div>
          )}
          {conversations.map((c) => (
            <div
              key={c.id}
              className={"conv-item" + (c.id === currentConvId ? " active" : "")}
              onClick={() => onSelect(c.id)}
              title={c.title}
            >
              <span className="conv-title">{c.title}</span>
              <button
                className="conv-del"
                title="Xóa hội thoại"
                onClick={(e) => { e.stopPropagation(); onDelete(c.id); }}
              >
                🗑
              </button>
            </div>
          ))}
        </div>

        <div className="sidebar-foot">
          {user ? (
            <div className="user-box">
              {user.avatar
                ? <img className="avatar" src={user.avatar} alt="" referrerPolicy="no-referrer" />
                : <span className="avatar avatar-fallback">{(user.name || "U")[0]}</span>}
              <div className="user-meta">
                <div className="user-name">{user.name || user.email}</div>
                <button className="linkish" onClick={onSignOut}>Đăng xuất</button>
              </div>
            </div>
          ) : (
            <button className="google-btn" onClick={onSignIn}>
              <GoogleIcon /> Đăng nhập với Google
            </button>
          )}
          <button className="settings-btn" onClick={onOpenSettings}>⚙ Cài đặt</button>
        </div>
      </aside>
    </>
  );
}

function GoogleIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3C33.7 32.4 29.3 35 24 35c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.9 1.2 8 3.1l5.7-5.7C34.5 5.1 29.5 3 24 3 12.4 3 3 12.4 3 24s9.4 21 21 21 21-9.4 21-21c0-1.2-.1-2.3-.4-3.5z"/>
      <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.7 16 19 13 24 13c3.1 0 5.9 1.2 8 3.1l5.7-5.7C34.5 5.1 29.5 3 24 3 16 3 9.1 7.6 6.3 14.7z"/>
      <path fill="#4CAF50" d="M24 45c5.2 0 9.9-2 13.4-5.2l-6.2-5.2C29.2 36 26.7 37 24 37c-5.3 0-9.7-2.6-11.3-6.9l-6.5 5C9 41.4 15.9 45 24 45z"/>
      <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.2-2.2 4.1-4.1 5.6l6.2 5.2C39.7 41.5 45 38 45 24c0-1.2-.1-2.3-.4-3.5z"/>
    </svg>
  );
}
