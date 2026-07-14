// Modal Cài đặt (tối giản): thông tin tài khoản + về ứng dụng.
export default function SettingsModal({ open, onClose, user, onSignOut }) {
  if (!open) return null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>Cài đặt</h3>
          <button className="cite-close" onClick={onClose} aria-label="Đóng">✕</button>
        </div>

        <div className="modal-body">
          <div className="set-row">
            <span className="set-label">Tài khoản</span>
            <span className="set-val">{user ? (user.name || user.email) : "Chưa đăng nhập"}</span>
          </div>
          <div className="set-row">
            <span className="set-label">Mô hình</span>
            <span className="set-val">LLM &lt; 14B (tuân thủ quy định BTC)</span>
          </div>
          <div className="set-row">
            <span className="set-label">Nguồn dữ liệu</span>
            <span className="set-val">Văn bản pháp luật SME · Thư viện Pháp luật</span>
          </div>
          <p className="set-note">
            NextGen trả lời dựa trên văn bản pháp luật chính thức và luôn kèm trích dẫn để kiểm chứng.
            Vui lòng đối chiếu bản gốc trước khi áp dụng.
          </p>
          {user && (
            <button className="danger-btn" onClick={onSignOut}>Đăng xuất</button>
          )}
        </div>
      </div>
    </div>
  );
}
