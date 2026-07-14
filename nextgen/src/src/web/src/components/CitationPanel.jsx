// Panel phải "Văn bản liên quan": mỗi thẻ = tiêu đề (số hiệu + tên) · Điều · đoạn trích · link.
export default function CitationPanel({ open, citations, onClose }) {
  if (!open) return null;
  return (
    <aside className="citations">
      <div className="cite-head">
        <div className="cite-title">📖 Văn bản liên quan</div>
        <button className="cite-close" onClick={onClose} aria-label="Đóng">✕</button>
      </div>

      <div className="cite-list">
        {citations.length === 0 && (
          <div className="cite-empty">Chưa có văn bản trích dẫn.</div>
        )}
        {citations.map((c, i) => (
          <article className="cite-card" key={i}>
            <h4 className="cite-doc">
              <span className="cite-so">{c.so_hieu}</span> {c.ten_van_ban}
            </h4>
            {c.dieu && <div className="cite-dieu">{c.dieu}</div>}
            {c.doan_trich && <p className="cite-snippet">{c.doan_trich}</p>}
            {c.url && (
              <a className="cite-more" href={c.url} target="_blank" rel="noopener noreferrer">
                Xem thêm →
              </a>
            )}
          </article>
        ))}
      </div>
    </aside>
  );
}
