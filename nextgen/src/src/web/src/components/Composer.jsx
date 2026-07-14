import { useState } from "react";

const MAX_WORDS = 500;
const countWords = (s) => (s.trim() ? s.trim().split(/\s+/).length : 0);

// Ô nhập câu hỏi: đếm từ (x/500), Enter để gửi, nút gửi xanh.
export default function Composer({ onSend, busy }) {
  const [text, setText] = useState("");
  const words = countWords(text);
  const over = words > MAX_WORDS;

  const submit = () => {
    const q = text.trim();
    if (!q || busy || over) return;
    onSend(q);
    setText("");
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="composer">
      <div className="composer-box">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Đặt câu hỏi bất kỳ…"
          rows={1}
          disabled={busy}
        />
        <button
          className="send"
          onClick={submit}
          disabled={busy || !text.trim() || over}
          title="Gửi (Enter)"
          aria-label="Gửi"
        >
          {busy ? <span className="spinner" /> : <SendIcon />}
        </button>
      </div>
      <div className="composer-foot">
        <span className={"wordcount" + (over ? " over" : "")}>
          {words}/{MAX_WORDS} từ
        </span>
      </div>
    </div>
  );
}

function SendIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <path d="M4 12l16-8-6 16-3-6-7-2z" fill="currentColor" />
    </svg>
  );
}
