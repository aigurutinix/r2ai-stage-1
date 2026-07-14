import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Tin nhắn: trợ lý (icon ✦, render markdown, có nút Copy) hoặc người dùng (bong bóng xám, phải).
export default function ChatMessage({ role, content, streaming, error, greeting }) {
  if (role === "user") {
    return (
      <div className="msg user">
        <div className="bubble">{content}</div>
      </div>
    );
  }

  const showActions = !greeting && !streaming && !error && !!content;

  return (
    <div className="msg assistant">
      <div className="spark">✦</div>
      <div className="assistant-col">
        <div className={"assistant-body" + (error ? " error" : "")}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content || ""}</ReactMarkdown>
          {streaming && <span className="caret" />}
        </div>
        {showActions && <MessageActions text={content} />}
      </div>
    </div>
  );
}

function MessageActions({ text }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard bị chặn (http) — bỏ qua */
    }
  };
  return (
    <div className="msg-actions">
      <button className="act" onClick={copy} title="Sao chép câu trả lời">
        {copied ? "✓ Đã sao chép" : "⧉ Sao chép"}
      </button>
    </div>
  );
}
