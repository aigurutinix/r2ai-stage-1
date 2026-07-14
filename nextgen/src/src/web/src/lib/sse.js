// Gọi POST /ask và đọc luồng SSE. Gọi lại các callback theo từng sự kiện:
//   onCitations({citations, relevant_docs, relevant_articles})
//   onToken(textDelta)
//   onDone({answer})
//   onError(message)
// Trả về một hàm abort() để hủy request nếu cần.
export function askStream(question, handlers = {}) {
  const controller = new AbortController();
  const { onCitations, onToken, onDone, onError } = handlers;

  (async () => {
    try {
      const resp = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: controller.signal,
      });
      if (!resp.ok || !resp.body) {
        const t = await resp.text().catch(() => "");
        throw new Error(t || `HTTP ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const raw = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          let ev = "message";
          let data = "";
          for (const line of raw.split("\n")) {
            if (line.startsWith("event:")) ev = line.slice(6).trim();
            else if (line.startsWith("data:")) data += line.slice(5).trim();
          }
          if (!data) continue;

          if (ev === "citations") onCitations && onCitations(JSON.parse(data));
          else if (ev === "token") onToken && onToken(JSON.parse(data));
          else if (ev === "done") onDone && onDone(JSON.parse(data));
          else if (ev === "error") onError && onError(JSON.parse(data).message || "Có lỗi xảy ra.");
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") onError && onError(err.message || String(err));
    }
  })();

  return () => controller.abort();
}
