"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { Activity, FileUp, MessageSquarePlus, PanelLeft, Search, Send, SquarePen, Trash2 } from "lucide-react";
import { parseSseChunk } from "@/lib/sse";
import type { AgentTraceStep, ChatMessage, ChatResponse, ChatStreamEvent, ChatStreamProgress, CompetitionRecord, CompetitionStreamEvent, Conversation, RetrievalTraceResult, RuntimeConfig } from "@/lib/types";

const API_BASE = "/backend-api";
const DEFAULT_DATABASE = process.env.NEXT_PUBLIC_DEFAULT_DATABASE || "default";
const LEGACY_STORAGE_KEYS = ["mscai_conversations", "mscai_active_conversation_id", "mscai_session_id"];

function createId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function createConversation(): Conversation {
  const id = createId();
  const now = Date.now();
  return { id, title: "Đoạn chat mới", messages: [], createdAt: now, updatedAt: now };
}

function compactTitle(text: string) {
  const value = text.replace(/\s+/g, " ").trim();
  return value.length > 44 ? `${value.slice(0, 44)}...` : value || "Đoạn chat mới";
}

const STAGE_TITLES: Record<string, string> = {
  request: "Request",
  memory: "Short-memory",
  intent: "Phân tích ý định",
  prepare_query: "Chuẩn bị truy vấn",
  categories: "Phân loại category",
  retrieval: "Retrieval",
  rerank: "Rerank",
  llm_filter: "LLM filter",
  answer: "Tổng hợp câu trả lời",
  format: "Định dạng và memory",
  result: "Kết quả",
  response: "Hoàn thiện response",
  competition: "Competition",
  competition_item: "Câu hỏi",
  competition_item_result: "Kết quả câu",
  competition_result: "Kết quả batch",
};

function formatElapsed(value?: number | null) {
  if (value === null || value === undefined) return "";
  return value < 1000 ? `${value} ms` : `${(value / 1000).toFixed(1)} s`;
}

function progressTone(status: ChatStreamProgress["status"]): AgentTraceStep["tone"] {
  if (status === "completed") return "success";
  if (status === "warning") return "warning";
  if (status === "error") return "error";
  return "info";
}

function isRetrievalResult(value: unknown): value is RetrievalTraceResult {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function topResults(metadata?: Record<string, unknown>): RetrievalTraceResult[] {
  const raw = metadata?.top_results;
  if (!Array.isArray(raw)) return [];
  return raw.filter(isRetrievalResult);
}

function metadataDetail(metadata?: Record<string, unknown>) {
  if (!metadata || Object.keys(metadata).length === 0) return "";
  return Object.entries(metadata)
    .filter(([key]) => key !== "top_results")
    .map(([key, value]) => `${key}: ${describeValue(value)}`)
    .join("; ");
}

function progressStep(data: ChatStreamProgress): AgentTraceStep {
  const details = [data.message, data.detail, metadataDetail(data.metadata)].filter(Boolean);
  return {
    stage: data.stage,
    status: data.status,
    title: STAGE_TITLES[data.stage] || data.stage || "Xử lý",
    detail: details.join(" · "),
    elapsedMs: data.elapsed_ms,
    tone: progressTone(data.status),
    topResults: topResults(data.metadata),
  };
}

function upsertProgress(trace: AgentTraceStep[], data: ChatStreamProgress) {
  const next = [...trace];
  let match = -1;
  for (let index = next.length - 1; index >= 0; index -= 1) {
    if (next[index].stage === data.stage && next[index].status !== "completed" && next[index].status !== "error") {
      match = index;
      break;
    }
  }
  const step = progressStep(data);
  if (match >= 0) next[match] = step;
  else next.push(step);
  return next;
}


function describeValue(value: unknown) {
  if (Array.isArray(value)) return value.length ? value.join(", ") : "không có";
  if (typeof value === "boolean") return value ? "bật" : "tắt";
  if (value === null || value === undefined || value === "") return "không có";
  return String(value);
}

function formatScore(value?: number) {
  if (typeof value !== "number" || Number.isNaN(value)) return "0.0000";
  return value.toFixed(4);
}

function retrievalTitle(result: RetrievalTraceResult) {
  return [result.law_id, result.law_name, result.article].filter(Boolean).join(" | ");
}

const AUTO_SCROLL_THRESHOLD = 96;

function isNearBottom(element: HTMLElement) {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= AUTO_SCROLL_THRESHOLD;
}


export default function Home() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState("");
  const [input, setInput] = useState("");
  const [showStream, setShowStream] = useState(true);
  const [chatStreaming, setChatStreaming] = useState(true);
  const [competitionMode, setCompetitionMode] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const competitionFileRef = useRef<HTMLInputElement>(null);
  const shouldAutoScrollRef = useRef(true);

  const activeConversation = useMemo(
    () => conversations.find((item) => item.id === activeId) || conversations[0],
    [activeId, conversations],
  );
  const messages = activeConversation?.messages || [];

  useEffect(() => {
    for (const key of LEGACY_STORAGE_KEYS) {
      window.localStorage.removeItem(key);
    }
    const initial = [createConversation()];
    setConversations(initial);
    setActiveId(initial[0].id);

    fetch(`${API_BASE}/api/v1/legal/config`)
      .then((response) => (response.ok ? response.json() : null))
      .then((config: RuntimeConfig | null) => {
        if (!config) return;
        setChatStreaming(config.chat_streaming);
        setCompetitionMode(config.competition_enabled);
      })
      .catch(() => {
        setChatStreaming(true);
      });
  }, []);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element || !shouldAutoScrollRef.current) return;
    requestAnimationFrame(() => {
      element.scrollTo({ top: element.scrollHeight, behavior: isSending ? "auto" : "smooth" });
    });
  }, [messages, isSending, showStream]);

  useEffect(() => {
    shouldAutoScrollRef.current = true;
  }, [activeId]);

  function handleMessagesScroll() {
    const element = scrollRef.current;
    if (!element) return;
    shouldAutoScrollRef.current = isNearBottom(element);
  }

  const visibleConversations = useMemo(() => {
    const keyword = searchTerm.trim().toLowerCase();
    const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
    if (!keyword) return sorted;
    return sorted.filter((item) => {
      const haystack = [item.title, ...item.messages.map((message) => message.content)].join(" ").toLowerCase();
      return haystack.includes(keyword);
    });
  }, [conversations, searchTerm]);

  function patchConversation(id: string, updater: (conversation: Conversation) => Conversation) {
    setConversations((items) => items.map((item) => (item.id === id ? updater(item) : item)));
  }

  function newChat() {
    const next = createConversation();
    setConversations((items) => [next, ...items]);
    setActiveId(next.id);
    setSearchTerm("");
    setInput("");
  }

  function deleteChat(conversationId: string) {
    setConversations((items) => {
      const remaining = items.filter((item) => item.id !== conversationId);
      if (remaining.length === 0) {
        const next = createConversation();
        setActiveId(next.id);
        return [next];
      }
      if (conversationId === activeId) {
        setActiveId(remaining[0].id);
      }
      return remaining;
    });
  }

  function updateAssistant(conversationId: string, assistantId: string, updater: (message: ChatMessage) => ChatMessage) {
    patchConversation(conversationId, (conversation) => ({
      ...conversation,
      updatedAt: Date.now(),
      messages: conversation.messages.map((item) => (item.id === assistantId ? updater(item) : item)),
    }));
  }

  function handleStreamEvent(event: ChatStreamEvent, conversationId: string, assistantId: string) {
    if (event.event === "status") {
      updateAssistant(conversationId, assistantId, (message) => ({
        ...message,
        stream: event.data.status === "running" ? message.stream : [...message.stream, event.data.message],
        trace: upsertProgress(message.trace, event.data),
      }));
      return;
    }
    if (event.event === "token") {
      updateAssistant(conversationId, assistantId, (message) => ({
        ...message,
        content:
          message.content === "Đang tạo câu trả lời..."
            ? event.data.token
            : message.content + event.data.token,
      }));
      return;
    }
    if (event.event === "error") {
      updateAssistant(conversationId, assistantId, (message) => ({
        ...message,
        content: event.data.message || "Có lỗi khi gọi backend.",
        stream: [...message.stream, "Lỗi"],
        trace: upsertProgress(message.trace, event.data),
      }));
      return;
    }
    if (event.event === "result") {
      const answer = event.data.answer;
      updateAssistant(conversationId, assistantId, (message) => ({
        ...message,
        content: answer.answer || "Không có câu trả lời.",
        sources: answer.relevant_articles || [],
        stream: [...message.stream, "Đã nhận kết quả"],
        trace: [
          ...message.trace,
          {
            stage: "result",
            status: "completed",
            title: "Kết quả",
            detail: `Nhận ${answer.relevant_articles?.length || 0} điều luật và ${answer.relevant_docs?.length || 0} văn bản nguồn.`,
            tone: "success",
          },
        ],
      }));
      return;
    }
    if (event.event === "done") {
      updateAssistant(conversationId, assistantId, (message) => ({
        ...message,
        stream: [...message.stream, event.data.message],
        trace: upsertProgress(message.trace, event.data),
      }));
    }
  }

  async function submitChat(question: string) {
    if (!question.trim() || isSending) return;
    const conversationId = activeConversation?.id || createConversation().id;
    const text = question.trim();
    shouldAutoScrollRef.current = true;
    const userMessage: ChatMessage = { id: createId(), role: "user", content: text, stream: [], trace: [], sources: [] };
    const assistantId = createId();
    const chatEndpoint = chatStreaming ? `${API_BASE}/api/v1/legal/chat/stream` : `${API_BASE}/api/v1/legal/chat`;
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "Đang tạo câu trả lời...",
      stream: [],
      trace: [{ stage: "client", status: "completed", title: "Gửi request", detail: `POST ${chatEndpoint}`, tone: "success" }],
      sources: [],
    };

    patchConversation(conversationId, (conversation) => ({
      ...conversation,
      title: conversation.messages.length ? conversation.title : compactTitle(text),
      messages: [...conversation.messages, userMessage, assistantMessage],
      updatedAt: Date.now(),
    }));
    setActiveId(conversationId);
    setInput("");
    setIsSending(true);

    try {
      const body = JSON.stringify({
        session_id: conversationId,
        message: text,
        databases: [DEFAULT_DATABASE],
        top_k: 8,
        competition_mode: competitionMode,
      });
      const response = await fetch(chatEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      if (!chatStreaming) {
        const data = (await response.json()) as ChatResponse;
        updateAssistant(conversationId, assistantId, (message) => ({
          ...message,
          content: data.answer.answer || "Không có câu trả lời.",
          sources: data.answer.relevant_articles || [],
          trace: [
            ...message.trace,
            {
              stage: "result",
              status: "completed",
              title: "Kết quả",
              detail: `Nhận ${data.answer.relevant_articles?.length || 0} điều luật và ${data.answer.relevant_docs?.length || 0} văn bản nguồn.`,
              tone: "success",
            },
          ],
        }));
        return;
      }

      if (!response.body) throw new Error("Không có stream body");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() || "";
        for (const chunk of chunks) {
          const event = parseSseChunk(chunk);
          if (event) handleStreamEvent(event, conversationId, assistantId);
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Không gọi được backend.";
      updateAssistant(conversationId, assistantId, (item) => ({
        ...item,
        content: message,
        stream: [...item.stream, "Lỗi kết nối"],
        trace: [...item.trace, { title: "Lỗi kết nối", detail: message, tone: "error" }],
      }));
    } finally {
      setIsSending(false);
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitChat(input);
  }


  function normalizeCompetitionItems(value: unknown) {
    if (!Array.isArray(value)) throw new Error("File phải là JSON array.");
    return value.map((item, index) => {
      if (!item || typeof item !== "object") throw new Error(`Item ${index + 1} không phải object.`);
      const record = item as { id?: unknown; question?: unknown };
      if (typeof record.question !== "string" || !record.question.trim()) {
        throw new Error(`Item ${index + 1} thiếu question.`);
      }
      return {
        id: typeof record.id === "number" ? record.id : index + 1,
        question: record.question.trim(),
        competition_mode: true,
      };
    });
  }

  async function handleCompetitionFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || isSending) return;

    const conversationId = activeConversation?.id || createConversation().id;
    const assistantId = createId();
    setIsSending(true);

    try {
      const items = normalizeCompetitionItems(JSON.parse(await file.text()));
      const userMessage: ChatMessage = {
        id: createId(),
        role: "user",
        content: `Competition file: ${file.name} (${items.length} câu hỏi)`,
        stream: [],
        trace: [],
        sources: [],
      };
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "Đang chạy competition mode...",
        stream: [],
        trace: [
          {
            stage: "client",
            status: "completed",
            title: "Gửi file",
            detail: `POST ${API_BASE}/api/v1/legal/competition · ${items.length} câu hỏi`,
            tone: "success",
          },
        ],
        sources: [],
      };
      patchConversation(conversationId, (conversation) => ({
        ...conversation,
        title: conversation.messages.length ? conversation.title : compactTitle(file.name),
        messages: [...conversation.messages, userMessage, assistantMessage],
        updatedAt: Date.now(),
      }));
      setActiveId(conversationId);

      const response = await fetch(`${API_BASE}/api/v1/legal/competition/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(items),
      });
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

      const results: CompetitionRecord[] = [];
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() || "";
        for (const chunk of chunks) {
          const event = parseSseChunk<CompetitionStreamEvent>(chunk);
          if (!event) continue;
          if (event.event === "status" || event.event === "done") {
            updateAssistant(conversationId, assistantId, (message) => ({
              ...message,
              content: event.data.stage === "competition" && event.data.status === "completed"
                ? `Hoàn tất ${results.length} câu hỏi.${event.data.metadata?.output_path ? `\nĐã lưu: ${event.data.metadata.output_path}` : ""}\n\n${JSON.stringify(results, null, 2)}`
                : event.data.message,
              trace: upsertProgress(message.trace, event.data),
            }));
          } else if (event.event === "competition_item_result") {
            const { index, total, ...record } = event.data;
            results.push(record);
            updateAssistant(conversationId, assistantId, (message) => ({
              ...message,
              content: `Đã hoàn tất ${index}/${total} câu hỏi...`,
              sources: Array.from(new Set(results.flatMap((item) => item.relevant_articles || []))).slice(0, 8),
              trace: [
                ...message.trace,
                {
                  stage: "competition_item_result",
                  status: "completed",
                  title: `Kết quả câu ${index}`,
                  detail: `${record.relevant_articles?.length || 0} điều luật · id: ${record.id ?? "không có"}`,
                  tone: "success",
                },
              ],
            }));
          } else if (event.event === "competition_result") {
            results.splice(0, results.length, ...event.data);
            updateAssistant(conversationId, assistantId, (message) => ({
              ...message,
              content: `Hoàn tất ${results.length} câu hỏi.\n\n${JSON.stringify(results, null, 2)}`,
              sources: Array.from(new Set(results.flatMap((item) => item.relevant_articles || []))).slice(0, 8),
              trace: [
                ...message.trace,
                {
                  stage: "competition_result",
                  status: "completed",
                  title: "Kết quả competition",
                  detail: `Nhận ${results.length} record submit.`,
                  tone: "success",
                },
              ],
            }));
          } else if (event.event === "error") {
            updateAssistant(conversationId, assistantId, (message) => ({
              ...message,
              content: event.data.message,
              trace: upsertProgress(message.trace, event.data),
            }));
          }
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Không xử lý được file competition.";
      patchConversation(conversationId, (conversation) => {
        const hasAssistant = conversation.messages.some((item) => item.id === assistantId);
        const assistantMessage: ChatMessage = {
          id: assistantId,
          role: "assistant",
          content: message,
          stream: [],
          trace: [{ title: "Lỗi competition", detail: message, tone: "error" }],
          sources: [],
        };
        return {
          ...conversation,
          messages: hasAssistant
            ? conversation.messages.map((item) => (item.id === assistantId ? assistantMessage : item))
            : [...conversation.messages, assistantMessage],
          updatedAt: Date.now(),
        };
      });
    } finally {
      setIsSending(false);
    }
  }

  return (
    <main className={sidebarOpen ? "shell" : "shell sidebarClosed"}>
      {sidebarOpen ? (
        <aside className="sidebar">
          <div className="brand">
            <span>MscAI</span>
            <button className="iconButton" onClick={() => setSidebarOpen(false)} aria-label="Ẩn sidebar"><PanelLeft size={19} /></button>
          </div>
          <nav className="nav">
            <button className="navItem active" onClick={newChat}><MessageSquarePlus size={20} /> Đoạn chat mới</button>
            <button className="navItem" onClick={() => setSearchOpen((value) => !value)}><Search size={20} /> Tìm kiếm đoạn chat</button>
          </nav>
          {searchOpen ? (
            <input
              className="searchInput"
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              autoFocus
              placeholder="Nhập từ khóa"
            />
          ) : null}
          <div className="sectionTitle">Đoạn chat</div>
          <div className="chatList">
            {visibleConversations.length === 0 ? <span className="muted">Không có đoạn chat</span> : null}
            {visibleConversations.map((conversation) => {
              const selected = conversation.id === activeConversation?.id;
              return (
                <div className={selected ? "chatRow selected" : "chatRow"} key={conversation.id}>
                  <button className="chatLink" onClick={() => setActiveId(conversation.id)}>
                    <span>{conversation.title}</span>
                    <small>{conversation.messages.length ? `${Math.ceil(conversation.messages.length / 2)} lượt hỏi` : "Mới"}</small>
                  </button>
                  <button
                    className="deleteChat"
                    type="button"
                    disabled={isSending && selected}
                    onClick={() => deleteChat(conversation.id)}
                    aria-label={`Xóa ${conversation.title}`}
                    title="Xóa đoạn chat"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              );
            })}
          </div>
        </aside>
      ) : null}

      <section className="chatArea">
        <header className="topbar">
          {!sidebarOpen ? <button className="iconButton" onClick={() => setSidebarOpen(true)} aria-label="Hiện sidebar"><PanelLeft size={19} /></button> : null}
          <button className="traceButton" onClick={() => setShowStream((value) => !value)} type="button" aria-pressed={showStream}>
            <Activity size={17} /> {showStream ? "Ẩn luồng" : "Hiện luồng"}
          </button>
          <button className="traceButton" onClick={() => setCompetitionMode((value) => !value)} type="button" aria-pressed={competitionMode}>
            <FileUp size={17} /> Competition
          </button>
          {isSending ? <span className="loader" aria-label="Đang xử lý" /> : null}
        </header>
        <div className="messages" ref={scrollRef} onScroll={handleMessagesScroll}>
          {messages.length === 0 ? <div className="empty">Hôm nay bạn có câu hỏi pháp lý gì?</div> : null}
          {messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              <div className="bubble">
                {message.role === "assistant" && showStream && (message.trace.length > 0 || message.stream.length > 0) ? (
                  <div className="streamBox">
                    <div className="streamHead">Luồng xử lý agent</div>
                    <div className="streamBody">
                      {message.trace.map((step, index) => (
                        <div className={`traceStep ${step.tone || "info"} ${step.status || ""}`} key={`${step.title}-${index}`}>
                          <span>{index + 1}</span>
                          <div>
                            <div className="traceTitleRow">
                              <strong>{step.title}</strong>
                              {step.elapsedMs !== null && step.elapsedMs !== undefined ? <small>{formatElapsed(step.elapsedMs)}</small> : null}
                            </div>
                            <p>{step.detail}</p>
                            {step.topResults && step.topResults.length > 0 ? (
                              <div className="retrievalTopList">
                                {step.topResults.slice(0, 5).map((result, resultIndex) => (
                                  <div className="retrievalTopItem" key={`${result.law_id || "doc"}-${result.article || resultIndex}-${resultIndex}`}>
                                    <div className="retrievalTopMeta">
                                      <span>#{result.rank || resultIndex + 1}</span>
                                      <span>score {formatScore(result.score)}</span>
                                      {result.source ? <span>{result.source}</span> : null}
                                      {typeof result.passed_threshold === "boolean" ? <span>{result.passed_threshold ? "giữ" : "loại"}</span> : null}
                                    </div>
                                    <strong>{retrievalTitle(result)}</strong>
                                    {result.article_title ? <small>{result.article_title}</small> : null}
                                  </div>
                                ))}
                              </div>
                            ) : null}
                          </div>
                        </div>
                      ))}
                      {message.trace.length === 0 ? message.stream.map((line, index) => <div className="traceLine" key={`${line}-${index}`}>• {line}</div>) : null}
                    </div>
                  </div>
                ) : null}
                <div>{message.content}</div>
                {message.sources.length > 0 ? (
                  <div className="sources">
                    <strong>Nguồn</strong>
                    {message.sources.slice(0, 6).map((source) => <span key={source}>{source}</span>)}
                  </div>
                ) : null}
              </div>
            </article>
          ))}
          <div ref={bottomRef} className="bottomAnchor" />
        </div>

        <div className={sidebarOpen ? "composerWrap" : "composerWrap full"}>
          {competitionMode ? (
            <div className="composer competitionComposer">
              <button className="iconButton" type="button" onClick={newChat} aria-label="Đoạn chat mới"><SquarePen size={20} /></button>
              <button
                className="fileButton"
                type="button"
                disabled={isSending}
                onClick={() => competitionFileRef.current?.click()}
              >
                <FileUp size={19} /> Chọn file JSON tập test
              </button>
              <input
                ref={competitionFileRef}
                className="hiddenFile"
                type="file"
                accept="application/json,.json"
                onChange={handleCompetitionFile}
              />
            </div>
          ) : (
            <form className="composer" onSubmit={onSubmit}>
              <button className="iconButton" type="button" onClick={newChat} aria-label="Đoạn chat mới"><SquarePen size={20} /></button>
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void submitChat(input);
                  }
                }}
                rows={1}
                placeholder="Hỏi bất kỳ điều gì"
              />
              <button className="sendButton" disabled={isSending || !input.trim()} type="submit" aria-label="Gửi"><Send size={18} /></button>
            </form>
          )}
        </div>
      </section>
    </main>
  );
}
