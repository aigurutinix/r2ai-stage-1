import { useEffect, useRef, useState } from "react";
import { askStream } from "./lib/sse.js";
import { getSupabase } from "./lib/supabase.js";
import * as db from "./lib/db.js";
import WireGrid from "./components/WireGrid.jsx";
import ChatMessage from "./components/ChatMessage.jsx";
import Composer from "./components/Composer.jsx";
import CitationPanel from "./components/CitationPanel.jsx";
import Sidebar from "./components/Sidebar.jsx";
import SettingsModal from "./components/SettingsModal.jsx";

const GREETING =
  "Chào bạn, tôi là **Trợ lý NextGen** — trợ lý AI hỗ trợ phân tích và trả lời câu hỏi về " +
  "pháp luật Việt Nam cho doanh nghiệp nhỏ và vừa. Hãy đặt câu hỏi pháp lý bất kỳ, " +
  "tôi sẽ trả lời kèm **trích dẫn văn bản** liên quan.";

const EXAMPLE_QUESTIONS = [
  "Doanh nghiệp nhỏ và vừa được hỗ trợ thuế như thế nào?",
  "Hộ kinh doanh chuyển đổi thành doanh nghiệp được hỗ trợ gì?",
  "Điều kiện được hỗ trợ pháp lý cho doanh nghiệp nhỏ và vừa?",
  "Doanh nghiệp nhỏ và vừa được hỗ trợ mặt bằng sản xuất không?",
];

const freshMessages = () => [{ role: "assistant", content: GREETING, greeting: true }];

function normalizeUser(u) {
  if (!u) return null;
  const m = u.user_metadata || {};
  return {
    id: u.id,
    email: u.email,
    name: m.full_name || m.name || u.email,
    avatar: m.avatar_url || m.picture || null,
  };
}

export default function App() {
  const [sb, setSb] = useState(null);
  const [user, setUser] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [currentConvId, setCurrentConvId] = useState(null);

  const [messages, setMessages] = useState(freshMessages);
  const [citations, setCitations] = useState([]);
  const [panelOpen, setPanelOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const scrollRef = useRef(null);

  // --- Khởi tạo Supabase + theo dõi phiên đăng nhập ---
  useEffect(() => {
    let unsub = null;
    getSupabase().then((client) => {
      if (!client) return;
      setSb(client);
      client.auth.getSession().then(({ data }) => setUser(normalizeUser(data?.session?.user)));
      const { data: sub } = client.auth.onAuthStateChange((_e, session) =>
        setUser(normalizeUser(session?.user))
      );
      unsub = sub?.subscription;
    });
    return () => unsub?.unsubscribe?.();
  }, []);

  // --- Nạp danh sách hội thoại khi đăng nhập ---
  const refreshConversations = async (u = user) => {
    if (sb && u) setConversations(await db.listConversations(sb, u.id));
  };
  useEffect(() => {
    if (sb && user) refreshConversations(user);
    else setConversations([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sb, user?.id]);

  const scrollToEnd = () =>
    requestAnimationFrame(() => {
      const el = scrollRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    });

  const setLastAssistant = (updater) =>
    setMessages((m) => {
      const copy = [...m];
      for (let i = copy.length - 1; i >= 0; i--) {
        if (copy[i].role === "assistant") {
          copy[i] = { ...copy[i], ...updater(copy[i]) };
          break;
        }
      }
      return copy;
    });

  // --- Auth actions ---
  const signIn = () =>
    sb?.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: window.location.origin },
    });
  const signOut = async () => {
    await sb?.auth.signOut();
    setUser(null);
    setConversations([]);
    newChat();
    setSettingsOpen(false);
  };

  // --- Conversation actions ---
  function newChat() {
    setCurrentConvId(null);
    setMessages(freshMessages());
    setCitations([]);
    setPanelOpen(false);
    setSidebarOpen(false);
  }

  async function selectConversation(id) {
    setSidebarOpen(false);
    if (!sb) return;
    setCurrentConvId(id);
    const rows = await db.loadMessages(sb, id);
    setMessages([
      ...freshMessages(),
      ...rows.map((r) => ({ role: r.role, content: r.content })),
    ]);
    // khôi phục citations của câu trả lời cuối để nút "Văn bản liên quan" dùng lại
    const lastWithCite = [...rows].reverse().find((r) => r.role === "assistant" && r.citations);
    setCitations(lastWithCite?.citations || []);
    setPanelOpen(false);
    scrollToEnd();
  }

  async function removeConversation(id) {
    if (!sb) return;
    await db.deleteConversation(sb, id);
    await refreshConversations();
    if (id === currentConvId) newChat();
  }

  // --- Gửi câu hỏi ---
  async function handleSend(question) {
    if (busy) return;
    setMessages((m) => [
      ...m,
      { role: "user", content: question },
      { role: "assistant", content: "", streaming: true },
    ]);
    setCitations([]);
    setBusy(true);
    setStatus("Đang truy hồi văn bản liên quan…");
    setSidebarOpen(false);
    scrollToEnd();

    // Đảm bảo có conversation + lưu câu hỏi (chỉ khi đã đăng nhập)
    let convId = currentConvId;
    if (sb && user) {
      if (!convId) {
        const conv = await db.createConversation(sb, user.id, question);
        if (conv) {
          convId = conv.id;
          setCurrentConvId(conv.id);
        }
      }
      if (convId) {
        await db.addMessage(sb, convId, "user", question, null);
        refreshConversations();
      }
    }

    let pendingCitations = [];
    askStream(question, {
      onCitations: (payload) => {
        const cites = payload.citations || [];
        setCitations(cites);
        pendingCitations = cites;
        if (cites.length && window.innerWidth > 900) setPanelOpen(true);
        setStatus("Đang soạn câu trả lời…");
      },
      onToken: (delta) => {
        setLastAssistant((prev) => ({ content: prev.content + delta }));
        scrollToEnd();
      },
      onDone: async (payload) => {
        const finalAnswer = payload.answer || "";
        setLastAssistant((prev) => ({
          content: finalAnswer || prev.content,
          streaming: false,
        }));
        setBusy(false);
        setStatus("");
        scrollToEnd();
        if (sb && user && convId) {
          await db.addMessage(sb, convId, "assistant", finalAnswer, pendingCitations);
          refreshConversations();
        }
      },
      onError: (msg) => {
        setLastAssistant((prev) => ({
          content: prev.content || `⚠️ ${msg}`,
          streaming: false,
          error: true,
        }));
        setBusy(false);
        setStatus(`Lỗi: ${msg}`);
      },
    });
  }

  return (
    <div className="app">
      <WireGrid />

      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        user={user}
        conversations={conversations}
        currentConvId={currentConvId}
        onNewChat={newChat}
        onSelect={selectConversation}
        onDelete={removeConversation}
        onSignIn={signIn}
        onSignOut={signOut}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      <div className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button className="hamburger" onClick={() => setSidebarOpen(true)} aria-label="Menu">☰</button>
            <div className="topbar-brand">Next<span className="brand-x">Gen</span></div>
          </div>
          {!panelOpen && citations.length > 0 && (
            <button className="reopen" onClick={() => setPanelOpen(true)}>
              📖 Văn bản liên quan ({citations.length})
            </button>
          )}
        </header>

        <div className="body">
          <main className="chat">
            <div className="messages" ref={scrollRef}>
              {messages.map((m, i) => (
                <ChatMessage
                  key={i}
                  role={m.role}
                  content={m.content}
                  streaming={m.streaming}
                  error={m.error}
                  greeting={m.greeting}
                />
              ))}

              {messages.length === 1 && !busy && (
                <div className="chips">
                  {EXAMPLE_QUESTIONS.map((q) => (
                    <button key={q} className="chip" onClick={() => handleSend(q)}>
                      {q}
                    </button>
                  ))}
                </div>
              )}

              {status && <div className="status">{status}</div>}
            </div>

            <Composer onSend={handleSend} busy={busy} />
            <div className="trust">
              🔒 Trả lời dựa trên văn bản pháp luật chính thức · vui lòng đối chiếu bản gốc trước khi áp dụng.
            </div>
          </main>

          <CitationPanel
            open={panelOpen}
            citations={citations}
            onClose={() => setPanelOpen(false)}
          />
        </div>
      </div>

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        user={user}
        onSignOut={signOut}
      />
    </div>
  );
}
