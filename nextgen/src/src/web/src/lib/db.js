// Thao tác lịch sử hội thoại trên Supabase (RLS đảm bảo mỗi user chỉ thấy dữ liệu của mình).
// Mọi hàm đều "best-effort": lỗi DB không được làm hỏng trải nghiệm chat.

export async function listConversations(sb, userId) {
  const { data, error } = await sb
    .from("conversations")
    .select("id, title, updated_at")
    .eq("user_id", userId)
    .order("updated_at", { ascending: false })
    .limit(100);
  if (error) {
    console.warn("[db] listConversations:", error.message);
    return [];
  }
  return data || [];
}

export async function createConversation(sb, userId, title) {
  const { data, error } = await sb
    .from("conversations")
    .insert({ user_id: userId, title: (title || "Cuộc trò chuyện mới").slice(0, 120) })
    .select("id, title, updated_at")
    .single();
  if (error) {
    console.warn("[db] createConversation:", error.message);
    return null;
  }
  return data;
}

export async function loadMessages(sb, conversationId) {
  const { data, error } = await sb
    .from("messages")
    .select("role, content, citations, created_at")
    .eq("conversation_id", conversationId)
    .order("created_at", { ascending: true });
  if (error) {
    console.warn("[db] loadMessages:", error.message);
    return [];
  }
  return data || [];
}

export async function addMessage(sb, conversationId, role, content, citations) {
  const { error } = await sb.from("messages").insert({
    conversation_id: conversationId,
    role,
    content,
    citations: citations && citations.length ? citations : null,
  });
  if (error) console.warn("[db] addMessage:", error.message);
  // đụng vào updated_at để hội thoại nhảy lên đầu danh sách
  await sb.from("conversations").update({ updated_at: new Date().toISOString() }).eq("id", conversationId);
}

export async function deleteConversation(sb, conversationId) {
  const { error } = await sb.from("conversations").delete().eq("id", conversationId);
  if (error) console.warn("[db] deleteConversation:", error.message);
}
