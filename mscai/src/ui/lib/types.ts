export type RuntimeConfig = {
  chat_streaming: boolean;
  token_streaming: boolean;
  competition_enabled: boolean;
};

export type LegalArticle = {
  id: string;
  article_id: string;
  law_id: string;
  law_name: string;
  doc_type: string;
  database?: string;
  category?: string;
  chapter?: string | null;
  article: string;
  article_title?: string | null;
  content: string;
  author?: string | null;
  extra?: string[];
  score?: number | null;
};

export type LegalAnswerResponse = {
  id?: number | null;
  session_id?: string | null;
  question: string;
  answer: string;
  relevant_docs: string[];
  relevant_articles: string[];
  selected_articles: LegalArticle[];
  debug: Record<string, unknown>;
};

export type CompetitionRecord = {
  id?: number | null;
  question: string;
  answer: string;
  relevant_docs: string[];
  relevant_articles: string[];
};

export type ChatResponse = {
  session_id?: string | null;
  message: string;
  answer: LegalAnswerResponse;
  tool_calls: Record<string, unknown>[];
};

export type ChatStreamProgress = {
  message: string;
  stage: string;
  status: "started" | "running" | "completed" | "warning" | "error";
  elapsed_ms?: number | null;
  detail?: string | null;
  metadata?: Record<string, unknown>;
};

export type ChatStreamEvent =
  | { event: "status"; data: ChatStreamProgress }
  | { event: "token"; data: { token: string; stage?: string } }
  | { event: "result"; data: ChatResponse }
  | { event: "done"; data: ChatStreamProgress }
  | { event: "error"; data: ChatStreamProgress };

export type CompetitionStreamEvent =
  | { event: "status"; data: ChatStreamProgress }
  | { event: "competition_item_result"; data: CompetitionRecord & { index: number; total: number } }
  | { event: "competition_result"; data: CompetitionRecord[] }
  | { event: "done"; data: ChatStreamProgress }
  | { event: "error"; data: ChatStreamProgress };

export type RetrievalTraceResult = {
  rank?: number;
  score?: number;
  source?: string;
  law_id?: string;
  law_name?: string;
  article?: string;
  article_title?: string;
  passed_threshold?: boolean;
};

export type AgentTraceStep = {
  stage?: string;
  status?: "started" | "running" | "completed" | "warning" | "error";
  title: string;
  detail: string;
  elapsedMs?: number | null;
  tone?: "info" | "success" | "warning" | "error";
  topResults?: RetrievalTraceResult[];
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  stream: string[];
  trace: AgentTraceStep[];
  sources: string[];
};

export type Conversation = {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
};
