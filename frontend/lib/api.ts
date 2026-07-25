export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  if (API_KEY) {
    headers["X-Api-Key"] = API_KEY;
  }
  return headers;
}

export type DocumentOut = {
  id: string;
  filename: string;
  status: string;
  size_bytes: number;
  page_count: number;
  chunk_count: number;
  created_at: string;
};

export type DocumentDetailOut = DocumentOut & {
  content_type?: string | null;
  updated_at: string;
  status_detail?: string | null;
};

export type DocumentChunkOut = {
  id: string;
  document_id: string;
  chunk_index: number;
  text: string;
  page_number?: number | null;
  metadata: Record<string, unknown>;
};

export type UploadResponse = {
  document: DocumentOut;
  warnings: string[];
};

export async function uploadDocuments(files: File[]): Promise<UploadResponse[]> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const response = await fetch(`${API_BASE}/documents/upload`, {
    method: "POST",
    headers: authHeaders(),
    body: formData
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const message = payload?.detail || "Upload failed";
    throw new Error(message);
  }

  return response.json();
}

export type QueryResponse = {
  session_id?: string | null;
  expanded_query?: string | null;
  answer: string;
  citations: Array<{
    document_id: string;
    document_name?: string | null;
    chunk_id?: string | null;
    chunk_index: number;
    page_number?: number | null;
    score?: number | null;
    text?: string | null;
  }>;
  latency_ms?: number;
  model?: string | null;
  usage?: { tokens_in?: number | null; tokens_out?: number | null; cost_estimate?: number | null };
};

export type StreamDonePayload = QueryResponse;

export type AnalyticsSummary = {
  query_count: number;
  avg_latency_ms?: number | null;
  top_questions: Array<{ query: string; count: number }>;
  top_documents: Array<{ document_id: string; filename: string; usage_count: number }>;
  total_cost: number;
};

export type ChatSession = {
  id: string;
  title?: string | null;
  created_at: string;
};

export type ChatMessage = {
  id: string;
  session_id: string;
  role: string;
  content: string;
  created_at: string;
};

export async function askQuestion(query: string): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ query })
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const message = payload?.detail || "Query failed";
    throw new Error(message);
  }

  return response.json();
}

export async function listDocuments(): Promise<DocumentOut[]> {
  const response = await fetch(`${API_BASE}/documents`);
  if (!response.ok) {
    throw new Error("Failed to load documents");
  }
  return response.json();
}

export async function getAnalyticsSummary(): Promise<AnalyticsSummary> {
  const response = await fetch(`${API_BASE}/analytics/summary`);
  if (!response.ok) {
    throw new Error("Failed to load analytics");
  }
  return response.json();
}

export async function createChatSession(title?: string): Promise<ChatSession> {
  const response = await fetch(`${API_BASE}/sessions`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ title })
  });
  if (!response.ok) {
    throw new Error("Failed to create session");
  }
  return response.json();
}

export async function listChatSessions(): Promise<ChatSession[]> {
  const response = await fetch(`${API_BASE}/sessions`);
  if (!response.ok) {
    throw new Error("Failed to load sessions");
  }
  return response.json();
}

export async function listChatMessages(sessionId: string): Promise<ChatMessage[]> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/messages`);
  if (!response.ok) {
    throw new Error("Failed to load messages");
  }
  return response.json();
}

export type StreamHandlers = {
  onToken: (token: string) => void;
  onDone: (payload: StreamDonePayload) => void;
  onError?: (message: string) => void;
};

export function askQuestionStream(
  query: string,
  options: {
    top_k?: number;
    session_id?: string | null;
    document_ids?: string[] | null;
  },
  handlers: StreamHandlers
): () => void {
  const params = new URLSearchParams({ query });
  if (options.top_k) params.set("top_k", String(options.top_k));
  if (options.session_id) params.set("session_id", options.session_id);
  if (options.document_ids && options.document_ids.length > 0) {
    params.set("document_ids", options.document_ids.join(","));
  }
  if (API_KEY) params.set("api_key", API_KEY);

  const source = new EventSource(`${API_BASE}/query/stream?${params.toString()}`);

  source.addEventListener("token", (event) => {
    const data = JSON.parse((event as MessageEvent).data);
    handlers.onToken(data.token);
  });

  source.addEventListener("done", (event) => {
    const data = JSON.parse((event as MessageEvent).data);
    handlers.onDone(data);
    source.close();
  });

  source.onerror = () => {
    handlers.onError?.("Streaming connection failed");
    source.close();
  };

  return () => source.close();
}

export function getWebsocketBase(): string {
  let base: string;
  if (API_BASE.startsWith("https")) {
    base = API_BASE.replace("https", "wss");
  } else {
    base = API_BASE.replace("http", "ws");
  }
  return API_KEY ? `${base}?api_key=${encodeURIComponent(API_KEY)}` : base;
}
