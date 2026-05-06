"use client";

import { useEffect, useRef, useState } from "react";
import {
  askQuestionStream,
  createChatSession,
  listDocuments,
  DocumentOut,
  StreamDonePayload
} from "@/lib/api";

type Message = { role: "user" | "assistant"; content: string };

export default function AskPanel() {
  const [query, setQuery] = useState<string>("");
  const [status, setStatus] = useState<string>("Idle");
  const [answer, setAnswer] = useState<string>("");
  const [response, setResponse] = useState<StreamDonePayload | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<Record<string, boolean>>({});
  const [messages, setMessages] = useState<Message[]>([]);
  const cancelRef = useRef<() => void>();

  useEffect(() => {
    createChatSession("New chat")
      .then((session) => setSessionId(session.id))
      .catch(() => setSessionId(null));

    listDocuments()
      .then((docs) => {
        setDocuments(docs);
        const map: Record<string, boolean> = {};
        docs.forEach((doc) => {
          map[doc.id] = true;
        });
        setSelectedDocs(map);
      })
      .catch(() => setDocuments([]));

    return () => {
      cancelRef.current?.();
    };
  }, []);

  const handleToggleDoc = (docId: string) => {
    setSelectedDocs((prev) => ({ ...prev, [docId]: !prev[docId] }));
  };

  const handleAsk = () => {
    if (!query.trim()) {
      setStatus("Enter a question.");
      return;
    }

    cancelRef.current?.();
    setAnswer("");
    setResponse(null);
    setStatus("Streaming answer...");

    const chosenDocs = Object.entries(selectedDocs)
      .filter(([, enabled]) => enabled)
      .map(([docId]) => docId);

    cancelRef.current = askQuestionStream(
      query,
      {
        top_k: 6,
        session_id: sessionId,
        document_ids: chosenDocs
      },
      {
        onToken: (token) => setAnswer((prev) => prev + token),
        onDone: (payload) => {
          setResponse(payload);
          setStatus("Answer ready.");
          setSessionId(payload.session_id || sessionId);
          setMessages((prev) => [
            ...prev,
            { role: "user", content: query },
            { role: "assistant", content: payload.answer }
          ]);
          setAnswer("");
        },
        onError: (message) => setStatus(message)
      }
    );
  };

  return (
    <section className="panel">
      <h2 className="panel-title">Ask your docs</h2>
      <p className="hero-sub">
        Streaming answers with citations, multi-document filters, and chat
        history.
      </p>

      <div className="filter-panel">
        <div className="filter-title">Filter documents</div>
        <div className="filter-grid">
          {documents.map((doc) => (
            <label key={doc.id} className="filter-item">
              <input
                type="checkbox"
                checked={selectedDocs[doc.id] ?? true}
                onChange={() => handleToggleDoc(doc.id)}
              />
              <span>{doc.filename}</span>
            </label>
          ))}
        </div>
      </div>

      <textarea
        className="textarea"
        placeholder="Ask about the contract terms, policy clauses, or key findings..."
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      <div className="cta-row">
        <button className="btn" onClick={handleAsk}>
          Ask now
        </button>
        <span className="status">{status}</span>
      </div>

      <div className="chat-shell">
        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={`chat-bubble ${message.role}`}
          >
            <strong>{message.role === "user" ? "You" : "Assistant"}</strong>
            <p>{message.content}</p>
          </div>
        ))}
        {answer && (
          <div className="chat-bubble assistant">
            <strong>Assistant</strong>
            <p>{answer}</p>
          </div>
        )}
      </div>

      {response && (
        <div className="results">
          <div className="result-card">
            <div className="kv">
              <span>Latency</span>
              <strong>{response.latency_ms?.toFixed(0)} ms</strong>
            </div>
            <div className="kv">
              <span>Tokens</span>
              <strong>
                {response.usage?.tokens_in || 0} in / {response.usage?.tokens_out || 0} out
              </strong>
            </div>
            <div className="kv">
              <span>Cost</span>
              <strong>${response.usage?.cost_estimate?.toFixed(4) || "0.0000"}</strong>
            </div>
            {response.expanded_query && (
              <div className="status">Expanded: {response.expanded_query}</div>
            )}
          </div>
          {response.citations.length > 0 && (
            <div className="result-card">
              <strong>Citations</strong>
              <div className="results">
                {response.citations.map((cite, idx) => (
                  <div key={`${cite.document_id}-${idx}`} className="citation">
                    <div className="kv">
                      <span>{cite.document_name || "document"}</span>
                      <strong>p.{cite.page_number ?? "-"}</strong>
                    </div>
                    <div className="status">{cite.text}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
