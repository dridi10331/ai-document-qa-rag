"use client";

import { useEffect, useRef, useState } from "react";
import { getWebsocketBase, uploadDocuments, UploadResponse } from "@/lib/api";

export default function UploadPanel() {
  const [files, setFiles] = useState<File[]>([]);
  const [status, setStatus] = useState<string>("Idle");
  const [results, setResults] = useState<UploadResponse[] | null>(null);
  const [statusMap, setStatusMap] = useState<Record<string, string>>({});
  const socketsRef = useRef<WebSocket[]>([]);

  useEffect(() => {
    return () => {
      socketsRef.current.forEach((socket) => socket.close());
      socketsRef.current = [];
    };
  }, []);

  const handleFiles = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextFiles = event.target.files
      ? Array.from(event.target.files)
      : [];
    setFiles(nextFiles);
  };

  const handleUpload = async () => {
    if (!files.length) {
      setStatus("Select at least one file.");
      return;
    }

    try {
      setStatus("Uploading and chunking...");
      const payload = await uploadDocuments(files);
      setResults(payload);
      payload.forEach((item) => {
        const socket = new WebSocket(
          `${getWebsocketBase()}/ws/documents/${item.document.id}`
        );
        socket.onmessage = (event) => {
          const data = JSON.parse(event.data);
          const detail = data.detail ? ` - ${data.detail}` : "";
          setStatusMap((prev) => ({
            ...prev,
            [item.document.id]: `${data.status}${detail}`
          }));
        };
        socketsRef.current.push(socket);
      });
      setStatus("Upload complete.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Upload failed";
      setStatus(message);
    }
  };

  return (
    <section className="panel">
      <h2 className="panel-title">Upload documents</h2>
      <p className="hero-sub">
        Drop PDFs, DOCX, TXT, or Markdown. The backend will parse and chunk
        instantly.
      </p>
      <input className="file" type="file" multiple onChange={handleFiles} />
      <div className="cta-row">
        <button className="btn" onClick={handleUpload}>
          Ingest files
        </button>
        <span className="status">{status}</span>
      </div>
      {results && (
        <div className="results">
          {results.map((item) => (
            <div className="result-card" key={item.document.id}>
              <div className="kv">
                <strong>{item.document.filename}</strong>
                <span>{item.document.status}</span>
              </div>
              <div className="kv">
                <span>Pages</span>
                <strong>{item.document.page_count}</strong>
              </div>
              <div className="kv">
                <span>Chunks</span>
                <strong>{item.document.chunk_count}</strong>
              </div>
              <div className="kv">
                <span>Status</span>
                <strong>
                  {statusMap[item.document.id] || item.document.status}
                </strong>
              </div>
              {item.warnings.length > 0 && (
                <div className="status">{item.warnings.join(" ")}</div>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
