import Link from "next/link";

import SiteHeader from "@/components/SiteHeader";

export default function HomePage() {
  return (
    <div className="app-shell">
      <SiteHeader />
      <main className="stack">
        <section className="hero">
          <p className="eyebrow">RAG studio</p>
          <h1 className="hero-title">Turn dense files into instant answers.</h1>
          <p className="hero-sub">
            Upload multi-format documents, run hybrid retrieval, and chat with
            grounded answers backed by citations. Built for analysts who need
            speed without sacrificing trust.
          </p>
          <div className="cta-row">
            <Link className="btn" href="/upload">
              Upload documents
            </Link>
            <Link className="btn secondary" href="/ask">
              Ask a question
            </Link>
          </div>
        </section>

        <section className="grid">
          <div className="card">
            <div className="card-title">Multi-format ingestion</div>
            <p>PDF, DOCX, TXT, and Markdown with OCR fallback for scans.</p>
          </div>
          <div className="card">
            <div className="card-title">Hybrid retrieval</div>
            <p>Dense vectors + keyword signals for precise answers.</p>
          </div>
          <div className="card">
            <div className="card-title">Streaming Q&A</div>
            <p>Live token streaming and real-time status via WebSockets.</p>
          </div>
          <div className="card">
            <div className="card-title">Analytics ready</div>
            <p>Track popular questions, usage trends, and cost signals.</p>
          </div>
        </section>
      </main>
      <div className="footer">RAG pipeline online: upload, retrieve, stream, analyze.</div>
    </div>
  );
}
