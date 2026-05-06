import Link from "next/link";

export default function SiteHeader() {
  return (
    <header className="top-bar">
      <div className="brand">
        <span className="brand-mark">RAG</span>
        <div>
          <div className="brand-title">Document Q&A Studio</div>
          <div className="eyebrow">Hybrid retrieval in motion</div>
        </div>
      </div>
      <nav className="nav">
        <Link href="/">Overview</Link>
        <Link href="/upload">Upload</Link>
        <Link href="/ask">Ask</Link>
        <Link href="/analytics">Analytics</Link>
      </nav>
    </header>
  );
}
