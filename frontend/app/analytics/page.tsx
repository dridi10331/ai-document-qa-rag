"use client";

import { useEffect, useState } from "react";
import SiteHeader from "@/components/SiteHeader";
import { AnalyticsSummary, getAnalyticsSummary } from "@/lib/api";

export default function AnalyticsPage() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);

  useEffect(() => {
    getAnalyticsSummary()
      .then(setSummary)
      .catch(() => setSummary(null));
  }, []);

  return (
    <div className="app-shell">
      <SiteHeader />
      <main className="stack">
        <section className="panel">
          <h2 className="panel-title">Analytics dashboard</h2>
          <p className="hero-sub">
            Live telemetry for query volume, top questions, and document usage.
          </p>
          <div className="grid">
            <div className="card">
              <div className="card-title">Total queries</div>
              <p>{summary?.query_count ?? 0}</p>
            </div>
            <div className="card">
              <div className="card-title">Avg latency</div>
              <p>{summary?.avg_latency_ms?.toFixed(0) ?? 0} ms</p>
            </div>
            <div className="card">
              <div className="card-title">Estimated cost</div>
              <p>${summary?.total_cost?.toFixed(4) ?? "0.0000"}</p>
            </div>
          </div>
        </section>

        <section className="panel">
          <h3 className="panel-title">Top questions</h3>
          <div className="results">
            {(summary?.top_questions ?? []).map((item) => (
              <div className="result-card" key={item.query}>
                <div className="kv">
                  <span>{item.query}</span>
                  <strong>{item.count}</strong>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="panel">
          <h3 className="panel-title">Top documents</h3>
          <div className="results">
            {(summary?.top_documents ?? []).map((item) => (
              <div className="result-card" key={item.document_id}>
                <div className="kv">
                  <span>{item.filename}</span>
                  <strong>{item.usage_count}</strong>
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
