# 🤖 Instrumented Hybrid RAG System

Instrumented hybrid retrieval pipeline designed to analyze retrieval failures, ranking behavior, and latency tradeoffs under constrained compute environments (512MB RAM).

---

# Overview

This system separates retrieval into independent stages:

- Recall → Hybrid retrieval (FAISS + BM25)
- Precision → Reranking
- Observability → Full pipeline instrumentation

The goal is not simply answering questions from documents, but exposing *why* certain chunks were retrieved, ranked, or discarded.

---

# System Architecture

```text
Query
  ↓
Query Expansion
  ↓
Hybrid Retrieval (FAISS + BM25)
  ↓
Score Fusion
  ↓
Reranking
  ↓
Generation + Instrumentation Logs
```

---

# What the System Provides

- Document upload and indexing
- Streaming RAG inference
- Citation-based responses
- Retrieval trace inspection
- Per-stage latency analysis
- Score-level observability

The system exposes how:
- query expansion,
- retrieval,
- fusion,
- reranking

influence final ranking behavior.

---

# Engineering Contributions

## 1. Instrumented Retrieval Pipeline

Every retrieval stage emits structured logs and metrics.

Tracked per request:
- Query expansion variants
- Vector similarity scores
- BM25 scores
- Fusion scores
- Reranking deltas
- Stage latency breakdown
- Token usage

Example instrumentation output:

```json
{
  "chunks": [{
    "vector_score": 0.87,
    "bm25_score": 12.3,
    "fused_score": 0.82,
    "reranked_score": 0.91
  }],
  "latency_breakdown": {
    "expansion_ms": 280,
    "retrieval_ms": 150,
    "reranking_ms": 1200,
    "generation_ms": 890
  }
}
```

This enables inspection of how each retrieval stage contributed to:
- ranking behavior,
- latency,
- retrieval failures

without requiring a fully labeled evaluation pipeline.

### Logging Strategy

Current implementation:
- synchronous SQLite logging
- optimized for low-throughput workloads (<100 QPS)

Production alternative:
- async message queue (Redis / RabbitMQ)
- centralized log aggregation
- backpressure handling
- distributed tracing

---

# 2. Hybrid Retrieval + Reranking

## Problem

Vector-only retrieval struggled with keyword-heavy queries such as:

```text
"Python developer with 5 years experience"
```

Cross-encoder reranking improved ranking quality but exceeded memory limits under the 512MB deployment constraint.

---

## Solution

Implemented:
- FAISS vector retrieval
- BM25 lexical retrieval
- weighted score fusion (65/35)
- reranking fallback using LLM-based scoring

---

## Benchmark Results

Evaluation performed on:
- 50 test queries
- 10-document corpus
- manually labeled relevance set

Latency:
- p50 → 920ms
- p95 → 1850ms
- p99 → 2400ms

Directional evaluation on a small labeled dataset suggested improved retrieval quality after hybrid retrieval and reranking.

| Configuration | Keyword Queries | Semantic Queries | Avg Latency | Top-3 Precision |
|---|---|---|---|---|
| Vector-only | 3/10 | 8/10 | 600ms | 5/10 |
| Hybrid Retrieval | 9/10 | 7/10 | 650ms | 6/10 |
| Hybrid + Reranking | 9/10 | 8/10 | 1200ms | 8/10 |

### Key Observation

Hybrid retrieval improved recall but destabilized ranking consistency, making reranking necessary for reliable top-k ordering.

---

# 3. Query Expansion Layer

LLM-generated query reformulation was used to improve retrieval recall for ambiguous or underspecified prompts.

Observed behavior:
- improved ambiguous query retrieval
- increased latency (~300ms)
- reduced determinism

Tradeoff:
higher recall at the cost of inference stability and response time.

---

# System Constraints and Design Decisions

| Constraint | Design Choice | Tradeoff |
|---|---|---|
| 512MB RAM | LLM reranking fallback | Higher latency |
| Free-tier deployment | Local FAISS index | Single-node architecture |
| No labeled dataset | Directional evaluation only | Limited statistical confidence |
| SQLite logging | Simplicity and reliability | Write contention at scale |

---

# Known Failure Modes

| Failure Mode | Cause | Impact |
|---|---|---|
| Negation queries | BM25 keyword dominance | Incorrect retrieval |
| List fragmentation | Fixed-size chunking | Partial context loss |
| Multi-hop reasoning | No cross-document reasoning layer | Missed relationships |
| Ranking instability | Stochastic reranking | Non-deterministic ordering |

---

# Example Failure Analysis

## Query

```text
"What is NOT covered by the warranty?"
```

## Observed Behavior

### Vector-only Retrieval
Retrieved chunks discussing warranty coverage instead of exclusions.

### Hybrid + Reranking
BM25 partially surfaced exclusion-related chunks.
Reranking corrected final ordering by prioritizing negation-aware results.

### Pipeline Observation

- vector retrieval failed semantically
- BM25 partially recovered lexical intent
- reranking corrected final ranking

---

# Scaling Limitations

Current architecture supports:
- ~10–50 documents
- single-user workloads
- low-concurrency inference

Known scaling bottlenecks:
- BM25 linear scan degradation at larger corpus sizes
- FAISS index rebuild cost
- SQLite write contention
- synchronous logging overhead

---

# Missing for Production

The current system intentionally prioritizes observability and constrained deployment experimentation over production scalability.

Missing components include:
- embedding cache
- query cache
- retry logic
- circuit breakers
- request queueing
- distributed indexing
- incremental index updates
- autoscaling reranking services

---

# Future Improvements

Potential production upgrades:
- Pinecone or Qdrant for distributed indexing
- async observability pipeline
- retrieval evaluation harness
- A/B testing framework
- Prometheus/Grafana metrics
- distributed reranking service

---

# Tech Stack

| Layer | Technology |
|---|---|
| Vector Retrieval | FAISS |
| Lexical Retrieval | BM25 |
| LLM Inference | Groq |
| Backend | FastAPI |
| Database | SQLite |
| Deployment | Render |

---

# API Endpoints

| Endpoint | Purpose |
|---|---|
| `/documents/upload` | Upload and index documents |
| `/query/stream` | Streaming RAG inference |
| `/eval/retrieval` | Retrieval diagnostics |

---

# Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt

cp .env.example .env
# Add GROQ_API_KEY

uvicorn app.main:app --reload
```

```bash
# Frontend
cd frontend

npm install
npm run dev
```

---

# Summary

This project focuses on retrieval observability and ranking analysis rather than chatbot functionality.

The core idea is that:
- retrieval quality,
- ranking behavior,
- latency tradeoffs,
- and failure modes

should be inspectable and measurable at every stage of the pipeline.

---

# License

MIT
````

