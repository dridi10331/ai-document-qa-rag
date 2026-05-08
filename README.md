# 🤖 Instrumented Hybrid RAG System

> **Core Innovation**: RAG systems fail silently due to lack of per-stage observability; this system turns retrieval into a debuggable distributed pipeline where every decision is observable, measurable, and explainable.

Production-style retrieval pipeline for analyzing RAG behavior under constrained compute (512MB RAM).

[Live Demo](https://ai-document-qa-rag.vercel.app) • [API Docs](https://rag-backend-u868.onrender.com/docs)

## What This Solves

Upload documents → Ask questions → Get answers with citations → **Inspect why each chunk was retrieved or ranked**

**Architecture**: Separates recall (hybrid FAISS + BM25), precision (reranking), and observability (full pipeline instrumentation) as independent layers.

## Architecture

```
Query → Expansion → Hybrid Retrieval (FAISS + BM25) → 
Fusion (65/35) → Reranking (cross-encoder/LLM) → 
Generation → Response + Observability Logs
```

## Engineering Contributions

### 1. Instrumented Retrieval Pipeline (Core Differentiator)

Every stage emits structured metrics: chunk scores, fusion values, reranker deltas, latency per stage.

**Logged per request**: Query expansion variants, FAISS scores, BM25 scores, fusion results, reranking deltas, latency breakdown, token usage.

**Enables debugging which stage caused retrieval failure** (expansion, retrieval, fusion, reranking) without labeled ground truth.

**Observability implementation**:
```json
{
  "chunks": [{
    "vector_score": 0.87, "bm25_score": 12.3,
    "fused_score": 0.82, "reranked_score": 0.91
  }],
  "latency_breakdown": {
    "expansion_ms": 280, "retrieval_ms": 150,
    "reranking_ms": 1200, "generation_ms": 890
  }
}
```

**Logging strategy**: Synchronous SQLite writes (acceptable <100 QPS). Production alternative: async message queue (Redis/RabbitMQ) → log aggregation (Datadog/CloudWatch) with backpressure handling.

### 2. Hybrid Retrieval + Reranking Under Memory Constraints

**Problem**: Vector-only missed keyword queries ("Python developer with 5 years"). Cross-encoder reranking requires ~800MB RAM (exceeds 512MB free tier).

**Solution**: Hybrid FAISS + BM25 (65/35 fusion) with automatic LLM reranking fallback.

**Performance benchmarks** (50 queries, 10-doc corpus):
- **Latency**: p50: 920ms, p95: 1850ms, p99: 2400ms
- **Retrieval quality** (manual evaluation, 20 test queries): Precision@5: ~70%, Recall@10: ~85%, Top-1 accuracy: ~60%

**Baseline comparison**:

| Configuration | Keyword Success | Semantic Success | Avg Latency | Top-3 Precision |
|---------------|----------------|------------------|-------------|-----------------|
| Vector-only | 3/10 | 8/10 | 600ms | 5/10 |
| Hybrid (no rerank) | 9/10 | 7/10 | 650ms | 6/10 |
| Hybrid + Rerank | 9/10 | 8/10 | 1200ms | 8/10 |

**Evaluation methodology**: 20 test queries manually labeled for relevance (binary: relevant/not relevant). Queries sampled from CV domain (technical skills, experience, education). Labeling: single annotator, no inter-rater reliability. Variance not measured (small sample size).

**Key insight**: Hybrid retrieval improves recall but destabilizes ranking distribution, requiring reranking to restore consistent top-k ordering.

### 3. Query Expansion Layer

LLM generates alternative phrasings for ambiguous queries. Improves recall (4/10 → 7/10 on ambiguous queries) but adds ~300ms latency and reduces determinism.

## Tech Stack & Constraints

| Layer | Stack | Why |
|-------|-------|-----|
| Vector DB | FAISS | Local memory constraint (no vector DB overhead) |
| LLM | Groq | Cost-free inference |
| Database | SQLite | Single-user, non-concurrent workload |
| Deployment | Render (512MB RAM) | Free tier constraint |

**Constraint → Design Decision**:

| Constraint | Choice | Tradeoff |
|------------|--------|----------|
| 512MB RAM | LLM reranking fallback | +1150ms latency, non-deterministic |
| Free tier | FAISS local | Single-node, no distributed indexing |
| No labeled data | LLM-as-judge eval | Directional debugging only |

**Scaling**: ~10-50 docs (demo scale), O(log n) FAISS, O(n) BM25 scan. **Breaks at**: 1k+ docs (BM25 scan degrades), 10k+ docs (FAISS rebuild blocks queries), concurrent writes (SQLite lock contention).

## Production Architecture Gaps

**Missing for production**:
- **Caching**: Embedding cache (Redis), query cache (LRU)
- **Resilience**: Retry logic for LLM failures, circuit breakers
- **Backpressure**: Rate limiting, request queuing
- **Partial failure recovery**: Fallback to vector-only if BM25 fails
- **Index updates**: Incremental indexing (current: full rebuild on delete)

## Observed Failure Modes

| Failure | Cause | Severity | User Impact |
|---------|-------|----------|-------------|
| Negation queries | BM25 keyword dominance | High (frequent) | Returns wrong answer |
| List fragmentation | Fixed-size chunking | High (frequent) | Incomplete context |
| Multi-hop reasoning | No cross-document layer | Medium (rare, high impact) | Misses relationships |
| Ranking instability | LLM reranker stochasticity | Medium | Inconsistent ranking |

**Failure reproduction** (negation query):

**Query**: "What is NOT covered by the warranty?"

**Before (vector-only)**: Top chunks discuss coverage, not exclusions → wrong answer  
**After (hybrid + rerank)**: Reranking promoted exclusion chunks → correct answer

**Pipeline trace**: Vector search failed (semantic similarity to "covered"), BM25 partial success, reranking fixed (LLM understood negation).

## Why This Matters Beyond This Project

**Real-world system analogy**:
- Similar to **Slack's enterprise search debugging layer**: exposes why certain messages rank higher
- Equivalent to **OpenAI's RAG evaluation stack**: per-stage instrumentation for retrieval quality analysis
- Mirrors **Notion's internal wiki search**: hybrid retrieval with observability

**Applicable to**: Enterprise search debugging (Slack, Notion, Confluence), production RAG systems at Perplexity/You.com, constrained deployment (edge devices, serverless).

## What I Would Do Next in Production

1. **Migrate to vector DB** (Pinecone/Qdrant): Distributed indexing, incremental updates, concurrent writes
2. **Distributed reranking service**: Separate service with autoscaling, cross-encoder model serving
3. **Evaluation harness**: Labeled dataset (100-200 queries), graded relevance judgments, statistical validation
4. **A/B testing framework**: Configuration experimentation (fusion weights, reranker choice), online metrics
5. **Async logging pipeline**: Message queue → log ingestion service → time-series DB (InfluxDB/Prometheus)

## Quick Start

```bash
# Backend
cd backend && pip install -r requirements.txt
cp .env.example .env  # add GROQ_API_KEY
uvicorn app.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

## API

| Endpoint | Purpose |
|----------|---------|
| `/documents/upload` | Upload + chunk + index |
| `/query/stream` | Streaming RAG inference |
| `/eval/retrieval` | Retrieval quality diagnostics |

## Summary

This is an **instrumented retrieval system** for debugging RAG failure modes, not a chatbot.

**Key contribution**: Separating recall, precision, and observability as independent layers enables per-stage debugging without labeled ground truth—a pattern applicable to production retrieval systems.

## License

MIT
