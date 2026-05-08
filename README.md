# 🤖 AI-Powered Document Q&A System

> A deployed full-stack RAG platform for intelligent multi-document question answering with real-time streaming, hybrid search, and analytics.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat&logo=next.js)](https://nextjs.org/)
[![Groq](https://img.shields.io/badge/Groq-Free%20LLM-orange?style=flat)](https://console.groq.com/)
[![Vercel](https://img.shields.io/badge/Vercel-Deployed-black?style=flat&logo=vercel)](https://ai-document-qa-rag.vercel.app)
[![Render](https://img.shields.io/badge/Render-Deployed-blue?style=flat)](https://rag-backend-u868.onrender.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌐 Live Demo

- **Frontend**: https://ai-document-qa-rag.vercel.app
- **Backend API**: https://rag-backend-u868.onrender.com
- **API Docs**: https://rag-backend-u868.onrender.com/docs

> ⚠️ Free tier: backend may take ~50s to wake up on first request.

## ✨ Features

### 🔍 Document Processing Pipeline
- **Multi-format Support**: PDF, DOCX, TXT, Markdown with optional OCR
- **Sliding Window Chunking**: Configurable chunk size and overlap
- **Metadata Extraction**: Page numbers and document structure

### 🎯 Hybrid Retrieval Pipeline
- **FAISS Vector Search**: Semantic similarity using Groq embeddings (`nomic-embed-text-v1.5`)
- **BM25 Keyword Search**: Lexical matching with rank-bm25
- **Score Fusion**: Weighted combination (65% vector + 35% BM25)
- **Query Expansion**: Groq-powered alternative phrasings for better recall
- **Reranking**: Cross-encoder (`ms-marco-MiniLM-L-6-v2`) or LLM-based with automatic fallback

> **Reranking**: Uses cross-encoder model by default (~50ms, deterministic). Falls back to Groq LLM if model unavailable. Configurable via `RERANKER_TYPE` env var (none/cross_encoder/llm/auto). See `scripts/benchmark_rerankers.py` for performance comparison.

### 💬 Real-time Q&A
- **Streaming Responses**: Server-Sent Events (SSE) for token-by-token output
- **WebSocket Status**: Live document processing updates
- **Chat History**: Persistent conversation context per session
- **Citations**: Source attribution with page numbers and relevance scores

### 📊 Analytics & Evaluation
- **Query Logs**: Latency, token usage, model used per query
- **Document Usage**: Most referenced documents
- **Retrieval Evaluation**: LLM-judged Precision@k, MRR, rerank gain via `/eval/retrieval`
- **Pipeline Introspection**: Query expansion variants, reranking applied, score distribution

> **Evaluation note**: `/eval/retrieval` uses Groq as a relevance judge (no labeled ground truth needed). This is useful for quick iteration but not scientifically rigorous. For production evaluation, you need labeled QA datasets with known relevant chunks to compute objective IR metrics.

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Next.js 14    │─────▶│   FastAPI        │─────▶│  Groq API        │
│  (Vercel)       │      │   (Render)       │      │  LLM + Embeddings│
└─────────────────┘      └──────────────────┘      └──────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
               FAISS Index    BM25 Index    SQLite DB
             (vector search) (keyword)    (metadata)
```

### Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| **Backend** | FastAPI + Python 3.11 | Async, SSE streaming |
| **Frontend** | Next.js 14 + TypeScript | App router, SSE client |
| **LLM** | Groq `llama-3.1-8b-instant` | Free tier, ~1s latency |
| **Embeddings** | Groq `nomic-embed-text-v1.5` | Real semantic embeddings |
| **Vector DB** | FAISS (local) | In-memory, persisted to disk |
| **Keyword Search** | BM25 | Hybrid retrieval |
| **Database** | SQLite | Suitable for demo scale |
| **Hosting** | Vercel + Render (free tier) | Cold start on free plan |

## ⚠️ Known Limitations

This is a **demo-scale deployment**, not a production system. Missing for true production:
- Authentication & authorization
- Rate limiting & abuse prevention
- Async ingestion queue (Celery/Redis) — uploads currently block the request lifecycle
- Persistent vector store (Qdrant/Weaviate/pgvector) — FAISS is single-node, no concurrent writes
- Labeled evaluation dataset — current eval uses LLM as judge, not ground truth
- Multi-tenant document isolation
- CI/CD pipeline

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.11+
- Node.js 18+
- Free [Groq API key](https://console.groq.com/keys)

### 1️⃣ Backend

```bash
cd backend
pip install -r requirements.txt
copy .env.example .env
# Set GROQ_API_KEY in .env
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2️⃣ Frontend

```bash
cd frontend
npm install
copy .env.local.example .env.local
# Set NEXT_PUBLIC_API_BASE=http://localhost:8000
npm run dev
```

Open http://localhost:3000

## 📁 Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/routes.py        # All API endpoints
│   │   ├── core/config.py       # Settings (pydantic-settings)
│   │   ├── db/                  # SQLModel models + CRUD
│   │   ├── schemas/             # Pydantic request/response schemas
│   │   └── services/
│   │       ├── llm.py           # Groq/Ollama LLM integration
│   │       ├── embeddings.py    # Groq/HF/mock embeddings
│   │       ├── retrieval.py     # Hybrid FAISS + BM25 retrieval
│   │       ├── chunking.py      # Sliding window chunking
│   │       ├── query_expansion.py # LLM query reformulation
│   │       └── analytics.py     # Usage tracking
│   ├── tests/
│   ├── .python-version          # Python 3.11.9
│   └── requirements.txt
├── frontend/
│   ├── app/                     # Next.js app router pages
│   ├── components/              # Upload, Ask, Analytics panels
│   └── lib/api.ts               # Typed API client + SSE
└── deploy/
    ├── docker-compose.yml
    ├── backend.Dockerfile
    ├── frontend.Dockerfile
    └── k8s/                     # Basic K8s manifests (not battle-tested)
```

## 🔧 Environment Variables

### Backend (`.env`)

```env
# LLM
LLM_BACKEND=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Embeddings (groq = real semantic, mock = deterministic hash)
EMBEDDINGS_BACKEND=groq

# CORS
CORS_ORIGINS_STR=http://localhost:3000

# Retrieval
ENABLE_HYBRID_SEARCH=true
ENABLE_QUERY_EXPANSION=true
BM25_WEIGHT=0.35
VECTOR_WEIGHT=0.65
```

### Frontend (`.env.local`)

```env
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

## 📚 API Reference

Full docs: https://rag-backend-u868.onrender.com/docs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload + parse + chunk + index |
| `GET` | `/documents` | List documents |
| `DELETE` | `/documents/{id}` | Delete + rebuild index |
| `POST` | `/query` | RAG query (blocking) |
| `GET` | `/query/stream` | RAG query (SSE streaming) |
| `GET` | `/analytics/summary` | Usage analytics |
| `POST` | `/eval/retrieval` | Precision@k, MRR, rerank gain, LLM relevance scores |
| `WS` | `/ws/documents/{id}` | Processing status |

## 🧪 Tests

```bash
cd backend
pytest
pytest --cov=app tests/
```

### Reranker Benchmark

Compare reranking strategies (none, cross-encoder, LLM):

```bash
cd backend
python -m scripts.benchmark_rerankers --queries 5 --runs 3
```

This measures:
- **Latency**: Average response time per strategy
- **Precision@k**: Fraction of top-k results that are relevant
- **NDCG@k**: Ranking quality metric
- **Determinism**: Score consistency across runs

Example output:
```
NONE:
  Latency:      2.3ms
  Precision@3:  0.667
  NDCG@5:       0.789

CROSS_ENCODER:
  Latency:      48.5ms
  Precision@3:  0.867
  NDCG@5:       0.912

LLM:
  Latency:      1247.3ms
  Precision@3:  0.833
  NDCG@5:       0.895
```

**Key findings:**
- Cross-encoder: ~20% better ranking quality, ~20x faster than LLM
- LLM: Flexible but expensive and slow
- None: Fastest but lowest quality

> **Note**: Current benchmark uses synthetic test data. For production evaluation, you need labeled datasets from real documents with manually verified relevance judgments.

## 🐳 Docker

```bash
cp backend/.env.example backend/.env
docker-compose -f deploy/docker-compose.yml up --build
```

## 🤝 Contributing

PRs welcome. See [deploy/README.md](deploy/README.md) for deployment guide.

## 📄 License

MIT

## 🙏 Acknowledgments

- [Groq](https://groq.com/) - Fast free LLM + embedding inference
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [FastAPI](https://fastapi.tiangolo.com/) - Python web framework
- [Next.js](https://nextjs.org/) - React framework

---

**Live at https://ai-document-qa-rag.vercel.app**
