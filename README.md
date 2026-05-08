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

### 🎯 Hybrid Retrieval
- **FAISS Vector Search**: Semantic similarity using Groq embeddings (`nomic-embed-text-v1.5`)
- **BM25 Keyword Search**: Lexical matching with rank-bm25
- **Score Fusion**: Weighted combination (65% vector + 35% BM25)
- **Query Expansion**: Groq-powered alternative phrasings for better recall
- **LLM Reranking**: Groq-based relevance scoring to improve chunk ordering

### 💬 Real-time Q&A
- **Streaming Responses**: Server-Sent Events (SSE) for token-by-token output
- **WebSocket Status**: Live document processing updates
- **Chat History**: Persistent conversation context per session
- **Citations**: Source attribution with page numbers and relevance scores

### 📊 Analytics
- **Query Logs**: Latency, token usage, model used
- **Document Usage**: Most referenced documents
- **Cost Tracking**: Token-level cost estimation

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
- Async ingestion queue (Celery/Redis)
- Persistent vector store (Qdrant/Weaviate/pgvector)
- Retrieval evaluation pipeline (RAGAS/DeepEval)
- Observability (Langfuse/OpenTelemetry)
- Reranking (cross-encoder/bge-reranker)
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
| `POST` | `/eval/retrieval` | Retrieval evaluation (scores, rerank, expansion) |
| `WS` | `/ws/documents/{id}` | Processing status |

## 🧪 Tests

```bash
cd backend
pytest
pytest --cov=app tests/
```

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
