# 🤖 AI-Powered Document Q&A System

> A production-grade Retrieval-Augmented Generation (RAG) system for intelligent multi-document question answering with real-time streaming, hybrid search, and comprehensive analytics.

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

## ✨ Features

### 🔍 Intelligent Document Processing
- **Multi-format Support**: PDF, DOCX, TXT, Markdown with optional OCR
- **Smart Chunking**: Context-aware text segmentation with configurable overlap
- **Metadata Extraction**: Automatic page numbers and document structure analysis

### 🎯 Advanced Retrieval
- **Hybrid Search**: Combines FAISS vector similarity with BM25 keyword matching
- **Query Expansion**: Automatic query reformulation for better results
- **Multi-document Reasoning**: Cross-reference information from multiple sources
- **Citation Tracking**: Full source attribution with page numbers

### 💬 Real-time Interaction
- **Streaming Responses**: Server-Sent Events (SSE) for token-by-token output
- **WebSocket Updates**: Live document processing status
- **Chat History**: Persistent conversation context
- **Session Management**: Multi-session support with history

### 📊 Analytics & Monitoring
- **Query Analytics**: Track usage patterns, latency, and performance
- **Token Usage**: Track tokens in/out per query
- **Document Insights**: Most-used documents and citation analysis

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────┐
│   Next.js 14    │─────▶│   FastAPI        │─────▶│  Groq API   │
│  (Vercel)       │      │   (Render)       │      │  (Free LLM) │
└─────────────────┘      └──────────────────┘      └─────────────┘
                                  │
                                  ├─────▶ FAISS (Vector DB)
                                  ├─────▶ BM25 (Keyword Search)
                                  └─────▶ SQLite (Metadata)
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI | High-performance async API |
| **Frontend** | Next.js 14 | Modern React framework with SSR |
| **LLM** | Groq (llama-3.1-8b-instant) | Free, fast language model |
| **Vector DB** | FAISS | Fast similarity search |
| **Search** | BM25 | Keyword-based retrieval |
| **Database** | SQLite | Metadata and analytics |
| **Hosting** | Vercel + Render | Free production deployment |

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.11+
- Node.js 18+
- Free [Groq API key](https://console.groq.com/keys)

### 1️⃣ Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Add your GROQ_API_KEY to .env

# Run the API server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2️⃣ Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
copy .env.local.example .env.local
# Set NEXT_PUBLIC_API_BASE=http://localhost:8000

# Run the development server
npm run dev
```

Open http://localhost:3000

## 📁 Project Structure

```
.
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── core/           # Configuration
│   │   ├── db/             # Database models & CRUD
│   │   ├── schemas/        # Pydantic schemas
│   │   ├── services/       # Business logic
│   │   │   ├── llm.py      # Groq/Ollama integration
│   │   │   ├── retrieval.py # Hybrid search
│   │   │   ├── embeddings.py # Vector embeddings
│   │   │   ├── chunking.py  # Document chunking
│   │   │   └── ...
│   │   └── utils/          # Utilities
│   ├── tests/              # Unit tests
│   ├── .python-version     # Python 3.11.9
│   └── requirements.txt
│
├── frontend/               # Next.js frontend
│   ├── app/               # App router pages
│   ├── components/        # React components
│   ├── lib/api.ts         # API client with SSE
│   └── package.json
│
└── deploy/                # Deployment configs
    ├── docker-compose.yml
    ├── backend.Dockerfile
    ├── frontend.Dockerfile
    └── k8s/              # Kubernetes manifests
```

## 🔧 Configuration

### Backend Environment Variables

```env
# LLM Backend
LLM_BACKEND=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant

# Embeddings (use mock for fast startup)
EMBEDDINGS_BACKEND=mock

# CORS
CORS_ORIGINS_STR=http://localhost:3000

# Search
ENABLE_HYBRID_SEARCH=true
ENABLE_QUERY_EXPANSION=true
BM25_WEIGHT=0.35
VECTOR_WEIGHT=0.65
```

### Frontend Environment Variables

```env
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

## 📚 API Documentation

- **Interactive Docs**: https://rag-backend-u868.onrender.com/docs
- **ReDoc**: https://rag-backend-u868.onrender.com/redoc

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload documents |
| `GET` | `/documents` | List all documents |
| `DELETE` | `/documents/{id}` | Delete document |
| `POST` | `/query` | Ask a question |
| `GET` | `/query/stream` | Stream answer (SSE) |
| `GET` | `/analytics/summary` | Get analytics |
| `WS` | `/ws/documents/{id}` | Live status updates |

## 🧪 Testing

```bash
cd backend
pytest
pytest --cov=app tests/
```

## 🐳 Docker Deployment

```bash
cp backend/.env.example backend/.env
# Edit backend/.env with your GROQ_API_KEY

docker-compose -f deploy/docker-compose.yml up --build
```

## ☸️ Kubernetes Deployment

```bash
kubectl create secret generic rag-secrets \
  --from-literal=groq-api-key=your_groq_api_key

kubectl apply -f deploy/k8s/
kubectl get pods
```

## 📊 Performance

- **Upload & Chunking**: ~2-3 seconds per document
- **Query Latency**: ~1 second (Groq is very fast)
- **Cost**: $0.00 (Groq free tier)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Groq](https://groq.com/) - Ultra-fast free LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Next.js](https://nextjs.org/) - React framework
- [Vercel](https://vercel.com/) - Frontend hosting
- [Render](https://render.com/) - Backend hosting

---

**Built with ❤️ | Live at https://ai-document-qa-rag.vercel.app**
