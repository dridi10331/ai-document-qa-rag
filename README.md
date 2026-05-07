# 🤖 AI-Powered Document Q&A System

> A production-grade Retrieval-Augmented Generation (RAG) system for intelligent multi-document question answering with real-time streaming, hybrid search, and comprehensive analytics.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat&logo=next.js)](https://nextjs.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-blue?style=flat)](https://ollama.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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
- **Cost Estimation**: Token usage and cost tracking
- **Document Insights**: Most-used documents and citation analysis

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Next.js   │─────▶│   FastAPI    │─────▶│   Ollama    │
│  Frontend   │      │   Backend    │      │  (Local LLM)│
└─────────────┘      └──────────────┘      └─────────────┘
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
| **LLM** | Ollama | Local, free language models |
| **Vector DB** | FAISS | Fast similarity search |
| **Embeddings** | SentenceTransformers | HuggingFace embeddings |
| **Search** | BM25 | Keyword-based retrieval |
| **Database** | SQLite | Metadata and analytics |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.ai/) installed and running

### 1️⃣ Install Ollama

```bash
# Download and install from https://ollama.ai
# Then pull a model:
ollama pull llama3
```

### 2️⃣ Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env

# Run the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 3️⃣ Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Configure environment
copy .env.local.example .env.local

# Run the development server
npm run dev
```

The UI will be available at `http://localhost:3000`

## 📁 Project Structure

```
.
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── core/           # Configuration
│   │   ├── db/             # Database models
│   │   ├── schemas/        # Pydantic schemas
│   │   ├── services/       # Business logic
│   │   └── utils/          # Utilities
│   ├── data/               # Document storage
│   ├── tests/              # Unit tests
│   └── requirements.txt
│
├── frontend/               # Next.js frontend
│   ├── app/               # App router pages
│   ├── components/        # React components
│   ├── lib/               # Utilities
│   └── package.json
│
└── deploy/                # Deployment configs
    ├── docker-compose.yml
    └── k8s/              # Kubernetes manifests
```

## 🔧 Configuration

### Backend Environment Variables

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
LLM_BACKEND=ollama

# Embeddings
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDINGS_BACKEND=hf

# Search Configuration
ENABLE_HYBRID_SEARCH=true
ENABLE_QUERY_EXPANSION=true
BM25_WEIGHT=0.35
VECTOR_WEIGHT=0.65

# Chunking
CHUNK_MIN_WORDS=120
CHUNK_MAX_WORDS=320
CHUNK_OVERLAP_WORDS=60
```

### Frontend Environment Variables

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 📚 API Documentation

Once the backend is running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload documents |
| `GET` | `/documents` | List all documents |
| `GET` | `/documents/{id}` | Get document details |
| `DELETE` | `/documents/{id}` | Delete document |
| `POST` | `/query` | Ask a question |
| `GET` | `/query/stream` | Stream answer tokens |
| `GET` | `/analytics/summary` | Get analytics |
| `WS` | `/ws/documents/{id}` | Document status updates |

## 🧪 Testing

```bash
# Run backend tests
cd backend
pytest

# Run with coverage
pytest --cov=app tests/
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f deploy/docker-compose.yml up --build

# Access the application
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

## ☸️ Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f deploy/k8s/

# Check deployment status
kubectl get pods
kubectl get services
```

## 🎨 Available Models

The system works with any Ollama model. Popular choices:

- **llama3** (Recommended) - Fast and accurate
- **mistral** - Excellent for reasoning
- **codellama** - Great for technical documents
- **phi3** - Lightweight and fast

```bash
# Pull additional models
ollama pull mistral
ollama pull codellama
```

## 🔍 Advanced Features

### Query Expansion
Automatically generates alternative phrasings of queries for better retrieval:
```python
ENABLE_QUERY_EXPANSION=true
QUERY_EXPANSION_MAX=3
```

### Hybrid Search
Combines semantic (vector) and keyword (BM25) search:
```python
ENABLE_HYBRID_SEARCH=true
BM25_WEIGHT=0.35
VECTOR_WEIGHT=0.65
```

### OCR Support
Enable OCR for scanned documents:
```bash
# Install system dependencies
# Tesseract: https://github.com/tesseract-ocr/tesseract
# Poppler: https://poppler.freedesktop.org/

# Enable in .env
ENABLE_OCR=true
```

## 📊 Performance

- **Ingestion**: ~1-2 seconds per page
- **Query Latency**: 2-5 seconds (depends on model)
- **Concurrent Users**: 50+ (with proper scaling)
- **Document Limit**: No hard limit (storage dependent)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM runtime
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [SentenceTransformers](https://www.sbert.net/) - Embeddings
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Next.js](https://nextjs.org/) - React framework

## 📞 Support

For issues and questions:
- Open an [Issue](https://github.com/yourusername/rag-system/issues)
- Check the [Documentation](./docs)

---

**Built with ❤️ using Ollama for free, local AI**
