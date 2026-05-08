# Backend - FastAPI RAG Service

FastAPI service for document ingestion, chunking, hybrid retrieval, and AI-powered Q&A using Groq.

## 🌐 Live API

- **Base URL**: https://rag-backend-u868.onrender.com
- **Docs**: https://rag-backend-u868.onrender.com/docs

## 📡 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/documents/upload` | Upload documents |
| `GET` | `/documents` | List all documents |
| `GET` | `/documents/{id}` | Get document details |
| `GET` | `/documents/{id}/chunks` | Get document chunks |
| `GET` | `/documents/{id}/status` | Get processing status |
| `DELETE` | `/documents/{id}` | Delete document |
| `POST` | `/query` | Ask a question |
| `GET` | `/query/stream` | Stream answer (SSE) |
| `GET` | `/analytics/summary` | Analytics summary |
| `POST` | `/sessions` | Create chat session |
| `GET` | `/sessions` | List sessions |
| `GET` | `/sessions/{id}/messages` | Get chat history |
| `WS` | `/ws/documents/{id}` | Live status updates |

## ⚙️ Environment Setup

```bash
copy .env.example .env
```

Required variables:

```env
# LLM (Groq - free at https://console.groq.com)
LLM_BACKEND=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Embeddings (groq = real semantic via nomic-embed-text-v1.5, mock = fast hash-based)
EMBEDDINGS_BACKEND=groq

# CORS
CORS_ORIGINS_STR=http://localhost:3000
```

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 🧪 Tests

```bash
pytest
pytest --cov=app tests/
```

## 📦 Requirements

Key dependencies:
- `fastapi` - Web framework
- `groq` - LLM inference (free)
- `faiss-cpu` - Vector search
- `rank-bm25` - Keyword search
- `sqlmodel` - Database ORM
- `PyPDF2`, `python-docx` - Document parsing

## 🔍 Architecture

```
Upload → Parse → Chunk → Embed → Index (FAISS + BM25)
Query → Expand → Retrieve → Rank → Generate (Groq) → Stream
```

## 📝 OCR Notes

OCR requires system installs:
- Tesseract OCR
- Poppler (for pdf2image)

Set `ENABLE_OCR=true` in `.env` to enable.
