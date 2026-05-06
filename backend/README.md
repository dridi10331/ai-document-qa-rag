# Backend

FastAPI service for document ingestion, chunking, retrieval, and analytics.

## Endpoints
- POST /documents/upload
- GET /documents
- GET /documents/{doc_id}
- GET /documents/{doc_id}/chunks
- GET /documents/{doc_id}/status
- DELETE /documents/{doc_id}
- GET /health
- POST /query
- GET /query/stream
- GET /analytics/summary
- POST /sessions
- GET /sessions
- GET /sessions/{session_id}/messages
- WS /ws/documents/{doc_id}

## Environment
Copy the env template:
- `copy .env.example .env`

**Ollama (Free & Local)**
The system uses Ollama for LLM. No API keys needed!

## Ollama Setup
1. Install Ollama from https://ollama.ai
2. Pull a model: `ollama pull llama3`
3. Start the backend - it works out of the box!

## OCR Notes
OCR requires system installs:
- Tesseract OCR
- Poppler (for pdf2image)

If not installed, OCR is skipped and empty pages remain empty.

## Tests
- `pytest`
