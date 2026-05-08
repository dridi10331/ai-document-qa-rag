# Deployment Guide

## 🌐 Live Deployment

| Service | URL |
|---------|-----|
| **Frontend** | https://ai-document-qa-rag.vercel.app |
| **Backend** | https://rag-backend-u868.onrender.com |
| **API Docs** | https://rag-backend-u868.onrender.com/docs |

---

## 🚀 Deploy Your Own (Vercel + Render)

### Step 1: Backend on Render

1. Go to https://render.com → **New Web Service**
2. Connect `dridi10331/ai-document-qa-rag`
3. Configure:
   - **Root Directory**: `backend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python3.11 -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables:

| Key | Value |
|-----|-------|
| `GROQ_API_KEY` | your key from https://console.groq.com |
| `GROQ_MODEL` | `llama-3.1-8b-instant` |
| `LLM_BACKEND` | `groq` |
| `EMBEDDINGS_BACKEND` | `mock` |
| `CORS_ORIGINS_STR` | `https://your-app.vercel.app` |

5. Click **Deploy**

### Step 2: Frontend on Vercel

1. Go to https://vercel.com → **New Project**
2. Import `dridi10331/ai-document-qa-rag`
3. Configure:
   - **Root Directory**: `frontend`
   - **Framework**: Next.js
4. Add environment variable:

| Key | Value |
|-----|-------|
| `NEXT_PUBLIC_API_BASE` | `https://your-backend.onrender.com` |

5. Click **Deploy**

---

## 🐳 Docker Compose (Local/VPS)

```bash
# Configure environment
cp backend/.env.example backend/.env
# Edit backend/.env with your GROQ_API_KEY

# Build and run
docker-compose -f deploy/docker-compose.yml up --build

# Access
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

---

## ☸️ Kubernetes

```bash
# Create secret
kubectl create secret generic rag-secrets \
  --from-literal=groq-api-key=your_groq_api_key

# Deploy
kubectl apply -f deploy/k8s/

# Check status
kubectl get pods
kubectl get services
```

---

## ⚠️ Security Notes

- **Never commit `.env` files** - they are in `.gitignore`
- **Rotate API keys** if accidentally exposed
- Set `CORS_ORIGINS_STR` to your exact frontend domain in production
- Use Render/Vercel environment variables for secrets, never hardcode them
