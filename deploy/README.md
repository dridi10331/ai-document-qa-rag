# Deployment Guide

## 🚀 Quick Deploy (Recommended)

### Option 1: Vercel (Frontend) + Render (Backend)

#### Backend on Render
1. Go to https://render.com and create a new **Web Service**
2. Connect your GitHub repo
3. Set build command: `pip install -r backend/requirements.txt`
4. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Set root directory: `backend`
6. Add environment variables:
   ```
   GROQ_API_KEY=your_groq_api_key
   GROQ_MODEL=llama-3.1-8b-instant
   LLM_BACKEND=groq
   EMBEDDINGS_BACKEND=hf
   CORS_ORIGINS_STR=https://your-frontend.vercel.app
   ```

#### Frontend on Vercel
1. Go to https://vercel.com and import your GitHub repo
2. Set root directory: `frontend`
3. Add environment variable:
   ```
   NEXT_PUBLIC_API_BASE=https://your-backend.onrender.com
   ```

---

## 🐳 Docker Compose (Local/VPS)

```bash
# Copy and configure env
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
# Create secret for API key
kubectl create secret generic rag-secrets \
  --from-literal=groq-api-key=your_groq_api_key

# Apply manifests
kubectl apply -f deploy/k8s/

# Check status
kubectl get pods
kubectl get services
```

---

## ⚠️ Security Notes

- **Never commit `.env` files** - they are in `.gitignore`
- **Rotate API keys** if accidentally exposed
- Use **environment variables** or **secrets managers** in production
- Set `CORS_ORIGINS_STR` to your exact frontend domain in production
