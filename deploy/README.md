# Deployment Scaffolds

This folder holds Docker and Kubernetes manifests for local or cluster deployment.

## Docker Compose
- `docker-compose.yml`
- `backend.Dockerfile`
- `frontend.Dockerfile`

From this folder:
- `docker compose up --build`

## Kubernetes
Manifests are under `k8s/` for backend and frontend deployments/services.
Edit image names and add Ollama configuration before applying.
