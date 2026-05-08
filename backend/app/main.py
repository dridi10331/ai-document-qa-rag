from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.session import init_db
from app.services.bm25_store import get_bm25_store
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store

settings = get_settings()

setup_logging()

app = FastAPI(title=settings.project_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.on_event("startup")
async def on_startup() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    # Only preload embeddings if not using mock backend
    if settings.embeddings_backend == "hf":
        embedder = get_embedding_service(settings)
        get_vector_store(settings, embedder.dimension)
        get_bm25_store(settings)
    elif settings.embeddings_backend == "groq":
        # Groq embeddings: nomic-embed-text-v1.5 = 1024 dims
        get_vector_store(settings, 1024)
        get_bm25_store(settings)
    else:
        # mock: 384 dims
        get_vector_store(settings, 384)
        get_bm25_store(settings)


@app.get("/")
def root() -> dict:
    return {"name": settings.project_name, "status": "ok"}
