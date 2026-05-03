from app.core.config import Settings
from app.services.bm25_store import get_bm25_store
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store


def get_index_sizes(settings: Settings) -> tuple[int, int]:
    embedder = get_embedding_service(settings)
    vector_store = get_vector_store(settings, embedder.dimension)
    bm25_store = get_bm25_store(settings)
    return vector_store.size(), bm25_store.size()
