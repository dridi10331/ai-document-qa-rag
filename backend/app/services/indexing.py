from __future__ import annotations

import logging

from sqlmodel import Session

from app.core.config import Settings
from app.db.crud import list_all_chunks, list_chunks_by_ids
from app.db.models import Chunk
from app.services.bm25_store import get_bm25_store
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def index_chunks(chunks: list[Chunk], settings: Settings) -> None:
    if not chunks:
        return
    texts = [chunk.text for chunk in chunks]
    embedder = get_embedding_service(settings)
    embedding_result = embedder.embed_texts(texts)

    vector_store = get_vector_store(settings, embedding_result.dimension)
    vector_store.add(embedding_result.vectors, [chunk.id for chunk in chunks])

    bm25_store = get_bm25_store(settings)
    bm25_store.add(texts, [chunk.id for chunk in chunks])

    logger.info("Indexed %s chunks", len(chunks))


def index_chunks_by_ids(chunk_ids: list[str], session: Session, settings: Settings) -> None:
    if not chunk_ids:
        return
    chunks = list_chunks_by_ids(session, chunk_ids)
    index_chunks(chunks, settings)


def rebuild_indexes(session: Session, settings: Settings) -> None:
    chunks = list_all_chunks(session)
    embedder = get_embedding_service(settings)
    vector_store = get_vector_store(settings, embedder.dimension)
    bm25_store = get_bm25_store(settings)
    if not chunks:
        vector_store.load_or_create(embedder.dimension)
        if vector_store.index is not None:
            vector_store.index.reset()
            vector_store.chunk_ids = []
            vector_store._persist()
        bm25_store.rebuild([], [])
        return

    texts = [chunk.text for chunk in chunks]
    embedding_result = embedder.embed_texts(texts)

    vector_store.load_or_create(embedding_result.dimension)
    vector_store.chunk_ids = []
    vector_store.index.reset()
    vector_store.add(embedding_result.vectors, [chunk.id for chunk in chunks])
    bm25_store.rebuild(texts, [chunk.id for chunk in chunks])

    logger.info("Rebuilt indexes with %s chunks", len(chunks))
