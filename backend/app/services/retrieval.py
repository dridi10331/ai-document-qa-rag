from __future__ import annotations

from dataclasses import dataclass

from sqlmodel import Session

from app.core.config import Settings
from app.db.crud import list_chunks_by_ids, list_documents_by_ids
from app.schemas.search import RetrievalChunk
from app.services.bm25_store import get_bm25_store
from app.services.embeddings import get_embedding_service
from app.services.query_expansion import expand_query
from app.services.vector_store import get_vector_store


@dataclass
class RetrievalResult:
    chunks: list[RetrievalChunk]
    expanded_query: str | None = None


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values()) or 1.0
    return {key: value / max_score for key, value in scores.items()}


def retrieve_chunks(
    session: Session,
    query: str,
    settings: Settings,
    top_k: int,
    document_ids: list[str] | None,
    use_hybrid: bool | None,
    enable_query_expansion: bool | None,
) -> RetrievalResult:
    embedder = get_embedding_service(settings)
    vector_store = get_vector_store(settings, embedder.dimension)
    bm25_store = get_bm25_store(settings)

    use_hybrid = settings.enable_hybrid_search if use_hybrid is None else use_hybrid
    enable_query_expansion = (
        settings.enable_query_expansion
        if enable_query_expansion is None
        else enable_query_expansion
    )

    expanded_queries: list[str] = []
    if enable_query_expansion:
        expanded_queries = expand_query(query, settings, settings.ollama_base_url)

    document_ids = [doc_id for doc_id in (document_ids or []) if doc_id]
    combined_query = " ".join([query] + expanded_queries)

    vector_query = combined_query
    vectors = embedder.embed_texts([vector_query])
    vector_results = vector_store.search(vectors.vectors[0], top_k * 4)
    vector_scores = {result.chunk_id: result.score for result in vector_results}

    bm25_scores = {}
    if use_hybrid:
        bm25_results = bm25_store.search(combined_query, top_k * 6)
        bm25_scores = {result.chunk_id: result.score for result in bm25_results}

    vector_scores = _normalize_scores(vector_scores)
    bm25_scores = _normalize_scores(bm25_scores)

    merged_scores: dict[str, float] = {}
    for chunk_id, score in vector_scores.items():
        merged_scores[chunk_id] = merged_scores.get(chunk_id, 0.0) + score * settings.vector_weight
    for chunk_id, score in bm25_scores.items():
        merged_scores[chunk_id] = merged_scores.get(chunk_id, 0.0) + score * settings.bm25_weight

    sorted_ids = sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)
    candidate_ids = [chunk_id for chunk_id, _ in sorted_ids][: max(top_k * 4, top_k)]
    chunks = list_chunks_by_ids(session, candidate_ids)

    if document_ids:
        chunks = [chunk for chunk in chunks if chunk.document_id in document_ids]

    doc_ids = list({chunk.document_id for chunk in chunks})
    docs = list_documents_by_ids(session, doc_ids)
    doc_lookup = {doc.id: doc.filename for doc in docs}

    chunk_lookup = {chunk.id: chunk for chunk in chunks}
    ranked_chunks: list[RetrievalChunk] = []
    for chunk_id, score in sorted_ids:
        if chunk_id not in chunk_lookup:
            continue
        chunk = chunk_lookup[chunk_id]
        ranked_chunks.append(
            RetrievalChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=score,
                page_number=chunk.page_number,
                document_name=doc_lookup.get(chunk.document_id),
            )
        )
        if len(ranked_chunks) >= top_k:
            break

    expanded_query_text = " | ".join(expanded_queries) if expanded_queries else None
    return RetrievalResult(chunks=ranked_chunks, expanded_query=expanded_query_text)

# Day 4: citation tracking
