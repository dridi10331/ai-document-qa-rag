from __future__ import annotations

import logging
from dataclasses import dataclass, field

import requests
from sqlmodel import Session

from app.core.config import Settings
from app.db.crud import list_chunks_by_ids, list_documents_by_ids
from app.schemas.search import RetrievalChunk
from app.services.bm25_store import get_bm25_store
from app.services.embeddings import get_embedding_service
from app.services.query_expansion import expand_query
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunks: list[RetrievalChunk]
    expanded_query: str | None = None
    reranked: bool = False


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values()) or 1.0
    return {key: value / max_score for key, value in scores.items()}


def _rerank_with_groq(
    query: str,
    chunks: list[RetrievalChunk],
    settings: Settings,
    top_k: int,
) -> tuple[list[RetrievalChunk], bool]:
    """
    Rerank chunks using Groq LLM as a cross-encoder substitute.

    Design notes:
    - True cross-encoders (bge-reranker) require local model loading
    - On free-tier cloud (512MB RAM), we use LLM-based reranking instead
    - LLM scores each chunk's relevance to the query (0-10)
    - This improves chunk ordering and reduces irrelevant context
    - Tradeoff: adds ~1-2s latency, uses extra tokens
    """
    if not settings.groq_api_key or len(chunks) <= 1:
        return chunks[:top_k], False

    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)

        # Build scoring prompt
        chunks_text = "\n\n".join([
            f"[{i}] {chunk.text[:300]}"
            for i, chunk in enumerate(chunks[:10])  # limit to top 10 candidates
        ])

        prompt = (
            f"Query: {query}\n\n"
            f"Rate each chunk's relevance to the query (0-10). "
            f"Return ONLY a JSON array of numbers in order. Example: [8, 3, 9, 1]\n\n"
            f"Chunks:\n{chunks_text}"
        )

        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )

        content = response.choices[0].message.content or "[]"
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            scores = [float(s) for s in __import__("json").loads(content[start:end])]
            # Re-sort chunks by rerank score
            scored = list(zip(scores, chunks[:len(scores)]))
            scored.sort(key=lambda x: x[0], reverse=True)
            reranked = [chunk for _, chunk in scored]
            # Append any chunks not scored
            scored_ids = {c.chunk_id for c in reranked}
            reranked += [c for c in chunks if c.chunk_id not in scored_ids]
            return reranked[:top_k], True

    except Exception as e:
        logger.warning("Reranking failed, using original order: %s", e)

    return chunks[:top_k], False


def retrieve_chunks(
    session: Session,
    query: str,
    settings: Settings,
    top_k: int,
    document_ids: list[str] | None,
    use_hybrid: bool | None,
    enable_query_expansion: bool | None,
) -> RetrievalResult:
    """
    Hybrid retrieval pipeline:
    1. Query expansion (optional) - generate alternative phrasings
    2. Vector search (FAISS) - semantic similarity
    3. BM25 search (optional) - keyword matching
    4. Score fusion - weighted combination
    5. Reranking (optional) - LLM-based relevance scoring

    Weighting rationale:
    - 65% vector: captures semantic meaning
    - 35% BM25: captures exact keyword matches
    - Empirically tuned; adjustable via BM25_WEIGHT/VECTOR_WEIGHT env vars
    """
    embedder = get_embedding_service(settings)
    vector_store = get_vector_store(settings, embedder.dimension)
    bm25_store = get_bm25_store(settings)

    use_hybrid = settings.enable_hybrid_search if use_hybrid is None else use_hybrid
    enable_query_expansion = (
        settings.enable_query_expansion
        if enable_query_expansion is None
        else enable_query_expansion
    )

    # Step 1: Query expansion
    expanded_queries: list[str] = []
    if enable_query_expansion:
        expanded_queries = expand_query(query, settings)

    document_ids = [doc_id for doc_id in (document_ids or []) if doc_id]
    combined_query = " ".join([query] + expanded_queries)

    # Step 2: Vector search
    vectors = embedder.embed_texts([combined_query])
    vector_results = vector_store.search(vectors.vectors[0], top_k * 4)
    vector_scores = {result.chunk_id: result.score for result in vector_results}

    # Step 3: BM25 keyword search
    bm25_scores: dict[str, float] = {}
    if use_hybrid:
        bm25_results = bm25_store.search(combined_query, top_k * 6)
        bm25_scores = {result.chunk_id: result.score for result in bm25_results}

    # Step 4: Score fusion (Reciprocal Rank Fusion style normalization)
    vector_scores = _normalize_scores(vector_scores)
    bm25_scores = _normalize_scores(bm25_scores)

    merged_scores: dict[str, float] = {}
    for chunk_id, score in vector_scores.items():
        merged_scores[chunk_id] = merged_scores.get(chunk_id, 0.0) + score * settings.vector_weight
    for chunk_id, score in bm25_scores.items():
        merged_scores[chunk_id] = merged_scores.get(chunk_id, 0.0) + score * settings.bm25_weight

    sorted_ids = sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)
    candidate_ids = [chunk_id for chunk_id, _ in sorted_ids][:max(top_k * 4, top_k)]
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

    # Step 5: Reranking
    reranked_chunks, was_reranked = _rerank_with_groq(query, ranked_chunks, settings, top_k)

    expanded_query_text = " | ".join(expanded_queries) if expanded_queries else None
    return RetrievalResult(
        chunks=reranked_chunks,
        expanded_query=expanded_query_text,
        reranked=was_reranked,
    )
