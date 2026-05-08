"""
Cross-Encoder Reranker Service

Implements deterministic reranking using cross-encoder models.

Cross-encoders vs Bi-encoders:
- Bi-encoders (FAISS): encode query and document separately, fast but less accurate
- Cross-encoders: encode query+document together, slower but more accurate

This module provides:
1. Cross-encoder reranking (sentence-transformers)
2. LLM-based reranking (Groq) as fallback
3. No reranking (baseline)
4. Performance comparison framework

Model choice: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on MS MARCO passage ranking
- 6 layers, ~80MB, fast inference
- Good balance of speed and accuracy
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from app.core.config import Settings
from app.schemas.search import RetrievalChunk

logger = logging.getLogger(__name__)


class RerankerType(str, Enum):
    """Available reranker implementations."""
    NONE = "none"
    CROSS_ENCODER = "cross_encoder"
    LLM = "llm"


@dataclass
class RerankResult:
    """Result of reranking operation."""
    chunks: list[RetrievalChunk]
    reranker_used: RerankerType
    latency_ms: float
    scores_changed: bool = False


# Global cross-encoder model instance
_cross_encoder_model = None


def get_cross_encoder_model():
    """
    Lazy-load cross-encoder model.
    
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Size: ~80MB
    - Speed: ~50ms for 10 pairs on CPU
    - Trained on MS MARCO passage ranking dataset
    """
    global _cross_encoder_model
    
    if _cross_encoder_model is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2")
            _cross_encoder_model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
            logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            _cross_encoder_model = None
    
    return _cross_encoder_model


def rerank_with_cross_encoder(
    query: str,
    chunks: list[RetrievalChunk],
    top_k: int,
) -> RerankResult:
    """
    Rerank chunks using cross-encoder model.
    
    Cross-encoder scores query-document pairs directly,
    providing more accurate relevance scores than bi-encoders.
    
    Args:
        query: Search query
        chunks: Candidate chunks from retrieval
        top_k: Number of chunks to return
    
    Returns:
        RerankResult with reranked chunks and metadata
    """
    start_time = time.time()
    
    if not chunks:
        return RerankResult(
            chunks=[],
            reranker_used=RerankerType.CROSS_ENCODER,
            latency_ms=0.0,
        )
    
    model = get_cross_encoder_model()
    if model is None:
        logger.warning("Cross-encoder model not available, returning original order")
        return RerankResult(
            chunks=chunks[:top_k],
            reranker_used=RerankerType.NONE,
            latency_ms=(time.time() - start_time) * 1000,
        )
    
    try:
        # Prepare query-document pairs
        # Limit to top candidates to avoid excessive computation
        candidates = chunks[:min(len(chunks), top_k * 2)]
        pairs = [[query, chunk.text] for chunk in candidates]
        
        # Score all pairs
        scores = model.predict(pairs)
        
        # Sort by cross-encoder score
        scored_chunks = list(zip(scores, candidates))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Update chunk scores with cross-encoder scores
        reranked = []
        for score, chunk in scored_chunks[:top_k]:
            # Create new chunk with updated score
            reranked.append(
                RetrievalChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=float(score),  # Cross-encoder score
                    page_number=chunk.page_number,
                    document_name=chunk.document_name,
                )
            )
        
        # Add remaining chunks not scored
        scored_ids = {c.chunk_id for c in reranked}
        reranked.extend([c for c in chunks if c.chunk_id not in scored_ids])
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RerankResult(
            chunks=reranked[:top_k],
            reranker_used=RerankerType.CROSS_ENCODER,
            latency_ms=latency_ms,
            scores_changed=True,
        )
    
    except Exception as e:
        logger.error(f"Cross-encoder reranking failed: {e}")
        return RerankResult(
            chunks=chunks[:top_k],
            reranker_used=RerankerType.NONE,
            latency_ms=(time.time() - start_time) * 1000,
        )


def rerank_with_llm(
    query: str,
    chunks: list[RetrievalChunk],
    settings: Settings,
    top_k: int,
) -> RerankResult:
    """
    Rerank chunks using Groq LLM.
    
    This is the original LLM-based reranking approach.
    Tradeoffs vs cross-encoder:
    - Slower (~1-2s vs ~50ms)
    - More expensive (uses API tokens)
    - Less deterministic (temperature=0 but still some variance)
    - More flexible (can use reasoning, handle edge cases)
    
    Args:
        query: Search query
        chunks: Candidate chunks from retrieval
        settings: Application settings with Groq API key
        top_k: Number of chunks to return
    
    Returns:
        RerankResult with reranked chunks and metadata
    """
    start_time = time.time()
    
    if not settings.groq_api_key or not chunks:
        return RerankResult(
            chunks=chunks[:top_k],
            reranker_used=RerankerType.NONE,
            latency_ms=(time.time() - start_time) * 1000,
        )
    
    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)
        
        # Limit candidates to avoid token limits
        candidates = chunks[:min(len(chunks), 10)]
        chunks_text = "\n\n".join([
            f"[{i}] {chunk.text[:300]}"
            for i, chunk in enumerate(candidates)
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
            import json
            scores = [float(s) for s in json.loads(content[start:end])]
            
            # Sort by LLM score
            scored = list(zip(scores, candidates[:len(scores)]))
            scored.sort(key=lambda x: x[0], reverse=True)
            
            reranked = []
            for score, chunk in scored:
                reranked.append(
                    RetrievalChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        score=score / 10.0,  # Normalize to 0-1
                        page_number=chunk.page_number,
                        document_name=chunk.document_name,
                    )
                )
            
            # Add remaining chunks
            scored_ids = {c.chunk_id for c in reranked}
            reranked.extend([c for c in chunks if c.chunk_id not in scored_ids])
            
            latency_ms = (time.time() - start_time) * 1000
            
            return RerankResult(
                chunks=reranked[:top_k],
                reranker_used=RerankerType.LLM,
                latency_ms=latency_ms,
                scores_changed=True,
            )
    
    except Exception as e:
        logger.error(f"LLM reranking failed: {e}")
    
    return RerankResult(
        chunks=chunks[:top_k],
        reranker_used=RerankerType.NONE,
        latency_ms=(time.time() - start_time) * 1000,
    )


def rerank_chunks(
    query: str,
    chunks: list[RetrievalChunk],
    settings: Settings,
    top_k: int,
    reranker: Literal["none", "cross_encoder", "llm", "auto"] = "auto",
) -> RerankResult:
    """
    Rerank chunks using specified strategy.
    
    Strategies:
    - none: No reranking, return original order
    - cross_encoder: Use cross-encoder model (fast, deterministic)
    - llm: Use Groq LLM (slow, flexible)
    - auto: Try cross-encoder, fallback to LLM, then none
    
    Args:
        query: Search query
        chunks: Candidate chunks from retrieval
        settings: Application settings
        top_k: Number of chunks to return
        reranker: Reranking strategy
    
    Returns:
        RerankResult with reranked chunks and metadata
    """
    if not chunks or len(chunks) <= 1:
        return RerankResult(
            chunks=chunks[:top_k],
            reranker_used=RerankerType.NONE,
            latency_ms=0.0,
        )
    
    if reranker == "none":
        return RerankResult(
            chunks=chunks[:top_k],
            reranker_used=RerankerType.NONE,
            latency_ms=0.0,
        )
    
    if reranker == "cross_encoder":
        return rerank_with_cross_encoder(query, chunks, top_k)
    
    if reranker == "llm":
        return rerank_with_llm(query, chunks, settings, top_k)
    
    # Auto mode: try cross-encoder first, fallback to LLM
    if reranker == "auto":
        result = rerank_with_cross_encoder(query, chunks, top_k)
        if result.reranker_used == RerankerType.CROSS_ENCODER:
            return result
        
        # Fallback to LLM
        logger.info("Cross-encoder unavailable, falling back to LLM reranking")
        return rerank_with_llm(query, chunks, settings, top_k)
    
    # Unknown reranker type
    logger.warning(f"Unknown reranker type: {reranker}, using none")
    return RerankResult(
        chunks=chunks[:top_k],
        reranker_used=RerankerType.NONE,
        latency_ms=0.0,
    )
