"""Tests for reranker service."""

from app.core.config import Settings
from app.schemas.search import RetrievalChunk
from app.services.reranker import rerank_chunks, RerankerType


def test_rerank_with_none():
    """Test that 'none' reranker returns original order."""
    chunks = [
        RetrievalChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            chunk_index=0,
            text="Python is a programming language",
            score=0.8,
        ),
        RetrievalChunk(
            chunk_id="chunk_2",
            document_id="doc_1",
            chunk_index=1,
            text="JavaScript is used for web development",
            score=0.6,
        ),
    ]
    
    settings = Settings()
    result = rerank_chunks(
        query="What is Python?",
        chunks=chunks,
        settings=settings,
        top_k=2,
        reranker="none",
    )
    
    assert result.reranker_used == RerankerType.NONE
    assert len(result.chunks) == 2
    assert result.chunks[0].chunk_id == "chunk_1"
    assert result.chunks[1].chunk_id == "chunk_2"
    assert result.latency_ms == 0.0
    assert not result.scores_changed


def test_rerank_with_cross_encoder():
    """Test cross-encoder reranking (may skip if model not available)."""
    chunks = [
        RetrievalChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            chunk_index=0,
            text="The weather is sunny today with clear skies",
            score=0.9,  # High initial score but irrelevant
        ),
        RetrievalChunk(
            chunk_id="chunk_2",
            document_id="doc_1",
            chunk_index=1,
            text="Python is a high-level programming language known for readability",
            score=0.5,  # Lower initial score but highly relevant
        ),
    ]
    
    settings = Settings()
    result = rerank_chunks(
        query="What is Python programming language?",
        chunks=chunks,
        settings=settings,
        top_k=2,
        reranker="cross_encoder",
    )
    
    # If cross-encoder is available, it should reorder chunks
    # If not available, it falls back to none
    assert result.reranker_used in [RerankerType.CROSS_ENCODER, RerankerType.NONE]
    assert len(result.chunks) == 2
    
    if result.reranker_used == RerankerType.CROSS_ENCODER:
        # Cross-encoder should rank Python chunk higher
        assert result.chunks[0].chunk_id == "chunk_2"
        assert result.scores_changed
        assert result.latency_ms > 0


def test_rerank_empty_chunks():
    """Test reranking with empty chunk list."""
    settings = Settings()
    result = rerank_chunks(
        query="test query",
        chunks=[],
        settings=settings,
        top_k=5,
        reranker="cross_encoder",
    )
    
    assert len(result.chunks) == 0
    # Empty chunks return early with NONE type
    assert result.reranker_used in [RerankerType.CROSS_ENCODER, RerankerType.NONE]
    assert result.latency_ms >= 0.0


def test_rerank_single_chunk():
    """Test reranking with single chunk (should skip reranking)."""
    chunks = [
        RetrievalChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            chunk_index=0,
            text="Single chunk",
            score=0.8,
        ),
    ]
    
    settings = Settings()
    result = rerank_chunks(
        query="test",
        chunks=chunks,
        settings=settings,
        top_k=5,
        reranker="cross_encoder",
    )
    
    assert len(result.chunks) == 1
    assert result.reranker_used == RerankerType.NONE
