"""Schemas for retrieval dashboard."""

from pydantic import BaseModel


class ChunkRankingRow(BaseModel):
    """Single row in chunk ranking table."""
    chunk_id: str
    text_preview: str
    vector_score: float | None
    bm25_score: float | None
    fused_score: float | None
    reranked_score: float | None
    final_rank: int


class RetrievalDashboardResponse(BaseModel):
    """Retrieval dashboard visualization data."""
    query: str
    expanded_query: str | None
    chunk_rankings: list[ChunkRankingRow]
    score_distribution: dict[str, list[float]]
    latency_waterfall: dict[str, float]
    reranking_diff: list[dict[str, int]]  # [{chunk_id, before_rank, after_rank}]
