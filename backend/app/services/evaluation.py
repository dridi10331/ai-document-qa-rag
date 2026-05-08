from __future__ import annotations

"""
Retrieval Evaluation Service

Implements basic IR metrics without external dependencies:
- Precision@k: fraction of top-k chunks that are relevant
- Recall@k: fraction of relevant chunks found in top-k
- MRR (Mean Reciprocal Rank): rank of first relevant result
- Average score: mean relevance score of retrieved chunks
- Rerank gain: score improvement from reranking

Note: True evaluation requires labeled ground truth.
This module provides:
1. Self-contained metrics from score distributions
2. LLM-based relevance judgment (Groq)
3. Latency tracking per pipeline stage
"""

import logging
import time
from dataclasses import dataclass, field

from app.core.config import Settings
from app.schemas.search import RetrievalChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    query: str
    chunks_retrieved: int
    avg_score: float
    max_score: float
    min_score: float
    score_std: float
    expanded_query: str | None
    reranked: bool
    latency_ms: float
    # LLM-judged relevance (0-1 per chunk)
    relevance_scores: list[float] = field(default_factory=list)
    precision_at_k: float | None = None
    mrr: float | None = None
    rerank_gain: float | None = None
    notes: list[str] = field(default_factory=list)


def compute_score_metrics(scores: list[float]) -> dict:
    if not scores:
        return {"avg": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
    avg = sum(scores) / len(scores)
    variance = sum((s - avg) ** 2 for s in scores) / len(scores)
    std = variance ** 0.5
    return {
        "avg": round(avg, 4),
        "max": round(max(scores), 4),
        "min": round(min(scores), 4),
        "std": round(std, 4),
    }


def judge_relevance_with_llm(
    query: str,
    chunks: list[RetrievalChunk],
    settings: Settings,
    threshold: float = 5.0,
) -> list[float]:
    """
    Use Groq LLM to judge relevance of each chunk to the query.
    Returns a list of relevance scores (0-10) per chunk.
    Chunks scoring >= threshold are considered relevant.
    """
    if not settings.groq_api_key or not chunks:
        return []

    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)

        chunks_text = "\n\n".join([
            f"[{i}] {chunk.text[:400]}"
            for i, chunk in enumerate(chunks[:8])
        ])

        prompt = (
            f"Query: {query}\n\n"
            f"For each chunk below, rate its relevance to the query from 0 to 10.\n"
            f"0 = completely irrelevant, 10 = perfectly answers the query.\n"
            f"Return ONLY a JSON array of numbers. Example: [8, 2, 9, 0, 5]\n\n"
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
            return scores[:len(chunks)]

    except Exception as e:
        logger.warning("LLM relevance judgment failed: %s", e)

    return []


def compute_precision_at_k(relevance_scores: list[float], k: int, threshold: float = 5.0) -> float:
    """Fraction of top-k chunks judged relevant by LLM."""
    if not relevance_scores:
        return 0.0
    top_k = relevance_scores[:k]
    relevant = sum(1 for s in top_k if s >= threshold)
    return round(relevant / len(top_k), 4)


def compute_mrr(relevance_scores: list[float], threshold: float = 5.0) -> float:
    """Mean Reciprocal Rank - rank of first relevant result."""
    for i, score in enumerate(relevance_scores):
        if score >= threshold:
            return round(1.0 / (i + 1), 4)
    return 0.0


def evaluate_retrieval(
    query: str,
    chunks_before_rerank: list[RetrievalChunk],
    chunks_after_rerank: list[RetrievalChunk],
    expanded_query: str | None,
    reranked: bool,
    latency_ms: float,
    settings: Settings,
) -> RetrievalMetrics:
    """
    Full retrieval evaluation pipeline.
    Computes score-based and LLM-judged metrics.
    """
    scores = [c.score for c in chunks_after_rerank if c.score is not None]
    score_stats = compute_score_metrics(scores)

    # LLM relevance judgment
    relevance_scores = judge_relevance_with_llm(query, chunks_after_rerank, settings)

    precision = None
    mrr = None
    rerank_gain = None
    notes = []

    if relevance_scores:
        k = min(5, len(relevance_scores))
        precision = compute_precision_at_k(relevance_scores, k)
        mrr = compute_mrr(relevance_scores)

        # Compute rerank gain if reranking was applied
        if reranked and chunks_before_rerank:
            before_scores = judge_relevance_with_llm(query, chunks_before_rerank[:k], settings)
            if before_scores:
                before_precision = compute_precision_at_k(before_scores, k)
                rerank_gain = round(precision - before_precision, 4)
                if rerank_gain > 0:
                    notes.append(f"Reranking improved Precision@{k} by +{rerank_gain:.2%}")
                elif rerank_gain < 0:
                    notes.append(f"Reranking did not improve results for this query")
                else:
                    notes.append("Reranking had no measurable effect on this query")

    if not relevance_scores:
        notes.append("LLM relevance judgment unavailable (no API key or error)")

    if expanded_query:
        variants = expanded_query.split(" | ")
        notes.append(f"Query expanded into {len(variants)} variant(s)")

    return RetrievalMetrics(
        query=query,
        chunks_retrieved=len(chunks_after_rerank),
        avg_score=score_stats["avg"],
        max_score=score_stats["max"],
        min_score=score_stats["min"],
        score_std=score_stats["std"],
        expanded_query=expanded_query,
        reranked=reranked,
        latency_ms=round(latency_ms, 2),
        relevance_scores=[round(s, 2) for s in relevance_scores],
        precision_at_k=precision,
        mrr=mrr,
        rerank_gain=rerank_gain,
        notes=notes,
    )
