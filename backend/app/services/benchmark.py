"""
Benchmark Suite for Retrieval Evaluation

Implements rigorous IR evaluation with:
1. Labeled QA datasets with ground truth
2. Standard IR metrics (P@k, R@k, MRR, NDCG, MAP)
3. Comparative benchmarking across retrieval strategies
4. Regression testing for retrieval quality

This addresses the key weakness: moving from heuristic/LLM-judged
evaluation to objective, reproducible metrics.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sqlmodel import Session

from app.core.config import Settings
from app.schemas.search import RetrievalChunk
from app.services.retrieval import retrieve_chunks

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkQuery:
    """Single query with ground truth relevant chunks."""
    query: str
    relevant_chunk_ids: list[str]  # Ground truth
    document_id: str | None = None
    category: str = "general"  # e.g., "factual", "conceptual", "multi-hop"


@dataclass
class BenchmarkDataset:
    """Collection of labeled queries for evaluation."""
    name: str
    description: str
    queries: list[BenchmarkQuery]
    
    @classmethod
    def from_json(cls, path: Path) -> BenchmarkDataset:
        """Load benchmark dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)
        queries = [
            BenchmarkQuery(
                query=q["query"],
                relevant_chunk_ids=q["relevant_chunk_ids"],
                document_id=q.get("document_id"),
                category=q.get("category", "general"),
            )
            for q in data["queries"]
        ]
        return cls(
            name=data["name"],
            description=data["description"],
            queries=queries,
        )
    
    def to_json(self, path: Path) -> None:
        """Save benchmark dataset to JSON file."""
        data = {
            "name": self.name,
            "description": self.description,
            "queries": [
                {
                    "query": q.query,
                    "relevant_chunk_ids": q.relevant_chunk_ids,
                    "document_id": q.document_id,
                    "category": q.category,
                }
                for q in self.queries
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


@dataclass
class IRMetrics:
    """Standard Information Retrieval metrics."""
    precision_at_k: dict[int, float] = field(default_factory=dict)  # P@1, P@3, P@5, P@10
    recall_at_k: dict[int, float] = field(default_factory=dict)  # R@1, R@3, R@5, R@10
    mrr: float = 0.0  # Mean Reciprocal Rank
    map_score: float = 0.0  # Mean Average Precision
    ndcg_at_k: dict[int, float] = field(default_factory=dict)  # NDCG@5, NDCG@10
    
    def to_dict(self) -> dict:
        return {
            "precision": self.precision_at_k,
            "recall": self.recall_at_k,
            "mrr": round(self.mrr, 4),
            "map": round(self.map_score, 4),
            "ndcg": self.ndcg_at_k,
        }


@dataclass
class BenchmarkResult:
    """Results for a single query."""
    query: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    reciprocal_rank: float
    average_precision: float
    ndcg_at_k: dict[int, float]
    latency_ms: float
    category: str = "general"


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results across all queries."""
    strategy_name: str
    dataset_name: str
    metrics: IRMetrics
    per_query_results: list[BenchmarkResult]
    total_queries: int
    avg_latency_ms: float
    timestamp: str
    
    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "dataset": self.dataset_name,
            "metrics": self.metrics.to_dict(),
            "total_queries": self.total_queries,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "timestamp": self.timestamp,
            "per_query": [
                {
                    "query": r.query,
                    "category": r.category,
                    "precision@5": r.precision_at_k.get(5, 0.0),
                    "recall@5": r.recall_at_k.get(5, 0.0),
                    "mrr": r.reciprocal_rank,
                    "latency_ms": r.latency_ms,
                }
                for r in self.per_query_results
            ],
        }
    
    def save(self, path: Path) -> None:
        """Save benchmark report to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision@k: fraction of top-k results that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for chunk_id in top_k if chunk_id in relevant)
    return relevant_in_top_k / k


def compute_recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@k: fraction of relevant items found in top-k."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for chunk_id in top_k if chunk_id in relevant)
    return relevant_in_top_k / len(relevant)


def compute_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank: 1/rank of first relevant result."""
    for i, chunk_id in enumerate(retrieved):
        if chunk_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Average Precision: mean of precision values at each relevant result."""
    if not relevant:
        return 0.0
    
    precisions = []
    relevant_count = 0
    
    for i, chunk_id in enumerate(retrieved):
        if chunk_id in relevant:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precisions.append(precision_at_i)
    
    return sum(precisions) / len(relevant) if precisions else 0.0


def compute_dcg(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Discounted Cumulative Gain@k."""
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved[:k]):
        if chunk_id in relevant:
            # Binary relevance: 1 if relevant, 0 otherwise
            # DCG formula: sum(rel_i / log2(i + 2))
            dcg += 1.0 / (i + 2).bit_length()  # log2(i + 2) approximation
    return dcg


def compute_ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain@k."""
    if not relevant:
        return 0.0
    
    dcg = compute_dcg(retrieved, relevant, k)
    
    # Ideal DCG: all relevant items ranked first
    ideal_retrieved = list(relevant) + [x for x in retrieved if x not in relevant]
    idcg = compute_dcg(ideal_retrieved, relevant, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_single_query(
    session: Session,
    query: BenchmarkQuery,
    settings: Settings,
    strategy: Literal["vector_only", "bm25_only", "hybrid", "hybrid_rerank"],
    top_k: int = 10,
) -> BenchmarkResult:
    """Evaluate retrieval for a single query with ground truth."""
    
    # Configure retrieval strategy
    use_hybrid = strategy in ["hybrid", "hybrid_rerank"]
    enable_expansion = strategy == "hybrid_rerank"
    
    # Override reranking temporarily
    original_groq_key = settings.groq_api_key
    if strategy != "hybrid_rerank":
        settings.groq_api_key = None  # Disable reranking
    
    start_time = time.time()
    
    try:
        result = retrieve_chunks(
            session=session,
            query=query.query,
            settings=settings,
            top_k=top_k,
            document_ids=[query.document_id] if query.document_id else None,
            use_hybrid=use_hybrid,
            enable_query_expansion=enable_expansion,
        )
    finally:
        settings.groq_api_key = original_groq_key  # Restore
    
    latency_ms = (time.time() - start_time) * 1000
    
    retrieved_ids = [chunk.chunk_id for chunk in result.chunks]
    relevant_set = set(query.relevant_chunk_ids)
    
    # Compute all metrics
    k_values = [1, 3, 5, 10]
    precision = {k: compute_precision_at_k(retrieved_ids, relevant_set, k) for k in k_values}
    recall = {k: compute_recall_at_k(retrieved_ids, relevant_set, k) for k in k_values}
    rr = compute_reciprocal_rank(retrieved_ids, relevant_set)
    ap = compute_average_precision(retrieved_ids, relevant_set)
    ndcg = {k: compute_ndcg_at_k(retrieved_ids, relevant_set, k) for k in [5, 10]}
    
    return BenchmarkResult(
        query=query.query,
        retrieved_chunk_ids=retrieved_ids,
        relevant_chunk_ids=query.relevant_chunk_ids,
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=rr,
        average_precision=ap,
        ndcg_at_k=ndcg,
        latency_ms=latency_ms,
        category=query.category,
    )


def run_benchmark(
    session: Session,
    dataset: BenchmarkDataset,
    settings: Settings,
    strategy: Literal["vector_only", "bm25_only", "hybrid", "hybrid_rerank"],
    top_k: int = 10,
) -> BenchmarkReport:
    """
    Run full benchmark evaluation on a dataset.
    
    Strategies:
    - vector_only: FAISS semantic search only
    - bm25_only: BM25 keyword search only (requires hybrid=True, vector_weight=0)
    - hybrid: FAISS + BM25 fusion
    - hybrid_rerank: hybrid + LLM reranking + query expansion
    """
    logger.info(f"Running benchmark: {dataset.name} with strategy={strategy}")
    
    results: list[BenchmarkResult] = []
    
    for query in dataset.queries:
        try:
            result = evaluate_single_query(
                session=session,
                query=query,
                settings=settings,
                strategy=strategy,
                top_k=top_k,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Query failed: {query.query[:50]}... Error: {e}")
    
    # Aggregate metrics
    if not results:
        raise ValueError("No successful query evaluations")
    
    k_values = [1, 3, 5, 10]
    aggregated_metrics = IRMetrics(
        precision_at_k={
            k: round(sum(r.precision_at_k[k] for r in results) / len(results), 4)
            for k in k_values
        },
        recall_at_k={
            k: round(sum(r.recall_at_k[k] for r in results) / len(results), 4)
            for k in k_values
        },
        mrr=round(sum(r.reciprocal_rank for r in results) / len(results), 4),
        map_score=round(sum(r.average_precision for r in results) / len(results), 4),
        ndcg_at_k={
            k: round(sum(r.ndcg_at_k[k] for r in results) / len(results), 4)
            for k in [5, 10]
        },
    )
    
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat()
    
    return BenchmarkReport(
        strategy_name=strategy,
        dataset_name=dataset.name,
        metrics=aggregated_metrics,
        per_query_results=results,
        total_queries=len(results),
        avg_latency_ms=avg_latency,
        timestamp=timestamp,
    )


def compare_strategies(
    session: Session,
    dataset: BenchmarkDataset,
    settings: Settings,
    strategies: list[Literal["vector_only", "bm25_only", "hybrid", "hybrid_rerank"]],
    output_dir: Path,
) -> dict[str, BenchmarkReport]:
    """
    Compare multiple retrieval strategies on the same dataset.
    Returns comparative analysis showing which strategy performs best.
    """
    reports = {}
    
    for strategy in strategies:
        report = run_benchmark(session, dataset, settings, strategy)
        reports[strategy] = report
        
        # Save individual report
        report_path = output_dir / f"{dataset.name}_{strategy}.json"
        report.save(report_path)
        logger.info(f"Saved report: {report_path}")
    
    # Generate comparison summary
    comparison = {
        "dataset": dataset.name,
        "strategies": {},
        "winner": {},
    }
    
    for strategy, report in reports.items():
        comparison["strategies"][strategy] = report.metrics.to_dict()
    
    # Determine best strategy per metric
    metrics_to_compare = ["mrr", "map"]
    for metric in metrics_to_compare:
        best_strategy = max(
            reports.items(),
            key=lambda x: getattr(x[1].metrics, metric if metric != "map" else "map_score")
        )
        comparison["winner"][metric] = {
            "strategy": best_strategy[0],
            "score": getattr(best_strategy[1].metrics, metric if metric != "map" else "map_score"),
        }
    
    # Save comparison
    comparison_path = output_dir / f"{dataset.name}_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Comparison saved: {comparison_path}")
    return reports
