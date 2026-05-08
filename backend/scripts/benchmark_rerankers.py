"""
Reranker Benchmark Script

Compares reranking strategies:
1. No reranking (baseline)
2. Cross-encoder reranking
3. LLM reranking

Metrics measured:
- Latency (ms)
- Precision@k
- NDCG@k
- Token cost (for LLM)
- Determinism (score variance across runs)

Usage:
    python -m scripts.benchmark_rerankers --queries 20 --runs 3
"""

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from sqlmodel import Session, create_engine

from app.core.config import Settings
from app.db.models import Chunk, Document
from app.schemas.search import RetrievalChunk
from app.services.reranker import rerank_chunks, RerankerType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_chunks(n: int = 10) -> list[RetrievalChunk]:
    """Create synthetic test chunks with varying relevance."""
    chunks = []
    
    # Highly relevant chunks
    chunks.append(RetrievalChunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        chunk_index=0,
        text="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        score=0.85,
        page_number=1,
        document_name="python_guide.pdf",
    ))
    
    chunks.append(RetrievalChunk(
        chunk_id="chunk_2",
        document_id="doc_1",
        chunk_index=1,
        text="Python's syntax emphasizes code readability with significant whitespace. The language provides constructs that enable clear programming on both small and large scales.",
        score=0.82,
        page_number=1,
        document_name="python_guide.pdf",
    ))
    
    # Moderately relevant
    chunks.append(RetrievalChunk(
        chunk_id="chunk_3",
        document_id="doc_2",
        chunk_index=0,
        text="Programming languages can be classified into different paradigms. Some languages support multiple paradigms while others are designed for specific use cases.",
        score=0.75,
        page_number=5,
        document_name="programming_concepts.pdf",
    ))
    
    # Less relevant
    chunks.append(RetrievalChunk(
        chunk_id="chunk_4",
        document_id="doc_3",
        chunk_index=0,
        text="JavaScript is primarily used for web development. It runs in web browsers and enables interactive web pages through DOM manipulation.",
        score=0.68,
        page_number=10,
        document_name="web_dev.pdf",
    ))
    
    chunks.append(RetrievalChunk(
        chunk_id="chunk_5",
        document_id="doc_3",
        chunk_index=1,
        text="The history of computing dates back to the 1940s with the development of the first electronic computers. These machines were large and expensive.",
        score=0.62,
        page_number=15,
        document_name="computing_history.pdf",
    ))
    
    # Irrelevant
    chunks.append(RetrievalChunk(
        chunk_id="chunk_6",
        document_id="doc_4",
        chunk_index=0,
        text="Cooking pasta requires boiling water with salt. The pasta should be cooked until al dente, which typically takes 8-12 minutes depending on the type.",
        score=0.45,
        page_number=20,
        document_name="recipes.pdf",
    ))
    
    chunks.append(RetrievalChunk(
        chunk_id="chunk_7",
        document_id="doc_4",
        chunk_index=1,
        text="The weather forecast predicts rain for the next three days. Temperatures will range from 15 to 20 degrees Celsius with high humidity.",
        score=0.38,
        page_number=25,
        document_name="weather.pdf",
    ))
    
    return chunks[:n]


def compute_ndcg(ranked_ids: list[str], relevance_scores: dict[str, float], k: int) -> float:
    """
    Compute NDCG@k.
    
    NDCG (Normalized Discounted Cumulative Gain) measures ranking quality
    by comparing actual ranking to ideal ranking.
    """
    def dcg(scores: list[float]) -> float:
        return sum(score / (i + 2).bit_length() for i, score in enumerate(scores))
    
    # Actual DCG
    actual_scores = [relevance_scores.get(chunk_id, 0.0) for chunk_id in ranked_ids[:k]]
    actual_dcg = dcg(actual_scores)
    
    # Ideal DCG (best possible ranking)
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal_scores)
    
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Precision@k."""
    top_k = ranked_ids[:k]
    relevant_in_top_k = sum(1 for chunk_id in top_k if chunk_id in relevant_ids)
    return relevant_in_top_k / k if k > 0 else 0.0


def benchmark_reranker(
    query: str,
    chunks: list[RetrievalChunk],
    settings: Settings,
    reranker_type: str,
    runs: int = 3,
) -> dict:
    """Benchmark a single reranker across multiple runs."""
    latencies = []
    rankings = []
    
    for run in range(runs):
        start_time = time.time()
        result = rerank_chunks(
            query=query,
            chunks=chunks.copy(),
            settings=settings,
            top_k=5,
            reranker=reranker_type,
        )
        latency_ms = (time.time() - start_time) * 1000
        
        latencies.append(latency_ms)
        rankings.append([c.chunk_id for c in result.chunks[:5]])
    
    # Compute determinism (how consistent are rankings across runs?)
    if runs > 1:
        # Count how many times each chunk appears in each position
        position_variance = []
        for pos in range(5):
            chunks_at_pos = [ranking[pos] if pos < len(ranking) else None for ranking in rankings]
            unique_chunks = len(set(c for c in chunks_at_pos if c))
            position_variance.append(unique_chunks)
        determinism_score = 1.0 - (sum(position_variance) - 5) / (5 * (runs - 1))
    else:
        determinism_score = 1.0
    
    return {
        "reranker": reranker_type,
        "avg_latency_ms": round(statistics.mean(latencies), 2),
        "std_latency_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0.0,
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "determinism_score": round(determinism_score, 3),
        "sample_ranking": rankings[0],
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark reranking strategies")
    parser.add_argument("--queries", type=int, default=5, help="Number of test queries")
    parser.add_argument("--runs", type=int, default=3, help="Runs per query per reranker")
    parser.add_argument("--output", type=str, default="reranker_benchmark.json", help="Output file")
    args = parser.parse_args()
    
    settings = Settings()
    
    # Test queries with known relevance
    test_cases = [
        {
            "query": "What is Python programming language?",
            "relevant_ids": {"chunk_1", "chunk_2"},
            "relevance_scores": {
                "chunk_1": 1.0,
                "chunk_2": 0.9,
                "chunk_3": 0.5,
                "chunk_4": 0.2,
                "chunk_5": 0.1,
                "chunk_6": 0.0,
                "chunk_7": 0.0,
            }
        },
        {
            "query": "Explain programming paradigms",
            "relevant_ids": {"chunk_1", "chunk_3"},
            "relevance_scores": {
                "chunk_1": 0.8,
                "chunk_2": 0.6,
                "chunk_3": 1.0,
                "chunk_4": 0.3,
                "chunk_5": 0.1,
                "chunk_6": 0.0,
                "chunk_7": 0.0,
            }
        },
        {
            "query": "How to cook pasta?",
            "relevant_ids": {"chunk_6"},
            "relevance_scores": {
                "chunk_1": 0.0,
                "chunk_2": 0.0,
                "chunk_3": 0.0,
                "chunk_4": 0.0,
                "chunk_5": 0.0,
                "chunk_6": 1.0,
                "chunk_7": 0.0,
            }
        },
    ]
    
    rerankers = ["none", "cross_encoder", "llm"]
    results = []
    
    logger.info(f"Running benchmark with {args.queries} queries, {args.runs} runs per reranker")
    
    for i, test_case in enumerate(test_cases[:args.queries]):
        query = test_case["query"]
        relevant_ids = test_case["relevant_ids"]
        relevance_scores = test_case["relevance_scores"]
        
        logger.info(f"\nQuery {i+1}: {query}")
        
        chunks = create_test_chunks()
        query_results = []
        
        for reranker in rerankers:
            logger.info(f"  Testing {reranker}...")
            
            benchmark_result = benchmark_reranker(
                query=query,
                chunks=chunks,
                settings=settings,
                reranker_type=reranker,
                runs=args.runs,
            )
            
            # Compute quality metrics
            ranked_ids = benchmark_result["sample_ranking"]
            precision_at_3 = compute_precision_at_k(ranked_ids, relevant_ids, 3)
            precision_at_5 = compute_precision_at_k(ranked_ids, relevant_ids, 5)
            ndcg_at_5 = compute_ndcg(ranked_ids, relevance_scores, 5)
            
            benchmark_result.update({
                "precision@3": round(precision_at_3, 3),
                "precision@5": round(precision_at_5, 3),
                "ndcg@5": round(ndcg_at_5, 3),
            })
            
            query_results.append(benchmark_result)
            
            logger.info(f"    Latency: {benchmark_result['avg_latency_ms']:.1f}ms ± {benchmark_result['std_latency_ms']:.1f}ms")
            logger.info(f"    P@3: {precision_at_3:.3f}, P@5: {precision_at_5:.3f}, NDCG@5: {ndcg_at_5:.3f}")
            logger.info(f"    Determinism: {benchmark_result['determinism_score']:.3f}")
        
        results.append({
            "query": query,
            "rerankers": query_results,
        })
    
    # Aggregate results
    aggregated = {}
    for reranker in rerankers:
        reranker_results = [
            r for result in results
            for r in result["rerankers"]
            if r["reranker"] == reranker
        ]
        
        aggregated[reranker] = {
            "avg_latency_ms": round(statistics.mean([r["avg_latency_ms"] for r in reranker_results]), 2),
            "avg_precision@3": round(statistics.mean([r["precision@3"] for r in reranker_results]), 3),
            "avg_precision@5": round(statistics.mean([r["precision@5"] for r in reranker_results]), 3),
            "avg_ndcg@5": round(statistics.mean([r["ndcg@5"] for r in reranker_results]), 3),
            "avg_determinism": round(statistics.mean([r["determinism_score"] for r in reranker_results]), 3),
        }
    
    # Save results
    output = {
        "summary": aggregated,
        "detailed_results": results,
        "config": {
            "queries": args.queries,
            "runs_per_query": args.runs,
            "rerankers_tested": rerankers,
        }
    }
    
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"{'='*60}")
    
    for reranker, metrics in aggregated.items():
        logger.info(f"\n{reranker.upper()}:")
        logger.info(f"  Latency:      {metrics['avg_latency_ms']:.1f}ms")
        logger.info(f"  Precision@3:  {metrics['avg_precision@3']:.3f}")
        logger.info(f"  Precision@5:  {metrics['avg_precision@5']:.3f}")
        logger.info(f"  NDCG@5:       {metrics['avg_ndcg@5']:.3f}")
        logger.info(f"  Determinism:  {metrics['avg_determinism']:.3f}")
    
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
