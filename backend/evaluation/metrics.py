"""
Retrieval evaluation metrics: NDCG@k, MRR, Recall@k, Precision@k
"""

import math
from typing import Any


def dcg_at_k(relevances: list[int], k: int) -> float:
    """
    Discounted Cumulative Gain at k.
    
    DCG@k = sum(rel_i / log2(i+1)) for i in [1, k]
    """
    relevances = relevances[:k]
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    
    NDCG@k = DCG@k / IDCG@k
    where IDCG@k is DCG@k of ideal ranking (sorted by relevance)
    """
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def mrr(relevances: list[int], threshold: int = 1) -> float:
    """
    Mean Reciprocal Rank.
    
    MRR = 1 / rank of first relevant result
    """
    for idx, rel in enumerate(relevances):
        if rel >= threshold:
            return 1.0 / (idx + 1)
    return 0.0


def precision_at_k(relevances: list[int], k: int, threshold: int = 1) -> float:
    """
    Precision at k.
    
    P@k = (# relevant in top-k) / k
    """
    relevances = relevances[:k]
    relevant_count = sum(1 for rel in relevances if rel >= threshold)
    return relevant_count / k if k > 0 else 0.0


def recall_at_k(relevances: list[int], k: int, total_relevant: int, threshold: int = 1) -> float:
    """
    Recall at k.
    
    R@k = (# relevant in top-k) / (total # relevant)
    """
    relevances = relevances[:k]
    relevant_count = sum(1 for rel in relevances if rel >= threshold)
    return relevant_count / total_relevant if total_relevant > 0 else 0.0


def average_precision(relevances: list[int], threshold: int = 1) -> float:
    """
    Average Precision.
    
    AP = sum(P@k * rel_k) / total_relevant
    """
    relevant_count = 0
    precision_sum = 0.0
    
    for idx, rel in enumerate(relevances):
        if rel >= threshold:
            relevant_count += 1
            precision_sum += relevant_count / (idx + 1)
    
    total_relevant = sum(1 for rel in relevances if rel >= threshold)
    return precision_sum / total_relevant if total_relevant > 0 else 0.0


def evaluate_ranking(
    retrieved_chunks: list[dict[str, Any]],
    gold_relevances: dict[str, int],
    k: int = 10
) -> dict[str, float]:
    """
    Evaluate a ranking against gold relevances.
    
    Args:
        retrieved_chunks: List of retrieved chunks with chunk_id
        gold_relevances: Dict mapping chunk_id -> relevance score (0-3)
        k: Cutoff for metrics
    
    Returns:
        Dict of metric name -> score
    """
    # Extract relevance scores for retrieved chunks
    relevances = [
        gold_relevances.get(chunk["chunk_id"], 0)
        for chunk in retrieved_chunks
    ]
    
    total_relevant = sum(1 for rel in gold_relevances.values() if rel >= 1)
    
    return {
        "ndcg@5": ndcg_at_k(relevances, 5),
        "ndcg@10": ndcg_at_k(relevances, 10),
        "mrr": mrr(relevances),
        "precision@5": precision_at_k(relevances, 5),
        "precision@10": precision_at_k(relevances, 10),
        "recall@5": recall_at_k(relevances, 5, total_relevant),
        "recall@10": recall_at_k(relevances, 10, total_relevant),
        "map": average_precision(relevances),
    }
