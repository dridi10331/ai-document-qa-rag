"""
Retrieval evaluation harness for comparing configurations.
"""

import statistics
from typing import Any

from evaluation.dataset import get_evaluation_dataset, get_gold_relevance
from evaluation.metrics import evaluate_ranking


class RetrievalEvaluationHarness:
    """Harness for evaluating retrieval configurations."""
    
    def __init__(self):
        self.dataset = get_evaluation_dataset()
    
    def evaluate_configuration(
        self,
        retrieval_fn: callable,
        config_name: str,
        top_k: int = 10
    ) -> dict[str, Any]:
        """
        Evaluate a retrieval configuration on full dataset.
        
        Args:
            retrieval_fn: Function that takes query string, returns list of chunks
            config_name: Name of configuration (e.g., "vector_only", "hybrid")
            top_k: Number of chunks to retrieve
        
        Returns:
            Aggregated metrics across all queries
        """
        all_metrics = []
        query_results = []
        
        for eval_query in self.dataset:
            query = eval_query["query"]
            query_id = eval_query["query_id"]
            
            # Retrieve chunks using provided function
            retrieved_chunks = retrieval_fn(query, top_k=top_k)
            
            # Build gold relevances dict
            gold_relevances = {
                judgment["chunk_id"]: judgment["relevance"]
                for judgment in eval_query["gold_chunks"]
            }
            
            # Evaluate ranking
            metrics = evaluate_ranking(retrieved_chunks, gold_relevances, k=top_k)
            all_metrics.append(metrics)
            
            query_results.append({
                "query_id": query_id,
                "query": query,
                "query_type": eval_query["query_type"],
                "metrics": metrics,
                "retrieved_count": len(retrieved_chunks),
            })
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_metrics)
        
        return {
            "config_name": config_name,
            "num_queries": len(self.dataset),
            "aggregated_metrics": aggregated,
            "query_results": query_results,
        }
    
    def _aggregate_metrics(self, all_metrics: list[dict[str, float]]) -> dict[str, Any]:
        """Aggregate metrics across queries with mean and std dev."""
        metric_names = all_metrics[0].keys()
        
        aggregated = {}
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics]
            aggregated[metric_name] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        
        return aggregated
    
    def compare_configurations(
        self,
        configs: list[tuple[str, callable]],
        top_k: int = 10
    ) -> dict[str, Any]:
        """
        Compare multiple retrieval configurations.
        
        Args:
            configs: List of (config_name, retrieval_fn) tuples
            top_k: Number of chunks to retrieve
        
        Returns:
            Comparison results
        """
        results = []
        
        for config_name, retrieval_fn in configs:
            result = self.evaluate_configuration(retrieval_fn, config_name, top_k)
            results.append(result)
        
        return {
            "comparison": results,
            "winner": self._determine_winner(results),
        }
    
    def _determine_winner(self, results: list[dict]) -> dict[str, str]:
        """Determine best configuration for each metric."""
        winners = {}
        
        metric_names = results[0]["aggregated_metrics"].keys()
        
        for metric_name in metric_names:
            best_config = max(
                results,
                key=lambda r: r["aggregated_metrics"][metric_name]["mean"]
            )
            winners[metric_name] = best_config["config_name"]
        
        return winners
