# Retrieval Benchmark Suite

This directory contains labeled datasets for rigorous retrieval evaluation.

## Why Benchmarks Matter

Without labeled ground truth, evaluation is:
- **Heuristic**: based on score distributions, not actual relevance
- **LLM-judged**: expensive, non-deterministic, self-referential
- **Unscientific**: cannot prove retrieval improvements objectively

With benchmarks, you get:
- **Reproducible metrics**: P@k, R@k, MRR, MAP, NDCG
- **Regression testing**: detect retrieval quality degradation
- **Strategy comparison**: prove which approach works best
- **Scientific rigor**: objective evaluation against ground truth

## Dataset Format

Each benchmark dataset is a JSON file with:

```json
{
  "name": "dataset_name",
  "description": "What this dataset tests",
  "queries": [
    {
      "query": "What is the capital of France?",
      "relevant_chunk_ids": ["chunk_abc123", "chunk_def456"],
      "document_id": "doc_xyz789",
      "category": "factual"
    }
  ]
}
```

### Fields

- **query**: The search query
- **relevant_chunk_ids**: Ground truth - which chunks should be retrieved
- **document_id** (optional): Restrict search to specific document
- **category** (optional): Query type (factual, conceptual, multi-hop, etc.)

## Creating Benchmarks

### Option 1: Manual Labeling

1. Upload documents to your system
2. Note chunk IDs from database
3. Create queries and label relevant chunks
4. Save as JSON in this directory

### Option 2: From Existing QA Pairs

If you have FAQ documents or QA datasets:

```python
from app.services.benchmark import BenchmarkDataset, BenchmarkQuery

queries = [
    BenchmarkQuery(
        query="How do I reset my password?",
        relevant_chunk_ids=["chunk_001", "chunk_002"],
        category="procedural"
    ),
    # ... more queries
]

dataset = BenchmarkDataset(
    name="customer_support_faq",
    description="Common customer support questions",
    queries=queries
)

dataset.to_json(Path("benchmarks/customer_support.json"))
```

### Option 3: Sample from Production Logs

1. Extract real user queries from logs
2. Have domain experts label relevant chunks
3. Build dataset from production patterns

## Running Benchmarks

```python
from pathlib import Path
from sqlmodel import Session
from app.core.config import Settings
from app.services.benchmark import BenchmarkDataset, run_benchmark, compare_strategies

# Load dataset
dataset = BenchmarkDataset.from_json(Path("benchmarks/sample_dataset.json"))

# Run single strategy
report = run_benchmark(
    session=session,
    dataset=dataset,
    settings=settings,
    strategy="hybrid_rerank",
    top_k=10
)

print(f"MRR: {report.metrics.mrr}")
print(f"MAP: {report.metrics.map_score}")
print(f"P@5: {report.metrics.precision_at_k[5]}")

# Compare all strategies
reports = compare_strategies(
    session=session,
    dataset=dataset,
    settings=settings,
    strategies=["vector_only", "bm25_only", "hybrid", "hybrid_rerank"],
    output_dir=Path("benchmarks/results")
)
```

## Metrics Explained

### Precision@k (P@k)
Fraction of top-k results that are relevant.
- P@5 = 0.8 means 4 out of 5 top results were relevant
- Higher is better
- Measures accuracy of top results

### Recall@k (R@k)
Fraction of all relevant items found in top-k.
- R@5 = 0.5 means found 50% of all relevant chunks in top 5
- Higher is better
- Measures completeness

### Mean Reciprocal Rank (MRR)
Average of 1/rank of first relevant result.
- MRR = 0.5 means first relevant result at rank 2 on average
- Range: 0-1, higher is better
- Measures how quickly users find relevant results

### Mean Average Precision (MAP)
Mean of average precision across all queries.
- Considers both precision and ranking quality
- Range: 0-1, higher is better
- Gold standard for ranking quality

### NDCG@k
Normalized Discounted Cumulative Gain.
- Rewards relevant results ranked higher
- Range: 0-1, higher is better
- Accounts for position of relevant results

## Benchmark Strategies

### vector_only
- FAISS semantic search only
- Good for: conceptual queries, paraphrasing
- Weak for: exact keyword matches

### bm25_only
- BM25 keyword search only
- Good for: exact terms, technical jargon
- Weak for: semantic similarity, synonyms

### hybrid
- FAISS + BM25 fusion (65% vector, 35% BM25)
- Good for: balanced retrieval
- Best general-purpose approach

### hybrid_rerank
- Hybrid + LLM reranking + query expansion
- Good for: maximum precision
- Tradeoff: higher latency, token cost

## Example Results

```json
{
  "strategy": "hybrid_rerank",
  "dataset": "technical_docs",
  "metrics": {
    "precision": {"1": 0.95, "3": 0.87, "5": 0.82, "10": 0.71},
    "recall": {"1": 0.23, "3": 0.51, "5": 0.68, "10": 0.89},
    "mrr": 0.8234,
    "map": 0.7891,
    "ndcg": {"5": 0.8456, "10": 0.8123}
  },
  "avg_latency_ms": 1247.3
}
```

## Best Practices

1. **Diverse queries**: Include factual, conceptual, multi-hop questions
2. **Multiple relevant chunks**: Some queries should have 1 relevant chunk, others 5+
3. **Negative examples**: Include queries with no relevant chunks
4. **Category balance**: Mix easy and hard queries
5. **Regular updates**: Add new queries as system evolves
6. **Version control**: Track benchmark datasets in git

## Regression Testing

Run benchmarks before/after changes:

```bash
# Baseline
python -m app.scripts.run_benchmark --dataset benchmarks/prod_queries.json --output results/baseline.json

# After optimization
python -m app.scripts.run_benchmark --dataset benchmarks/prod_queries.json --output results/optimized.json

# Compare
python -m app.scripts.compare_results results/baseline.json results/optimized.json
```

## Integration with CI/CD

Add to your pipeline:

```yaml
- name: Run Retrieval Benchmarks
  run: |
    python -m app.scripts.run_benchmark --all
    python -m app.scripts.check_regression --threshold 0.05
```

This fails the build if MRR drops by more than 5%.
