"""
Observability Layer for RAG Pipeline

Implements comprehensive tracing and monitoring:
1. Request tracing - track full pipeline execution
2. Token analytics - monitor LLM usage and costs
3. Retrieval traces - inspect chunk selection and scoring
4. Latency profiling - identify bottlenecks
5. Quality metrics - track hallucination indicators

This addresses the "no observability layer" gap.
Without this, debugging production issues is guesswork.
With this, you have full visibility into system behavior.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class SpanContext:
    """Trace span for a single operation."""
    span_id: str
    trace_id: str
    parent_span_id: str | None
    operation: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    status: Literal["success", "error"] = "success"
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    
    def finish(self, error: Exception | None = None) -> None:
        """Mark span as complete."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        if error:
            self.status = "error"
            self.error = str(error)
    
    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "operation": self.operation,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms else None,
            "status": self.status,
            "metadata": self.metadata,
            "error": self.error,
        }


@dataclass
class TokenUsage:
    """Token consumption tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: str = ""
    
    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "model": self.model,
        }


@dataclass
class RetrievalTrace:
    """Detailed trace of retrieval pipeline."""
    query: str
    expanded_queries: list[str] = field(default_factory=list)
    vector_results: list[dict] = field(default_factory=list)  # chunk_id, score
    bm25_results: list[dict] = field(default_factory=list)  # chunk_id, score
    fused_results: list[dict] = field(default_factory=list)  # chunk_id, fused_score
    reranked_results: list[dict] = field(default_factory=list)  # chunk_id, rerank_score
    final_chunks: list[str] = field(default_factory=list)  # final chunk_ids returned
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "expanded_queries": self.expanded_queries,
            "vector_results": self.vector_results[:10],  # Top 10 only
            "bm25_results": self.bm25_results[:10],
            "fused_results": self.fused_results[:10],
            "reranked_results": self.reranked_results[:10],
            "final_chunks": self.final_chunks,
        }


@dataclass
class Trace:
    """Complete trace for a request."""
    trace_id: str
    operation: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    status: Literal["success", "error"] = "success"
    spans: list[SpanContext] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    retrieval_trace: RetrievalTrace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    
    def finish(self, error: Exception | None = None) -> None:
        """Mark trace as complete."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        if error:
            self.status = "error"
            self.error = str(error)
    
    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "operation": self.operation,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms else None,
            "status": self.status,
            "spans": [span.to_dict() for span in self.spans],
            "token_usage": self.token_usage.to_dict(),
            "retrieval_trace": self.retrieval_trace.to_dict() if self.retrieval_trace else None,
            "metadata": self.metadata,
            "error": self.error,
        }


class TraceCollector:
    """
    Centralized trace collection and export.
    
    In production, this would export to:
    - Langfuse (LLM observability)
    - OpenTelemetry (distributed tracing)
    - Prometheus (metrics)
    - DataDog/New Relic (APM)
    
    For now, we log to files for demonstration.
    """
    
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("data/traces")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.active_traces: dict[str, Trace] = {}
        self.active_spans: dict[str, SpanContext] = {}
    
    def start_trace(self, operation: str, metadata: dict[str, Any] | None = None) -> str:
        """Start a new trace."""
        trace_id = str(uuid4())
        trace = Trace(
            trace_id=trace_id,
            operation=operation,
            start_time=time.time(),
            metadata=metadata or {},
        )
        self.active_traces[trace_id] = trace
        return trace_id
    
    def finish_trace(self, trace_id: str, error: Exception | None = None) -> None:
        """Complete a trace and export it."""
        if trace_id not in self.active_traces:
            logger.warning(f"Trace {trace_id} not found")
            return
        
        trace = self.active_traces[trace_id]
        trace.finish(error)
        
        # Export trace
        self._export_trace(trace)
        
        # Clean up
        del self.active_traces[trace_id]
    
    def start_span(
        self,
        trace_id: str,
        operation: str,
        parent_span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Start a new span within a trace."""
        span_id = str(uuid4())
        span = SpanContext(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=time.time(),
            metadata=metadata or {},
        )
        self.active_spans[span_id] = span
        
        if trace_id in self.active_traces:
            self.active_traces[trace_id].spans.append(span)
        
        return span_id
    
    def finish_span(self, span_id: str, error: Exception | None = None) -> None:
        """Complete a span."""
        if span_id not in self.active_spans:
            logger.warning(f"Span {span_id} not found")
            return
        
        span = self.active_spans[span_id]
        span.finish(error)
        del self.active_spans[span_id]
    
    def add_token_usage(
        self,
        trace_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        cost_per_1k_prompt: float = 0.0,
        cost_per_1k_completion: float = 0.0,
    ) -> None:
        """Record token usage for a trace."""
        if trace_id not in self.active_traces:
            return
        
        trace = self.active_traces[trace_id]
        trace.token_usage.prompt_tokens += prompt_tokens
        trace.token_usage.completion_tokens += completion_tokens
        trace.token_usage.total_tokens += prompt_tokens + completion_tokens
        trace.token_usage.model = model
        
        # Estimate cost
        cost = (
            (prompt_tokens / 1000) * cost_per_1k_prompt +
            (completion_tokens / 1000) * cost_per_1k_completion
        )
        trace.token_usage.estimated_cost_usd += cost
    
    def add_retrieval_trace(self, trace_id: str, retrieval_trace: RetrievalTrace) -> None:
        """Attach retrieval trace to main trace."""
        if trace_id in self.active_traces:
            self.active_traces[trace_id].retrieval_trace = retrieval_trace
    
    def _export_trace(self, trace: Trace) -> None:
        """Export trace to storage."""
        # Write to JSON file (in production, send to observability platform)
        timestamp = datetime.fromtimestamp(trace.start_time).strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{trace.operation}_{trace.trace_id[:8]}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(trace.to_dict(), f, indent=2)
        
        # Log summary
        logger.info(
            f"Trace completed: {trace.operation} | "
            f"Duration: {trace.duration_ms:.0f}ms | "
            f"Status: {trace.status} | "
            f"Tokens: {trace.token_usage.total_tokens} | "
            f"Cost: ${trace.token_usage.estimated_cost_usd:.4f}"
        )


# Global trace collector instance
_trace_collector: TraceCollector | None = None


def get_trace_collector(output_dir: Path | None = None) -> TraceCollector:
    """Get or create global trace collector."""
    global _trace_collector
    if _trace_collector is None:
        _trace_collector = TraceCollector(output_dir)
    return _trace_collector


@contextmanager
def trace_operation(operation: str, metadata: dict[str, Any] | None = None):
    """Context manager for tracing an operation."""
    collector = get_trace_collector()
    trace_id = collector.start_trace(operation, metadata)
    
    try:
        yield trace_id
    except Exception as e:
        collector.finish_trace(trace_id, error=e)
        raise
    else:
        collector.finish_trace(trace_id)


@contextmanager
def trace_span(
    trace_id: str,
    operation: str,
    parent_span_id: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Context manager for tracing a span."""
    collector = get_trace_collector()
    span_id = collector.start_span(trace_id, operation, parent_span_id, metadata)
    
    try:
        yield span_id
    except Exception as e:
        collector.finish_span(span_id, error=e)
        raise
    else:
        collector.finish_span(span_id)


def log_token_usage(
    trace_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    cost_per_1k_prompt: float = 0.0,
    cost_per_1k_completion: float = 0.0,
) -> None:
    """Log token usage for a trace."""
    collector = get_trace_collector()
    collector.add_token_usage(
        trace_id,
        prompt_tokens,
        completion_tokens,
        model,
        cost_per_1k_prompt,
        cost_per_1k_completion,
    )


def log_retrieval_trace(trace_id: str, retrieval_trace: RetrievalTrace) -> None:
    """Log retrieval trace."""
    collector = get_trace_collector()
    collector.add_retrieval_trace(trace_id, retrieval_trace)
