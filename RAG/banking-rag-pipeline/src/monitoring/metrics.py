"""
src/monitoring/metrics.py
==========================
Prometheus metrics instrumentation for the banking RAG pipeline.

Metrics tracked:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LATENCY METRICS:
  bankrag_query_duration_seconds      - End-to-end query latency
  bankrag_retrieval_duration_seconds  - Retrieval phase latency
  bankrag_generation_duration_seconds - LLM generation latency
  bankrag_embedding_duration_seconds  - Embedding latency

QUALITY METRICS:
  bankrag_retrieval_score             - Average retrieval relevance scores
  bankrag_faithfulness_score          - RAGAS faithfulness
  bankrag_answer_relevancy_score      - RAGAS answer relevancy
  bankrag_hallucination_detected      - Hallucination detection events

SECURITY METRICS:
  bankrag_pii_detections_total        - PII detection counts by type
  bankrag_compliance_violations_total - Compliance violations by type
  bankrag_blocked_queries_total       - Blocked queries by reason
  bankrag_auth_failures_total         - Auth failures

BUSINESS METRICS:
  bankrag_queries_total               - Total queries by role/category
  bankrag_token_usage_total           - Token consumption by model
  bankrag_cost_usd_total              - API cost tracking
  bankrag_documents_ingested_total    - Document ingestion count
  bankrag_cache_hits_total            - Cache performance
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


# ─── Metrics Registry ─────────────────────────────────────────────────────────

class BankRAGMetrics:
    """
    Central metrics registry using Prometheus client.
    Singleton pattern - one instance per application.
    """

    def __init__(self):
        self._initialized = False
        self._init_metrics()

    def _init_metrics(self):
        try:
            from prometheus_client import (
                Counter, Histogram, Gauge, Summary,
                REGISTRY, CollectorRegistry
            )
        except ImportError:
            raise ImportError("prometheus-client required: pip install prometheus-client")

        # ── Latency Histograms ──────────────────────────────────────────────
        self.query_duration = Histogram(
            "bankrag_query_duration_seconds",
            "End-to-end query processing duration",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
            labelnames=["user_role", "query_category", "cache_hit"],
        )

        self.retrieval_duration = Histogram(
            "bankrag_retrieval_duration_seconds",
            "Document retrieval duration (embedding + vector search + rerank)",
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            labelnames=["retrieval_type"],  # dense | hybrid
        )

        self.generation_duration = Histogram(
            "bankrag_generation_duration_seconds",
            "LLM generation duration",
            buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0],
            labelnames=["model", "user_role"],
        )

        self.embedding_duration = Histogram(
            "bankrag_embedding_duration_seconds",
            "Query embedding duration",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5],
            labelnames=["model"],
        )

        self.guardrails_duration = Histogram(
            "bankrag_guardrails_duration_seconds",
            "Guardrails processing duration (PII + compliance)",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
            labelnames=["check_type"],  # pii | compliance | output_validation
        )

        # ── Quality Metrics ─────────────────────────────────────────────────
        self.retrieval_score = Histogram(
            "bankrag_retrieval_score",
            "Distribution of retrieval relevance scores",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            labelnames=["user_role"],
        )

        self.faithfulness_score = Histogram(
            "bankrag_faithfulness_score",
            "RAGAS faithfulness score (answer grounded in context)",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        self.answer_relevancy_score = Histogram(
            "bankrag_answer_relevancy_score",
            "RAGAS answer relevancy score",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        self.context_precision_score = Histogram(
            "bankrag_context_precision_score",
            "RAGAS context precision score",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        self.context_recall_score = Histogram(
            "bankrag_context_recall_score",
            "RAGAS context recall score",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        self.hallucination_detected = Counter(
            "bankrag_hallucination_detected_total",
            "Number of potential hallucinations detected",
            labelnames=["model", "user_role"],
        )

        # ── Security Counters ───────────────────────────────────────────────
        self.pii_detections = Counter(
            "bankrag_pii_detections_total",
            "PII detected in queries or responses",
            labelnames=["pii_type", "location"],  # location: query | response
        )

        self.compliance_violations = Counter(
            "bankrag_compliance_violations_total",
            "Regulatory compliance violations detected",
            labelnames=["violation_type", "severity", "regulation"],
        )

        self.blocked_queries = Counter(
            "bankrag_blocked_queries_total",
            "Queries blocked by guardrails",
            labelnames=["block_reason"],  # pii | compliance | rate_limit | auth
        )

        self.auth_failures = Counter(
            "bankrag_auth_failures_total",
            "Authentication failure events",
            labelnames=["failure_type"],  # invalid_token | expired | revoked | wrong_role
        )

        self.rate_limit_exceeded = Counter(
            "bankrag_rate_limit_exceeded_total",
            "Rate limit exceeded events",
            labelnames=["user_role"],
        )

        # ── Business Metrics ────────────────────────────────────────────────
        self.queries_total = Counter(
            "bankrag_queries_total",
            "Total queries processed",
            labelnames=["user_role", "document_category", "success"],
        )

        self.token_usage = Counter(
            "bankrag_token_usage_total",
            "Total LLM tokens consumed",
            labelnames=["model", "token_type"],  # token_type: input | output
        )

        self.api_cost = Counter(
            "bankrag_cost_usd_total",
            "Total API cost in USD",
            labelnames=["model", "operation"],  # operation: embedding | generation
        )

        self.documents_ingested = Counter(
            "bankrag_documents_ingested_total",
            "Documents ingested into vector store",
            labelnames=["file_type", "document_category"],
        )

        self.cache_hits = Counter(
            "bankrag_cache_hits_total",
            "Query cache hit/miss events",
            labelnames=["cache_type", "result"],  # result: hit | miss
        )

        # ── Gauges (current state) ──────────────────────────────────────────
        self.active_queries = Gauge(
            "bankrag_active_queries",
            "Number of queries currently being processed",
        )

        self.vector_store_documents = Gauge(
            "bankrag_vector_store_documents_total",
            "Total documents in vector store",
            labelnames=["collection"],
        )

        self.vector_store_chunks = Gauge(
            "bankrag_vector_store_chunks_total",
            "Total chunks in vector store",
            labelnames=["collection"],
        )

        self.bm25_index_size = Gauge(
            "bankrag_bm25_index_size",
            "Number of documents in BM25 index",
        )

        self._initialized = True
        logger.info("metrics_initialized")

    # ── Convenience Context Managers ─────────────────────────────────────────

    @contextmanager
    def track_query(self, user_role: str, query_category: str, cache_hit: bool):
        """Context manager to track end-to-end query duration."""
        self.active_queries.inc()
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.query_duration.labels(
                user_role=user_role,
                query_category=query_category,
                cache_hit=str(cache_hit).lower(),
            ).observe(duration)
            self.active_queries.dec()

    @contextmanager
    def track_retrieval(self, retrieval_type: str = "hybrid"):
        """Context manager to track retrieval duration."""
        start = time.time()
        try:
            yield
        finally:
            self.retrieval_duration.labels(retrieval_type=retrieval_type).observe(time.time() - start)

    @contextmanager
    def track_generation(self, model: str, user_role: str):
        """Context manager to track LLM generation duration."""
        start = time.time()
        try:
            yield
        finally:
            self.generation_duration.labels(model=model, user_role=user_role).observe(time.time() - start)

    def record_query_result(
        self,
        user_role: str,
        document_category: str,
        success: bool,
        tokens_input: int,
        tokens_output: int,
        model: str,
        cost_usd: float,
        retrieval_scores: list,
        cache_hit: bool = False,
    ):
        """Record all metrics for a completed query."""
        self.queries_total.labels(
            user_role=user_role,
            document_category=document_category,
            success=str(success).lower(),
        ).inc()

        self.token_usage.labels(model=model, token_type="input").inc(tokens_input)
        self.token_usage.labels(model=model, token_type="output").inc(tokens_output)
        self.api_cost.labels(model=model, operation="generation").inc(cost_usd)
        self.cache_hits.labels(cache_type="query", result="hit" if cache_hit else "miss").inc()

        for score in retrieval_scores:
            self.retrieval_score.labels(user_role=user_role).observe(score)

    def record_security_event(
        self,
        pii_types: list = None,
        compliance_violations: list = None,
        blocked: bool = False,
        block_reason: str = None,
    ):
        """Record security-related metrics."""
        if pii_types:
            for pii_type in pii_types:
                self.pii_detections.labels(pii_type=pii_type, location="query").inc()

        if compliance_violations:
            for violation in compliance_violations:
                self.compliance_violations.labels(
                    violation_type=violation,
                    severity="high",
                    regulation="mixed",
                ).inc()

        if blocked and block_reason:
            self.blocked_queries.labels(block_reason=block_reason).inc()


# ─── Singleton Instance ───────────────────────────────────────────────────────

_metrics_instance: Optional[BankRAGMetrics] = None


def get_metrics() -> BankRAGMetrics:
    """Get or create the global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = BankRAGMetrics()
    return _metrics_instance
