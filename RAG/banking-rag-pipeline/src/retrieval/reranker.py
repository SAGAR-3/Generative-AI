"""
src/retrieval/reranker.py
==========================
Cross-encoder reranker for improving retrieval precision.

Why reranking?
- Initial retrieval (HNSW + BM25) optimizes for recall
- Reranking optimizes for precision using a more expensive
  but accurate model that jointly scores (query, document) pairs
- Cross-encoders outperform bi-encoders for relevance ranking

Supported rerankers:
1. Cohere Rerank API (recommended for production)
2. Local cross-encoder (ms-marco-MiniLM-L-6-v2)
"""

import asyncio
import time
from typing import List, Optional
import structlog

from src.embeddings.vector_store import SearchResult

logger = structlog.get_logger(__name__)


# ─── Cohere Reranker ──────────────────────────────────────────────────────────

class CohereReranker:
    """
    Cohere Rerank API-based reranker.
    
    Model: rerank-english-v3.0
    Max docs per call: 1000
    Latency: ~200ms for 10 docs
    Best for: Production, high accuracy
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0",
        top_n: int = 5,
    ):
        try:
            import cohere
            self.client = cohere.AsyncClient(api_key=api_key)
        except ImportError:
            raise ImportError("cohere required: pip install cohere")

        self.model = model
        self.top_n = top_n

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Rerank search results using Cohere Rerank API.

        Args:
            query: Original user query
            results: Initial search results from retriever
            top_n: Number of results to return (defaults to self.top_n)

        Returns:
            Reranked results with updated scores
        """
        if not results:
            return []

        n = top_n or self.top_n
        documents = [r.content for r in results]

        start = time.time()
        try:
            response = await self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_n=min(n, len(results)),
                return_documents=False,
            )

            # Rebuild results list in reranked order
            reranked = []
            for item in response.results:
                result = results[item.index]
                result.score = item.relevance_score  # Cohere relevance score 0-1
                reranked.append(result)

            latency = (time.time() - start) * 1000
            logger.info(
                "reranking_complete",
                original_count=len(results),
                reranked_count=len(reranked),
                latency_ms=round(latency, 1),
                model=self.model,
            )
            return reranked

        except Exception as e:
            logger.error("reranking_failed", error=str(e), fallback="returning_original")
            # Graceful fallback: return original results truncated
            return results[:n]


# ─── Local Cross-Encoder Reranker ─────────────────────────────────────────────

class LocalCrossEncoderReranker:
    """
    Local cross-encoder reranker using sentence-transformers.
    
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    No API cost, runs locally on CPU/GPU
    Slightly lower accuracy than Cohere for banking-specific text
    Good for: Development, cost-sensitive deployments
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
    ):
        try:
            from sentence_transformers import CrossEncoder
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("loading_cross_encoder", model=model_name, device=device)
            self.model = CrossEncoder(model_name, device=device)
        except ImportError:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")

        self.top_n = top_n

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: Optional[int] = None,
    ) -> List[SearchResult]:
        """Rerank using local cross-encoder model."""
        if not results:
            return []

        n = top_n or self.top_n
        pairs = [(query, r.content) for r in results]

        start = time.time()
        loop = asyncio.get_event_loop()

        # Run in thread pool (CPU-bound)
        scores = await loop.run_in_executor(
            None,
            lambda: self.model.predict(pairs)
        )

        # Apply scores and sort
        scored_results = list(zip(scores, results))
        scored_results.sort(key=lambda x: x[0], reverse=True)

        reranked = []
        for score, result in scored_results[:n]:
            result.score = float(score)
            reranked.append(result)

        latency = (time.time() - start) * 1000
        logger.info(
            "local_reranking_complete",
            count=len(reranked),
            latency_ms=round(latency, 1),
        )
        return reranked


# ─── Reranker Factory ─────────────────────────────────────────────────────────

class RerankerFactory:
    @staticmethod
    def create(provider: str = "cohere", **kwargs):
        if provider == "cohere":
            api_key = kwargs.get("api_key") or __import__("os").getenv("COHERE_API_KEY")
            if not api_key:
                logger.warning("cohere_key_missing_using_local_reranker")
                return LocalCrossEncoderReranker(top_n=kwargs.get("top_n", 5))
            return CohereReranker(
                api_key=api_key,
                model=kwargs.get("model", "rerank-english-v3.0"),
                top_n=kwargs.get("top_n", 5),
            )
        elif provider == "local":
            return LocalCrossEncoderReranker(
                model_name=kwargs.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                top_n=kwargs.get("top_n", 5),
            )
        else:
            raise ValueError(f"Unknown reranker provider: {provider}")
