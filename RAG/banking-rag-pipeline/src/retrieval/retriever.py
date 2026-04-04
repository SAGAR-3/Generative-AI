"""
src/retrieval/retriever.py
===========================
Hybrid retrieval engine combining dense vector search with BM25 sparse search.

Strategy: Reciprocal Rank Fusion (RRF) for score combination.

Why Hybrid?
- Dense (semantic): catches "what documents are about FHA requirements" even
  without exact keyword match
- Sparse (BM25): catches exact terms like "FICO 580", "3.5% down payment",
  "debt-to-income 43%", "Reg Z", "APR"
- Combined: best of both worlds for banking queries

Performance:
- Dense retrieval: ~50ms (Qdrant HNSW)
- BM25 retrieval: ~5ms (in-memory)
- Reranking: ~200ms (Cohere API)
- Total P95: < 500ms retrieval, < 3s end-to-end
"""

import asyncio
import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

from src.embeddings.vector_store import QdrantVectorStore, SearchResult

logger = structlog.get_logger(__name__)


# ─── BM25 Index ───────────────────────────────────────────────────────────────

class BM25Index:
    """
    In-memory BM25 index for sparse keyword retrieval.
    Built from the same chunks stored in Qdrant.
    Rebuilt on startup or when new documents are ingested.
    
    BM25 parameters:
    - k1=1.5: Term frequency saturation
    - b=0.75: Length normalization factor
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[str] = []
        self.chunk_ids: List[str] = []
        self.metadatas: List[Dict] = []
        self._bm25 = None
        self._built = False

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index from chunk dictionaries.
        
        Args:
            chunks: List of {"chunk_id": ..., "content": ..., metadata...}
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 required: pip install rank-bm25")

        self.corpus = []
        self.chunk_ids = []
        self.metadatas = []

        for chunk in chunks:
            content = chunk.get("content", "")
            # Simple whitespace tokenization + lowercase
            # In production, use NLTK/spaCy with banking-specific stopwords
            tokens = content.lower().split()
            self.corpus.append(tokens)
            self.chunk_ids.append(chunk.get("chunk_id", ""))
            self.metadatas.append({k: v for k, v in chunk.items() if k != "content"})

        self._bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
        self._built = True

        logger.info("bm25_index_built", corpus_size=len(self.corpus))

    def search(
        self,
        query: str,
        top_k: int = 10,
        access_levels: Optional[List[str]] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """
        BM25 keyword search.

        Returns:
            List of (chunk_id, bm25_score, metadata) sorted by score descending
        """
        if not self._built:
            logger.warning("bm25_not_built", query=query)
            return []

        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)

        # Filter by access level
        results = []
        for i, (chunk_id, score) in enumerate(zip(self.chunk_ids, scores)):
            if score <= 0:
                continue
            if access_levels and self.metadatas[i].get("access_level") not in access_levels:
                continue
            results.append((chunk_id, float(score), self.metadatas[i]))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ─── Reciprocal Rank Fusion ───────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Formula: RRF(d) = Σ weight_i / (k + rank_i(d))

    Args:
        ranked_lists: List of [(id, score), ...] sorted by score desc
        k: Constant preventing dominance of high-rank items (typically 60)
        weights: Optional weight per list (default: equal weights)

    Returns:
        Dict of {id: rrf_score}
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    rrf_scores: Dict[str, float] = defaultdict(float)

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (item_id, _score) in enumerate(ranked_list, start=1):
            rrf_scores[item_id] += weight / (k + rank)

    return dict(rrf_scores)


# ─── Hybrid Retriever ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Production hybrid retriever combining:
    1. Dense vector search (Qdrant HNSW)
    2. Sparse keyword search (BM25)
    3. Score fusion via Reciprocal Rank Fusion

    RBAC-aware: enforces access level filtering at the retrieval layer.
    """

    # Weight: how much to trust dense vs sparse retrieval
    DENSE_WEIGHT = 0.7
    SPARSE_WEIGHT = 0.3

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedder,
        bm25_index: Optional[BM25Index] = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_index = bm25_index or BM25Index()

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        access_levels: Optional[List[str]] = None,
        document_categories: Optional[List[str]] = None,
        regulatory_tags: Optional[List[str]] = None,
        min_score: float = 0.0,
        use_hybrid: bool = True,
    ) -> List[SearchResult]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: User's question
            top_k: Number of chunks to return
            access_levels: SECURITY - Only return chunks at these access levels
            document_categories: Filter by document category
            regulatory_tags: Filter by regulatory category
            min_score: Minimum relevance score (0.0 - 1.0)
            use_hybrid: If False, use dense-only retrieval

        Returns:
            List of SearchResult sorted by relevance score
        """
        start_time = time.time()

        # Step 1: Embed query
        embed_start = time.time()
        query_embedding = await self.embedder.embed_text(query)
        embed_latency = (time.time() - embed_start) * 1000

        # Step 2: Dense vector search
        dense_start = time.time()
        dense_results = await self.vector_store.search(
            query_vector=query_embedding.vector,
            top_k=top_k * 2,  # Over-fetch for fusion
            access_levels=access_levels,
            document_categories=document_categories,
            regulatory_tags=regulatory_tags,
            score_threshold=0.0,  # Apply threshold after fusion
        )
        dense_latency = (time.time() - dense_start) * 1000

        if not use_hybrid or not self.bm25_index._built:
            # Dense-only path
            results = dense_results[:top_k]
        else:
            # Step 3: BM25 sparse search
            bm25_start = time.time()
            bm25_raw = self.bm25_index.search(
                query=query,
                top_k=top_k * 2,
                access_levels=access_levels,
            )
            bm25_latency = (time.time() - bm25_start) * 1000

            # Step 4: Reciprocal Rank Fusion
            dense_ranked = [(r.chunk_id, r.score) for r in dense_results]
            bm25_ranked = [(cid, score) for cid, score, _ in bm25_raw]

            rrf_scores = reciprocal_rank_fusion(
                ranked_lists=[dense_ranked, bm25_ranked],
                weights=[self.DENSE_WEIGHT, self.SPARSE_WEIGHT],
            )

            # Step 5: Build result list from dense results (they have full content)
            dense_map = {r.chunk_id: r for r in dense_results}
            bm25_map = {cid: (score, meta) for cid, score, meta in bm25_raw}

            fused_results = []
            for chunk_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
                if chunk_id in dense_map:
                    result = dense_map[chunk_id]
                    result.score = rrf_score
                    fused_results.append(result)
                # Note: BM25-only results would need content from a doc store
                # In production, maintain a content cache alongside BM25

            results = fused_results[:top_k]

        # Apply minimum score threshold
        results = [r for r in results if r.score >= min_score]

        total_latency = (time.time() - start_time) * 1000
        logger.info(
            "retrieval_complete",
            query_preview=query[:50],
            results=len(results),
            embed_ms=round(embed_latency, 1),
            dense_ms=round(dense_latency, 1),
            total_ms=round(total_latency, 1),
            hybrid=use_hybrid,
        )

        return results

    async def retrieve_with_metadata(
        self,
        query: str,
        user_role: str,
        **kwargs,
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Retrieve with role-based access control.

        Banking RBAC levels:
        - customer: public only
        - loan_officer: public + internal
        - underwriter: public + internal + confidential
        - compliance_officer: all
        - admin: all
        """
        ROLE_ACCESS_MAP = {
            "customer": ["public"],
            "loan_officer": ["public", "internal"],
            "underwriter": ["public", "internal", "confidential"],
            "compliance_officer": ["public", "internal", "confidential", "restricted"],
            "admin": ["public", "internal", "confidential", "restricted"],
        }

        access_levels = ROLE_ACCESS_MAP.get(user_role, ["public"])

        results = await self.retrieve(
            query=query,
            access_levels=access_levels,
            **kwargs,
        )

        retrieval_metadata = {
            "user_role": user_role,
            "access_levels": access_levels,
            "num_results": len(results),
        }

        return results, retrieval_metadata
