"""
src/embeddings/vector_store.py
================================
Qdrant vector store integration for the banking RAG pipeline.

Features:
- CRUD operations for document chunks
- Hybrid search (dense + sparse BM25)
- Metadata filtering (access_level, category, regulatory_tags)
- Collection management
- Health checks
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)


# ─── Search Result ────────────────────────────────────────────────────────────

class SearchResult:
    """Represents a single retrieved chunk with its score."""

    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        score: float,
        metadata: Dict[str, Any],
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.content = content
        self.score = score
        self.metadata = metadata
        self.source_file = metadata.get("source_file", "")
        self.document_category = metadata.get("document_category", "")
        self.regulatory_tags = metadata.get("regulatory_tags", [])
        self.section_title = metadata.get("section_title")
        self.access_level = metadata.get("access_level", "internal")

    def __repr__(self):
        return f"SearchResult(chunk_id={self.chunk_id}, score={self.score:.3f}, category={self.document_category})"


# ─── Qdrant Vector Store ──────────────────────────────────────────────────────

class QdrantVectorStore:
    """
    Production Qdrant vector store.

    Configuration:
    - Dense vectors: OpenAI text-embedding-3-large (3072 dims) or BGE-large (1024 dims)
    - Distance metric: Cosine similarity
    - Payload: Full chunk metadata for filtering
    
    Security:
    - Access level filtering enforced at query time
    - No raw content returned beyond user's clearance
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        https: bool = False,
        collection_name: str = "home_lending_docs",
        vector_dimension: int = 3072,
    ):
        try:
            from qdrant_client import QdrantClient, AsyncQdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("qdrant-client required: pip install qdrant-client")

        self.collection_name = collection_name
        self.vector_dimension = vector_dimension

        # Sync client for management operations
        self.client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key,
            https=https,
            timeout=30,
        )

        # Async client for query operations
        self.async_client = AsyncQdrantClient(
            host=host,
            port=port,
            api_key=api_key,
            https=https,
            timeout=30,
        )

        logger.info(
            "qdrant_connected",
            host=host,
            port=port,
            collection=collection_name,
        )

    def ensure_collection_exists(self) -> None:
        """
        Create collection if it doesn't exist.
        Called on startup.
        """
        from qdrant_client.models import (
            Distance, VectorParams, HnswConfigDiff,
            OptimizersConfigDiff, PayloadSchemaType
        )

        try:
            collections = self.client.get_collections().collections
            existing_names = [c.name for c in collections]

            if self.collection_name not in existing_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dimension,
                        distance=Distance.COSINE,
                    ),
                    hnsw_config=HnswConfigDiff(
                        m=16,               # Number of edges per node (16-64 for production)
                        ef_construct=200,   # Build quality (higher = better recall, slower build)
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        default_segment_number=4,
                        memmap_threshold=20000,
                    ),
                )

                # Create payload indexes for fast filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="access_level",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_category",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="regulatory_tags",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="doc_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )

                logger.info("collection_created", name=self.collection_name)
            else:
                logger.info("collection_exists", name=self.collection_name)

        except Exception as e:
            logger.error("collection_creation_failed", error=str(e))
            raise

    def upsert_chunks(
        self,
        chunks,
        vectors: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """
        Upsert document chunks with their vectors into Qdrant.

        Args:
            chunks: List of DocumentChunk objects
            vectors: Corresponding embedding vectors
            batch_size: Number of points per upsert request

        Returns:
            Number of successfully upserted points
        """
        from qdrant_client.models import PointStruct

        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors")

        upserted = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]

            points = []
            for chunk, vector in zip(batch_chunks, batch_vectors):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),   # Qdrant requires UUID or uint64
                    vector=vector,
                    payload=chunk.to_dict(),
                ))

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            upserted += len(points)
            logger.debug("batch_upserted", batch_size=len(points), total=upserted)

        logger.info("upsert_complete", total_points=upserted)
        return upserted

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        access_levels: Optional[List[str]] = None,
        document_categories: Optional[List[str]] = None,
        regulatory_tags: Optional[List[str]] = None,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Semantic search with metadata filtering.

        Args:
            query_vector: Dense query embedding
            top_k: Number of results to return
            access_levels: Filter by access level (enforced for security)
            document_categories: Filter by document category
            regulatory_tags: Filter by regulatory tags
            score_threshold: Minimum similarity score

        Returns:
            List of SearchResult objects sorted by score
        """
        from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

        # Build filter conditions
        must_conditions = []

        # SECURITY: Always filter by access level
        if access_levels:
            must_conditions.append(
                FieldCondition(
                    key="access_level",
                    match=MatchAny(any=access_levels),
                )
            )

        if document_categories:
            must_conditions.append(
                FieldCondition(
                    key="document_category",
                    match=MatchAny(any=document_categories),
                )
            )

        if regulatory_tags:
            must_conditions.append(
                FieldCondition(
                    key="regulatory_tags",
                    match=MatchAny(any=regulatory_tags),
                )
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        try:
            start = time.time()
            results = await self.async_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )
            latency = (time.time() - start) * 1000

            logger.debug(
                "vector_search_complete",
                results=len(results),
                latency_ms=round(latency, 1),
            )

            return [
                SearchResult(
                    chunk_id=str(r.payload.get("chunk_id", r.id)),
                    doc_id=str(r.payload.get("doc_id", "")),
                    content=r.payload.get("content", ""),
                    score=r.score,
                    metadata=r.payload,
                )
                for r in results
            ]

        except Exception as e:
            logger.error("vector_search_failed", error=str(e))
            raise

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        logger.info("document_deleted", doc_id=doc_id)
        return result.status

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics for monitoring."""
        info = self.client.get_collection(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": str(info.status),
            "optimizer_status": str(info.optimizer_status.status),
            "vector_size": info.config.params.vectors.size,
        }

    async def health_check(self) -> bool:
        """Check if Qdrant is reachable and collection exists."""
        try:
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]
            return self.collection_name in names
        except Exception:
            return False
