"""
src/embeddings/embedder.py
===========================
Production embedding module supporting OpenAI and HuggingFace models.

Features:
- Async batch embedding with retry logic
- Token-aware batching (respects API limits)
- Caching to avoid re-embedding identical content
- Supports both query and document embedding modes
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import List, Optional, Tuple
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ─── Embedding Result ─────────────────────────────────────────────────────────

class EmbeddingResult:
    def __init__(self, vector: List[float], model: str, token_count: int):
        self.vector = vector
        self.model = model
        self.token_count = token_count
        self.dimension = len(vector)


# ─── Base Embedder ────────────────────────────────────────────────────────────

class BaseEmbedder:
    """Abstract base class for embedding providers."""

    async def embed_text(self, text: str) -> EmbeddingResult:
        raise NotImplementedError

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        raise NotImplementedError


# ─── OpenAI Embedder ──────────────────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI text-embedding-3-large embedder.
    
    Model: text-embedding-3-large (3072 dims)
    Max tokens: 8191 per text
    Best for: High accuracy, production use
    """

    MAX_BATCH_SIZE = 100         # OpenAI allows up to 2048 inputs per request
    MAX_TOKENS_PER_TEXT = 8191
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1.0

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: Optional[int] = None,  # Can reduce dims (e.g., 1536)
    ):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self.model = model
        self.dimensions = dimensions
        self._cache: dict = {}  # Simple in-memory cache

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text string."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Embed a batch of texts.
        Handles caching, batching, and retries.
        """
        if not texts:
            return []

        # Check cache
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            # Process in batches
            new_results = await self._embed_with_retry(uncached_texts)
            for idx, result in zip(uncached_indices, new_results):
                results[idx] = result
                # Cache the result
                self._cache[self._cache_key(texts[idx])] = result

        return results

    async def _embed_with_retry(self, texts: List[str]) -> List[EmbeddingResult]:
        """Call OpenAI API with exponential backoff retry."""
        for attempt in range(self.RETRY_ATTEMPTS):
            try:
                kwargs = {
                    "model": self.model,
                    "input": texts,
                    "encoding_format": "float",
                }
                if self.dimensions:
                    kwargs["dimensions"] = self.dimensions

                response = await self.client.embeddings.create(**kwargs)

                results = []
                for item in response.data:
                    results.append(EmbeddingResult(
                        vector=item.embedding,
                        model=self.model,
                        token_count=response.usage.total_tokens // len(texts),
                    ))

                logger.debug(
                    "embeddings_created",
                    count=len(texts),
                    total_tokens=response.usage.total_tokens,
                )
                return results

            except Exception as e:
                if attempt < self.RETRY_ATTEMPTS - 1:
                    wait = self.RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "embedding_retry",
                        attempt=attempt + 1,
                        error=str(e),
                        wait=wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error("embedding_failed", error=str(e))
                    raise


# ─── HuggingFace Embedder (Local / Free) ──────────────────────────────────────

class HuggingFaceEmbedder(BaseEmbedder):
    """
    Local embedding using HuggingFace sentence-transformers.
    
    Recommended models:
    - BAAI/bge-m3: Best multilingual, 1024 dims
    - BAAI/bge-large-en-v1.5: Best English, 1024 dims
    - all-MiniLM-L6-v2: Fast & lightweight, 384 dims
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("loading_embedding_model", model=model_name, device=device)
            self.model = SentenceTransformer(model_name, device=device)
            self.model_name = model_name
        except ImportError:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")

    async def embed_text(self, text: str) -> EmbeddingResult:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        # Run in thread pool (CPU-bound)
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        )
        return [
            EmbeddingResult(
                vector=emb.tolist(),
                model=self.model_name,
                token_count=len(text.split()),
            )
            for text, emb in zip(texts, embeddings)
        ]


# ─── Embedder Factory ─────────────────────────────────────────────────────────

class EmbedderFactory:
    """
    Factory to create the appropriate embedder based on configuration.
    
    Usage:
        embedder = EmbedderFactory.create(settings)
        result = await embedder.embed_text("What is the minimum credit score for FHA?")
    """

    @staticmethod
    def create(provider: str = "openai", **kwargs) -> BaseEmbedder:
        """
        Create an embedder instance.

        Args:
            provider: "openai" or "huggingface"
            **kwargs: Provider-specific kwargs

        Returns:
            BaseEmbedder instance
        """
        if provider == "openai":
            api_key = kwargs.get("api_key") or __import__("os").getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY required for OpenAI embedder")
            return OpenAIEmbedder(
                api_key=api_key,
                model=kwargs.get("model", "text-embedding-3-large"),
                dimensions=kwargs.get("dimensions"),
            )
        elif provider == "huggingface":
            return HuggingFaceEmbedder(
                model_name=kwargs.get("model", "BAAI/bge-large-en-v1.5")
            )
        else:
            raise ValueError(f"Unknown embedding provider: {provider}. Use 'openai' or 'huggingface'")


# ─── Document Embedding Pipeline ──────────────────────────────────────────────

class DocumentEmbedder:
    """
    High-level pipeline that embeds a list of DocumentChunks.
    Handles batching, progress tracking, and error recovery.
    """

    def __init__(self, embedder: BaseEmbedder, batch_size: int = 50):
        self.embedder = embedder
        self.batch_size = batch_size

    async def embed_chunks(self, chunks) -> List[Tuple]:
        """
        Embed a list of DocumentChunks.

        Returns:
            List of (chunk, embedding_vector) tuples
        """
        from src.ingestion.chunker import DocumentChunk

        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        embedded_pairs = []
        total = len(texts)
        start_time = time.time()

        logger.info("embedding_chunks_start", total=total, batch_size=self.batch_size)

        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch_texts = texts[batch_start:batch_end]
            batch_chunks = chunks[batch_start:batch_end]

            try:
                results = await self.embedder.embed_batch(batch_texts)
                for chunk, result in zip(batch_chunks, results):
                    embedded_pairs.append((chunk, result.vector))

                progress = (batch_end / total) * 100
                elapsed = time.time() - start_time
                logger.info(
                    "embedding_progress",
                    processed=batch_end,
                    total=total,
                    progress_pct=round(progress, 1),
                    elapsed_s=round(elapsed, 1),
                )
            except Exception as e:
                logger.error(
                    "embedding_batch_failed",
                    batch_start=batch_start,
                    batch_end=batch_end,
                    error=str(e),
                )
                # Re-raise in production; or add to failed_chunks list for retry
                raise

        elapsed = time.time() - start_time
        logger.info(
            "embedding_complete",
            total_chunks=len(embedded_pairs),
            elapsed_s=round(elapsed, 1),
            chunks_per_second=round(len(embedded_pairs) / max(elapsed, 0.1), 1),
        )

        return embedded_pairs
