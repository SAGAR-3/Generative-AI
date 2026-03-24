"""
Vector Store Service
Uses ChromaDB (local, no infra needed) for storing and retrieving BRD chunk embeddings.
Swap client for Pinecone / Weaviate in production.
"""
from __future__ import annotations
import hashlib
import os

# ── ChromaDB client (local persistent) ───────────────────────────────────────
try:
    import chromadb
    from chromadb.utils import embedding_functions

    _client = chromadb.PersistentClient(path="./chroma_db")

    # Use OpenAI embeddings if key present, else fall back to sentence-transformers
    if os.getenv("OPENAI_API_KEY"):
        _ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )
    else:
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


def _get_collection(name: str = "brd_chunks"):
    if not CHROMA_AVAILABLE:
        raise RuntimeError("chromadb not installed. Run: pip install chromadb")
    return _client.get_or_create_collection(name=name, embedding_function=_ef)


# ── Public API ────────────────────────────────────────────────────────────────

def store_chunks(document_id: str, chunks: list[str]) -> None:
    """Embed and persist BRD chunks into the vector store."""
    collection = _get_collection()

    ids       = [f"{document_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"document_id": document_id, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )


def retrieve_relevant_chunks(
    document_id: str,
    query: str,
    n_results: int = 6,
) -> list[str]:
    """
    Hybrid retrieval: semantic similarity filtered to a specific document.
    Returns the top-n most relevant chunks.
    """
    collection = _get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"document_id": {"$eq": document_id}},
    )

    docs = results.get("documents", [[]])[0]
    return docs


def delete_document_chunks(document_id: str) -> None:
    """Remove all chunks belonging to a document."""
    collection = _get_collection()
    collection.delete(where={"document_id": {"$eq": document_id}})
