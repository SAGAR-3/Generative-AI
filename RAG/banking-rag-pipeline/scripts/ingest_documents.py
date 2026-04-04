"""
scripts/ingest_documents.py
============================
CLI script to batch-ingest banking documents into the RAG pipeline.

Usage:
    python scripts/ingest_documents.py --source data/sample/ --collection home_lending
    python scripts/ingest_documents.py --source docs/compliance/ --access-level confidential
    python scripts/ingest_documents.py --file docs/fha_guidelines.pdf --category loan_product

Features:
- Progress tracking
- Duplicate detection (SHA-256)
- Batch embedding with rate limit handling
- Detailed ingestion report
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from dotenv import load_dotenv
load_dotenv()

logger = structlog.get_logger(__name__)


async def ingest_documents(
    source: str,
    collection: str = "home_lending_docs",
    access_level: str = "internal",
    category_override: str = None,
    batch_size: int = 50,
    dry_run: bool = False,
):
    """
    Main ingestion pipeline.

    Args:
        source: File or directory path
        collection: Qdrant collection name
        access_level: Document security classification
        category_override: Override auto-detected category
        batch_size: Embedding batch size
        dry_run: If True, process but don't store
    """
    from config.settings import get_settings
    from src.ingestion.document_loader import DocumentLoader
    from src.ingestion.chunker import DocumentChunker
    from src.embeddings.embedder import EmbedderFactory, DocumentEmbedder
    from src.embeddings.vector_store import QdrantVectorStore
    from src.retrieval.retriever import BM25Index

    settings = get_settings()
    start_time = time.time()

    print(f"""
╔══════════════════════════════════════════════════════╗
║      BankRAG Document Ingestion Pipeline             ║
╚══════════════════════════════════════════════════════╝
  Source:       {source}
  Collection:   {collection}
  Access Level: {access_level}
  Dry Run:      {dry_run}
  Chunk Size:   {settings.ingestion.chunk_size}
  Batch Size:   {batch_size}
""")

    # Initialize components
    print("🔧 Initializing components...")
    loader = DocumentLoader()
    chunker = DocumentChunker(
        chunk_size=settings.ingestion.chunk_size,
        chunk_overlap=settings.ingestion.chunk_overlap,
    )

    embedder = EmbedderFactory.create(
        provider=settings.embedding.provider,
        api_key=getattr(settings.llm, f"{settings.embedding.provider}_api_key", None),
        model=settings.embedding.model,
    )
    doc_embedder = DocumentEmbedder(embedder, batch_size=batch_size)

    if not dry_run:
        vector_store = QdrantVectorStore(
            host=settings.vector_store.host,
            port=settings.vector_store.port,
            collection_name=collection,
            vector_dimension=settings.embedding.dimension,
        )
        vector_store.ensure_collection_exists()

    # Load documents
    print(f"\n📄 Loading documents from: {source}")
    source_path = Path(source)
    documents = []

    if source_path.is_file():
        try:
            doc = loader.load_file(source_path, access_level=access_level)
            documents.append(doc)
        except Exception as e:
            print(f"  ❌ Failed to load {source_path.name}: {e}")
    elif source_path.is_dir():
        for doc in loader.load_directory(source_path, access_level=access_level):
            documents.append(doc)
    else:
        print(f"❌ Source not found: {source}")
        return

    if not documents:
        print("❌ No documents loaded. Exiting.")
        return

    print(f"  ✅ Loaded {len(documents)} documents")

    # Chunk documents
    print(f"\n✂️  Chunking documents...")
    all_chunks = []
    for doc in documents:
        if category_override:
            doc.metadata.document_category = category_override
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  📃 {Path(doc.metadata.source_file).name}: {len(chunks)} chunks")

    print(f"\n  Total chunks: {len(all_chunks)}")
    avg_chunk_size = sum(len(c.content) for c in all_chunks) // max(len(all_chunks), 1)
    print(f"  Avg chunk size: {avg_chunk_size} chars")

    if dry_run:
        print("\n🔍 DRY RUN: Skipping embedding and storage.")
        print("  Sample chunk:")
        if all_chunks:
            print(f"    ID: {all_chunks[0].chunk_id}")
            print(f"    Category: {all_chunks[0].document_category}")
            print(f"    Preview: {all_chunks[0].content[:100]}...")
        return

    # Embed chunks
    print(f"\n🧠 Generating embeddings (model: {settings.embedding.model})...")
    embed_start = time.time()
    embedded_pairs = await doc_embedder.embed_chunks(all_chunks)
    embed_time = time.time() - embed_start

    chunk_objects = [pair[0] for pair in embedded_pairs]
    vectors = [pair[1] for pair in embedded_pairs]

    print(f"  ✅ Embedded {len(vectors)} chunks in {embed_time:.1f}s")
    print(f"  ⚡ {len(vectors)/max(embed_time, 0.1):.0f} chunks/second")
    print(f"  📐 Vector dimension: {len(vectors[0]) if vectors else 0}")

    # Store in vector database
    print(f"\n💾 Storing in Qdrant collection '{collection}'...")
    store_start = time.time()
    upserted = vector_store.upsert_chunks(chunk_objects, vectors)
    store_time = time.time() - store_start

    print(f"  ✅ Stored {upserted} chunks in {store_time:.1f}s")

    # Build BM25 index (for hybrid retrieval)
    print(f"\n🔍 Building BM25 sparse index...")
    bm25 = BM25Index()
    chunk_dicts = [c.to_dict() for c in chunk_objects]
    bm25.build(chunk_dicts)
    print(f"  ✅ BM25 index built with {len(chunk_dicts)} documents")
    print("  ⚠️  Note: BM25 index is in-memory. Save to disk for persistence.")

    # Final report
    total_time = time.time() - start_time
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INGESTION COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Documents:    {len(documents)}
  Chunks:       {len(all_chunks)}
  Stored:       {upserted}
  Total Time:   {total_time:.1f}s

  Collection Stats:
  {vector_store.get_collection_stats()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

    # Category breakdown
    from collections import Counter
    categories = Counter(c.document_category for c in chunk_objects)
    print("  Document categories:")
    for cat, count in categories.most_common():
        print(f"    {cat}: {count} chunks")

    return {
        "documents": len(documents),
        "chunks": len(all_chunks),
        "stored": upserted,
        "duration_s": total_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ingest banking documents into the RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest_documents.py --source data/sample/
  python scripts/ingest_documents.py --source docs/fha_guidelines.pdf --access-level internal
  python scripts/ingest_documents.py --source docs/compliance/ --access-level confidential --dry-run
        """,
    )
    parser.add_argument("--source", required=True, help="File or directory to ingest")
    parser.add_argument("--collection", default="home_lending_docs", help="Qdrant collection name")
    parser.add_argument(
        "--access-level",
        choices=["public", "internal", "confidential", "restricted"],
        default="internal",
        help="Security classification for all documents",
    )
    parser.add_argument("--category", default=None, help="Override document category")
    parser.add_argument("--batch-size", type=int, default=50, help="Embedding batch size")
    parser.add_argument("--dry-run", action="store_true", help="Process but don't store")

    args = parser.parse_args()

    asyncio.run(ingest_documents(
        source=args.source,
        collection=args.collection,
        access_level=args.access_level,
        category_override=args.category,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
