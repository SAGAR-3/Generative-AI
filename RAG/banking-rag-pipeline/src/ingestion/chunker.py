"""
src/ingestion/chunker.py
========================
Banking-aware document chunker.

Strategy:
- Semantic chunking (respects sentence and paragraph boundaries)
- Banking-specific section detection (rates, guidelines, compliance)
- Metadata preservation per chunk (source, page, section, regulatory tags)
- Overlap to preserve cross-chunk context
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
import structlog

from src.ingestion.document_loader import LoadedDocument, DocumentMetadata

logger = structlog.get_logger(__name__)


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """
    A single chunk of a document, ready for embedding.
    Each chunk carries full provenance metadata.
    """
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    total_chunks: int

    # Provenance
    source_file: str
    document_category: str
    regulatory_tags: List[str]
    access_level: str

    # Position info
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    char_start: int = 0
    char_end: int = 0

    # Token estimate (approx: 1 token ≈ 4 chars)
    token_estimate: int = 0

    # Extra metadata (preserved from parent doc)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for vector store payload."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "source_file": self.source_file,
            "document_category": self.document_category,
            "regulatory_tags": self.regulatory_tags,
            "access_level": self.access_level,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_estimate": self.token_estimate,
        }


# ─── Section Header Detection ─────────────────────────────────────────────────

# Banking document section patterns
SECTION_PATTERNS = [
    # Numbered sections: "1. Eligibility Requirements"
    re.compile(r"^(\d+\.[\d\.]*)\s+([A-Z][^\n]{3,80})$", re.MULTILINE),
    # ALL CAPS headers
    re.compile(r"^([A-Z][A-Z\s]{4,60})$", re.MULTILINE),
    # Title case headers with colon
    re.compile(r"^([A-Z][a-zA-Z\s]{4,60}):$", re.MULTILINE),
    # Markdown-style headers
    re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE),
    # Page markers
    re.compile(r"^\[Page \d+\]$", re.MULTILINE),
]

def detect_section(text: str) -> Optional[str]:
    """Detect the section title at the start of a text block."""
    first_lines = text.strip()[:200]
    for pattern in SECTION_PATTERNS:
        match = pattern.search(first_lines)
        if match:
            # Return the last capturing group (the title part)
            groups = [g for g in match.groups() if g]
            return groups[-1].strip() if groups else None
    return None


# ─── Smart Text Splitter ──────────────────────────────────────────────────────

class BankingTextSplitter:
    """
    Banking-aware recursive text splitter.

    Split order:
    1. Section headers (preserve document structure)
    2. Double newlines (paragraphs)
    3. Single newlines
    4. Sentence boundaries (. ? !)
    5. Comma/semicolon
    6. Hard character split (last resort)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        length_function=len,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

        # Ordered separators
        self.separators = [
            "\n\n\n",   # Large section breaks
            "\n\n",     # Paragraph breaks
            "\n",       # Line breaks
            ". ",       # Sentence end
            "? ",       # Question end
            "! ",       # Exclamation end
            "; ",       # Semicolon
            ", ",       # Comma
            " ",        # Word boundary
            "",         # Character level (last resort)
        ]

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks respecting sentence/paragraph boundaries."""
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the best available separator."""
        if not text.strip():
            return []

        if self.length_function(text) <= self.chunk_size:
            return [text.strip()]

        # Find the best separator for this text
        separator = separators[-1]  # fallback
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split using the chosen separator
        splits = text.split(separator) if separator else list(text)
        splits = [s for s in splits if s.strip()]

        # Merge splits into chunks with overlap
        return self._merge_splits(splits, separator, new_separators)

    def _merge_splits(
        self,
        splits: List[str],
        separator: str,
        new_separators: List[str],
    ) -> List[str]:
        """Merge short splits into chunks, recursively splitting long ones."""
        chunks = []
        current_chunk_parts: List[str] = []
        current_length = 0

        for split in splits:
            split_length = self.length_function(split)

            if current_length + split_length + len(separator) > self.chunk_size:
                # Flush current chunk
                if current_chunk_parts:
                    chunk_text = separator.join(current_chunk_parts).strip()
                    if chunk_text:
                        chunks.append(chunk_text)

                    # Apply overlap: keep last N chars
                    if self.chunk_overlap > 0:
                        overlap_text = chunk_text[-self.chunk_overlap:]
                        current_chunk_parts = [overlap_text]
                        current_length = self.length_function(overlap_text)
                    else:
                        current_chunk_parts = []
                        current_length = 0

                # If split itself is too long, recurse
                if split_length > self.chunk_size and new_separators:
                    sub_chunks = self._split_recursive(split, new_separators)
                    # Add all but last as complete chunks
                    for sub in sub_chunks[:-1]:
                        chunks.append(sub)
                    # Last sub-chunk starts new accumulation
                    if sub_chunks:
                        current_chunk_parts = [sub_chunks[-1]]
                        current_length = self.length_function(sub_chunks[-1])
                    continue

            current_chunk_parts.append(split)
            current_length += split_length + len(separator)

        # Flush remaining
        if current_chunk_parts:
            chunk_text = separator.join(current_chunk_parts).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return [c for c in chunks if c.strip()]


# ─── Main Chunker ─────────────────────────────────────────────────────────────

class DocumentChunker:
    """
    Production document chunker for banking RAG pipeline.

    Converts LoadedDocuments → List[DocumentChunk] with:
    - Smart splitting preserving banking document structure
    - Section header detection and attribution
    - Full provenance metadata on every chunk
    - Deduplication of empty/trivial chunks
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_length: int = 50,
    ):
        self.splitter = BankingTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.min_chunk_length = min_chunk_length

    def chunk_document(self, document: LoadedDocument) -> List[DocumentChunk]:
        """
        Chunk a single LoadedDocument into DocumentChunks.

        Args:
            document: Loaded document to chunk

        Returns:
            List of DocumentChunk objects ready for embedding
        """
        logger.info(
            "chunking_document",
            doc_id=document.doc_id,
            content_length=len(document.content),
        )

        raw_chunks = self.splitter.split_text(document.content)

        # Filter trivial chunks
        raw_chunks = [
            c for c in raw_chunks
            if len(c.strip()) >= self.min_chunk_length
        ]

        if not raw_chunks:
            logger.warning("no_chunks_produced", doc_id=document.doc_id)
            return []

        # Track position in original document
        search_start = 0
        chunks: List[DocumentChunk] = []

        for i, chunk_text in enumerate(raw_chunks):
            # Find position in original content
            char_start = document.content.find(chunk_text[:50], search_start)
            if char_start == -1:
                char_start = search_start
            char_end = char_start + len(chunk_text)
            search_start = max(0, char_end - self.splitter.chunk_overlap)

            # Detect section for this chunk
            section = detect_section(chunk_text)

            chunk_id = f"{document.doc_id}_chunk_{i:04d}"

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                content=chunk_text,
                chunk_index=i,
                total_chunks=len(raw_chunks),
                source_file=document.metadata.source_file,
                document_category=document.metadata.document_category,
                regulatory_tags=document.metadata.regulatory_tags,
                access_level=document.metadata.access_level,
                section_title=section,
                char_start=char_start,
                char_end=char_end,
                token_estimate=len(chunk_text) // 4,
            )
            chunks.append(chunk)

        logger.info(
            "chunking_complete",
            doc_id=document.doc_id,
            num_chunks=len(chunks),
            avg_chunk_size=sum(len(c.content) for c in chunks) // max(len(chunks), 1),
        )

        return chunks

    def chunk_documents(self, documents: List[LoadedDocument]) -> List[DocumentChunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        logger.info(
            "batch_chunking_complete",
            documents=len(documents),
            total_chunks=len(all_chunks),
        )
        return all_chunks
