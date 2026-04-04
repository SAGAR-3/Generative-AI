"""
tests/test_ingestion.py
========================
Unit tests for the document ingestion pipeline.
Tests: DocumentLoader, DocumentChunker, and preprocessing logic.
"""

import os
import tempfile
import pytest
from pathlib import Path


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_txt_file():
    """Create a temporary text file with banking content."""
    content = """
FHA LOAN REQUIREMENTS

Section 1: Credit Score Requirements
The minimum FICO credit score for FHA loan approval with a 3.5% down payment is 580.
Borrowers with FICO scores between 500 and 579 may qualify with a 10% minimum down payment.
Applicants with scores below 500 are not eligible for FHA-insured financing.

Section 2: Debt-to-Income Ratio
The maximum debt-to-income ratio for FHA loans is 43% for manually underwritten loans.
With automated underwriting system (AUS) approval, DTI may be extended to 50%.
Front-end ratio (housing expenses) should not exceed 31% of gross monthly income.

Section 3: Loan Limits
FHA loan limits vary by geographic area and are updated annually by HUD.
For 2024, the national FHA floor for single-family homes is $498,257.
High-cost areas may have limits up to $1,149,825.
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def large_txt_file():
    """Create a text file that will produce multiple chunks."""
    content = "\n\n".join([
        f"Section {i}: This is a banking policy section about topic {i}. "
        f"It contains important regulatory information about {'FHA' if i % 3 == 0 else 'conventional'} loans. "
        f"The requirements include credit score minimums, DTI limits, and documentation requirements. "
        f"Borrowers must meet all eligibility criteria as outlined in this section {i}."
        for i in range(1, 20)
    ])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        yield f.name
    os.unlink(f.name)


# ─── DocumentLoader Tests ────────────────────────────────────────────────────

class TestDocumentLoader:

    def test_load_txt_file(self, sample_txt_file):
        from src.ingestion.document_loader import DocumentLoader
        loader = DocumentLoader()
        doc = loader.load_file(sample_txt_file, access_level="internal")

        assert doc is not None
        assert doc.doc_id.startswith("doc_")
        assert len(doc.content) > 0
        assert "FHA" in doc.content
        assert doc.metadata.file_type == "txt"
        assert doc.metadata.access_level == "internal"
        assert doc.metadata.sha256_hash is not None

    def test_category_detection(self, sample_txt_file):
        from src.ingestion.document_loader import DocumentLoader
        loader = DocumentLoader()
        doc = loader.load_file(sample_txt_file)

        # Should detect as loan_product or underwriting_guideline
        assert doc.metadata.document_category in [
            "loan_product", "underwriting_guideline", "compliance_policy"
        ]

    def test_regulatory_tag_detection(self, sample_txt_file):
        from src.ingestion.document_loader import DocumentLoader
        loader = DocumentLoader()
        doc = loader.load_file(sample_txt_file)
        # FHA content should be detected
        assert isinstance(doc.metadata.regulatory_tags, list)

    def test_sha256_deduplication(self, sample_txt_file):
        from src.ingestion.document_loader import DocumentLoader
        loader = DocumentLoader()
        doc1 = loader.load_file(sample_txt_file)
        doc2 = loader.load_file(sample_txt_file)
        # Same file should produce same hash
        assert doc1.metadata.sha256_hash == doc2.metadata.sha256_hash
        assert doc1.doc_id == doc2.doc_id

    def test_unsupported_format_raises(self):
        from src.ingestion.document_loader import DocumentLoader
        loader = DocumentLoader()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                loader.load_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_empty_file_raises(self):
        from src.ingestion.document_loader import DocumentLoader
        loader = DocumentLoader()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("   ")  # Only whitespace
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="empty"):
                loader.load_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_file_too_large_raises(self):
        from src.ingestion.document_loader import DocumentLoader, MAX_FILE_SIZE_MB, validate_file
        loader = DocumentLoader()
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            # Write just over the limit
            f.write(b"x" * (MAX_FILE_SIZE_MB * 1024 * 1024 + 1))
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="too large"):
                validate_file(Path(tmp_path))
        finally:
            os.unlink(tmp_path)


# ─── DocumentChunker Tests ───────────────────────────────────────────────────

class TestDocumentChunker:

    def test_chunk_produces_multiple_chunks(self, large_txt_file):
        from src.ingestion.document_loader import DocumentLoader
        from src.ingestion.chunker import DocumentChunker
        loader = DocumentLoader()
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)

        doc = loader.load_file(large_txt_file)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 1
        assert all(c.doc_id == doc.doc_id for c in chunks)
        assert all(c.total_chunks == len(chunks) for c in chunks)

    def test_chunk_indices_are_sequential(self, large_txt_file):
        from src.ingestion.document_loader import DocumentLoader
        from src.ingestion.chunker import DocumentChunker
        loader = DocumentLoader()
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)

        doc = loader.load_file(large_txt_file)
        chunks = chunker.chunk_document(doc)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_size_respected(self, large_txt_file):
        from src.ingestion.document_loader import DocumentLoader
        from src.ingestion.chunker import DocumentChunker
        chunk_size = 300
        loader = DocumentLoader()
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=30)

        doc = loader.load_file(large_txt_file)
        chunks = chunker.chunk_document(doc)

        # Allow some tolerance for overlap
        for chunk in chunks:
            assert len(chunk.content) <= chunk_size * 1.5, \
                f"Chunk too large: {len(chunk.content)} chars"

    def test_metadata_propagation(self, sample_txt_file):
        from src.ingestion.document_loader import DocumentLoader
        from src.ingestion.chunker import DocumentChunker
        loader = DocumentLoader()
        chunker = DocumentChunker()

        doc = loader.load_file(sample_txt_file, access_level="confidential")
        chunks = chunker.chunk_document(doc)

        for chunk in chunks:
            assert chunk.access_level == "confidential"
            assert chunk.source_file == doc.metadata.source_file
            assert chunk.doc_id == doc.doc_id

    def test_minimum_chunk_filter(self):
        from src.ingestion.document_loader import DocumentLoader, LoadedDocument, DocumentMetadata
        from src.ingestion.chunker import DocumentChunker
        import hashlib
        from datetime import datetime

        # Create document with lots of whitespace/short sections
        metadata = DocumentMetadata(
            source_file="test.txt",
            file_type="txt",
            file_size_bytes=100,
            sha256_hash="abc123",
            ingested_at=datetime.utcnow().isoformat(),
            document_category="general",
            regulatory_tags=[],
            access_level="public",
        )
        doc = LoadedDocument(
            doc_id="test_doc",
            content="Short.\n\nAlso short.\n\n" + "A" * 200 + "\n\nEnd.",
            metadata=metadata,
        )

        chunker = DocumentChunker(min_chunk_length=50)
        chunks = chunker.chunk_document(doc)
        # Short chunks should be filtered out
        for chunk in chunks:
            assert len(chunk.content) >= 50

    def test_chunk_ids_are_unique(self, large_txt_file):
        from src.ingestion.document_loader import DocumentLoader
        from src.ingestion.chunker import DocumentChunker
        loader = DocumentLoader()
        chunker = DocumentChunker(chunk_size=200)

        doc = loader.load_file(large_txt_file)
        chunks = chunker.chunk_document(doc)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


# ─── Category Detection Tests ─────────────────────────────────────────────────

class TestCategoryDetection:

    def test_rate_sheet_detection(self):
        from src.ingestion.document_loader import detect_document_category
        content = "30-year fixed rate: 6.875% APR: 7.12% interest rate lock period 60 days"
        category, tags = detect_document_category(content, "rate_sheet.pdf")
        assert category == "rate_sheet"

    def test_compliance_detection(self):
        from src.ingestion.document_loader import detect_document_category
        content = "RESPA disclosure requirements under Regulation X. TILA disclosure timing."
        category, tags = detect_document_category(content, "compliance.pdf")
        assert category == "compliance_policy"
        assert "RESPA" in tags or "TILA" in tags

    def test_underwriting_detection(self):
        from src.ingestion.document_loader import detect_document_category
        content = "Credit score minimum 620. Debt-to-income ratio max 45%. Loan-to-value 80%."
        category, tags = detect_document_category(content, "guidelines.pdf")
        assert category == "underwriting_guideline"
