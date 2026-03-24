"""
Document Parser Service
Handles: PDF, DOCX, plain text, Confluence-like HTML.
Returns clean text + semantic chunks.
"""
import re
import io
from pathlib import Path


def extract_text(filename: str, content_bytes: bytes) -> str:
    """Route to the correct extractor based on file extension."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(content_bytes)
    elif ext in (".docx", ".doc"):
        return _extract_docx(content_bytes)
    elif ext in (".html", ".htm"):
        return _extract_html(content_bytes.decode("utf-8", errors="ignore"))
    else:
        return content_bytes.decode("utf-8", errors="ignore")


def _extract_pdf(data: bytes) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n\n".join(pages)
    except ImportError:
        # Fallback: return raw bytes as utf-8 with best-effort
        return data.decode("utf-8", errors="ignore")


def _extract_docx(data: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        return data.decode("utf-8", errors="ignore")


def _extract_html(html: str) -> str:
    # Simple tag stripper — use BeautifulSoup if available
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n")
    except ImportError:
        return re.sub(r"<[^>]+>", "", html)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, max_tokens: int = 500) -> list[str]:
    """
    Split text into semantic chunks.
    Strategy: split by heading markers / double-newlines, then cap by token count.
    """
    # Split on section headings or blank lines
    raw_chunks = re.split(r"\n{2,}|(?=\n#{1,3} )", text)
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]

    chunks: list[str] = []
    buffer = ""

    for segment in raw_chunks:
        words = segment.split()
        # Rough token estimate: 1 token ≈ 0.75 words
        if len(buffer.split()) + len(words) > int(max_tokens * 0.75):
            if buffer:
                chunks.append(buffer.strip())
            buffer = segment
        else:
            buffer = (buffer + "\n\n" + segment).strip() if buffer else segment

    if buffer:
        chunks.append(buffer.strip())

    return chunks
