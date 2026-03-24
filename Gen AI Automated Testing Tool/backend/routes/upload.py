"""
Upload Route
POST /api/upload  — accepts a BRD file, extracts text, chunks it, stores embeddings.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from models.schemas import BRDDocument
from services.document_parser import extract_text, chunk_text
from services import vector_store

import uuid

router = APIRouter()

# In-memory store for demo (replace with a database in production)
_documents: dict[str, BRDDocument] = {}


@router.post("/", response_model=BRDDocument)
async def upload_brd(file: UploadFile = File(...)):
    """
    Upload a BRD document (PDF, DOCX, TXT, HTML).
    Extracts text, chunks it, and stores embeddings for RAG.
    """
    allowed = {".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm"}
    ext = "." + (file.filename or "").rsplit(".", 1)[-1].lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}",
        )

    content_bytes = await file.read()

    if len(content_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=413, detail="File too large. Max 10 MB.")

    # Extract and chunk
    text   = extract_text(file.filename, content_bytes)
    chunks = chunk_text(text)

    doc = BRDDocument(
        filename=file.filename,
        content=text,
        chunks=chunks,
    )

    # Store embeddings (best-effort — vector store may not be installed)
    try:
        vector_store.store_chunks(doc.id, chunks)
    except Exception as e:
        print(f"[vector_store] Warning: {e} — continuing without embeddings.")

    _documents[doc.id] = doc
    return doc


@router.get("/{document_id}", response_model=BRDDocument)
def get_document(document_id: str):
    doc = _documents.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@router.get("/")
def list_documents():
    return [
        {"id": d.id, "filename": d.filename, "chunks": len(d.chunks)}
        for d in _documents.values()
    ]
