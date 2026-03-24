"""
Generate Route
POST /api/generate  — runs RAG retrieval + LLM to produce test cases.
POST /api/generate/review — submit human review actions.
"""
from fastapi import APIRouter, HTTPException
from models.schemas import (
    GenerateRequest, GenerateResponse,
    ReviewAction, TestCase
)
from services import llm_service, vector_store
from routes.upload import _documents

router = APIRouter()

# In-memory results store (replace with DB in production)
_results: dict[str, list[TestCase]] = {}


@router.post("/", response_model=GenerateResponse)
def generate_test_cases(req: GenerateRequest):
    """
    Retrieve relevant BRD chunks via RAG, call the LLM,
    and return structured test cases.
    """
    doc = _documents.get(req.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found. Upload it first.")

    # RAG: try to retrieve semantically relevant chunks, fall back to all chunks
    try:
        query  = "actors, business rules, user flows, acceptance criteria, validation"
        chunks = vector_store.retrieve_relevant_chunks(req.document_id, query, n_results=10)
        if not chunks:
            raise ValueError("empty")
    except Exception:
        chunks = doc.chunks[:20]  # Fallback: use first 20 raw chunks

    # LLM generation
    try:
        test_cases, coverage_gaps = llm_service.generate_test_cases(
            brd_chunks=chunks,
            test_types=req.test_types,
            max_cases=req.max_cases,
            model=req.model,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    _results[req.document_id] = test_cases

    return GenerateResponse(
        document_id=req.document_id,
        test_cases=test_cases,
        total=len(test_cases),
        coverage_gaps=coverage_gaps,
    )


@router.get("/{document_id}", response_model=GenerateResponse)
def get_results(document_id: str):
    cases = _results.get(document_id, [])
    return GenerateResponse(
        document_id=document_id,
        test_cases=cases,
        total=len(cases),
    )


@router.post("/review")
def review_test_case(action: ReviewAction):
    """
    Accept a QA engineer's review decision.
    approve → status = approved
    reject  → status = rejected (feedback stored)
    edit    → replace with edited version
    """
    # Find and update the test case across all results
    for doc_id, cases in _results.items():
        for i, tc in enumerate(cases):
            if tc.id == action.test_case_id:
                if action.action == "approve":
                    cases[i].status = "approved"
                elif action.action == "reject":
                    cases[i].status = "rejected"
                elif action.action == "edit" and action.edited_case:
                    cases[i] = action.edited_case
                    cases[i].status = "approved"
                return {"success": True, "test_case_id": action.test_case_id}

    raise HTTPException(status_code=404, detail="Test case not found.")
