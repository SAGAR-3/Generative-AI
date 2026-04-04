"""
src/api/routes.py
==================
FastAPI route handlers.

This is the main orchestration layer that connects:
  Auth → Guardrails (input) → Retrieval → Generation → Guardrails (output) → Response

Every request goes through:
1. JWT authentication
2. Rate limiting
3. PII detection on query
4. Compliance check on query
5. Hybrid retrieval
6. Reranking
7. Context assembly
8. LLM generation
9. PII detection on response
10. Compliance check on response
11. Audit logging
12. Metrics recording
"""

import time
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, List, Optional

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Request, UploadFile, File, status
from fastapi.responses import StreamingResponse

from src.api.models import (
    AuthTokenResponse,
    BlockedQueryResponse,
    CollectionStatsResponse,
    EvaluationResponse,
    FeedbackRequest,
    IngestRequest,
    LoginRequest,
    QueryRequest,
    QueryResponse,
    GuardrailStatus,
    PerformanceMetrics,
    SourceCitation,
    UserRole,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


# ─── Auth Dependency ──────────────────────────────────────────────────────────

async def get_current_user(authorization: str = Header(None)):
    """Extract and validate JWT from Authorization header."""
    from src.api.main import app_state

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization[7:]  # Strip "Bearer "
    try:
        payload = app_state.jwt_handler.verify_token(token)
        from src.security.auth import User
        return User(
            user_id=payload.sub,
            email=payload.email,
            role=payload.role,
            is_active=True,
        )
    except ValueError as e:
        app_state.metrics.auth_failures.labels(failure_type="invalid_token").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─── Authentication Routes ────────────────────────────────────────────────────

@router.post("/auth/login", response_model=AuthTokenResponse, tags=["Authentication"])
async def login(request: LoginRequest):
    """
    Authenticate and receive JWT tokens.

    In production:
    - Verify credentials against your user database/LDAP/SSO
    - Implement MFA for loan officers and above
    - Log all auth events for compliance
    """
    from src.api.main import app_state
    from src.security.audit_logger import AuditEventType

    # DEMO: Accept any email/password — replace with real auth
    # In production: query DB, verify bcrypt hash, check MFA
    demo_roles = {
        "customer@bank.com": UserRole.CUSTOMER,
        "officer@bank.com": UserRole.LOAN_OFFICER,
        "underwriter@bank.com": UserRole.UNDERWRITER,
        "compliance@bank.com": UserRole.COMPLIANCE_OFFICER,
        "admin@bank.com": UserRole.ADMIN,
    }

    role = demo_roles.get(request.email, UserRole.CUSTOMER)
    user_id = f"user_{request.email.split('@')[0]}"
    session_id = str(uuid.uuid4())

    from src.security.auth import User
    user = User(user_id=user_id, email=request.email, role=role, is_active=True)

    access_token = app_state.jwt_handler.create_access_token(user, session_id)
    refresh_token = app_state.jwt_handler.create_refresh_token(user, session_id)

    app_state.audit_logger.log_auth_event(
        event_type=AuditEventType.LOGIN_SUCCESS,
        user_id=user_id,
        email=request.email,
        success=True,
    )

    return AuthTokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=30 * 60,  # 30 minutes
        user_role=role,
    )


# ─── Main Query Route ─────────────────────────────────────────────────────────

@router.post("/query", tags=["Query"])
async def query(
    request_body: QueryRequest,
    http_request: Request,
    user=Depends(get_current_user),
):
    """
    **Main RAG query endpoint.**

    Process a natural language question about home lending using the full RAG pipeline:
    1. PII detection & redaction
    2. Compliance check on query
    3. Hybrid retrieval (dense + BM25)
    4. Cross-encoder reranking
    5. LLM generation with citation
    6. Output compliance & PII check
    7. Audit logging

    Returns grounded answers with source citations.
    """
    from src.api.main import app_state
    from src.generation.prompts import build_rag_prompt, format_context
    from src.security.audit_logger import AuditEventType

    query_id = str(uuid.uuid4())
    start_time = time.time()
    timings = {}
    ip_address = http_request.client.host if http_request.client else None

    # ── Rate Limiting ──────────────────────────────────────────────────────
    is_allowed, remaining = app_state.rate_limiter.is_allowed(
        key=f"{user.user_id}:query",
        limit=user.rate_limit_per_minute,
        window_seconds=60,
    )
    if not is_allowed:
        app_state.metrics.rate_limit_exceeded.labels(user_role=user.role.value).inc()
        app_state.metrics.blocked_queries.labels(block_reason="rate_limit").inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Limit: {user.rate_limit_per_minute} req/min",
            headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
        )

    logger.info(
        "query_received",
        query_id=query_id,
        user_id=user.user_id,
        role=user.role.value,
        question_preview=request_body.question[:50],
    )

    # ── 1. PII Detection on Query ──────────────────────────────────────────
    t0 = time.time()
    pii_result = app_state.pii_redactor.redact(request_body.question)
    timings["pii_check_ms"] = (time.time() - t0) * 1000

    if pii_result.was_modified:
        app_state.metrics.record_security_event(
            pii_types=[e.entity_type.value for e in pii_result.entities_found]
        )

    # Block if too many high-risk PII entities
    if app_state.pii_redactor.should_block(pii_result):
        app_state.audit_logger.log_security_event(
            event_type=AuditEventType.QUERY_BLOCKED,
            user_id=user.user_id,
            description="Query blocked: excessive high-risk PII detected",
            severity="high",
            ip_address=ip_address,
        )
        return BlockedQueryResponse(
            query_id=query_id,
            block_reason="pii_overload",
            message="Your query contains sensitive financial information. "
                    "Please contact your loan officer directly for personalized assistance. "
                    "Do not share account numbers, SSNs, or other sensitive data via chat.",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # Use redacted question for processing
    safe_question = pii_result.redacted_text

    # ── 2. Compliance Check on Query ───────────────────────────────────────
    query_compliance = app_state.compliance_checker.check_query(safe_question)
    if query_compliance.should_block:
        app_state.audit_logger.log_security_event(
            event_type=AuditEventType.QUERY_BLOCKED,
            user_id=user.user_id,
            description=f"Query blocked: compliance violation - {[v.violation_type.value for v in query_compliance.violations]}",
            severity="critical",
            ip_address=ip_address,
        )
        return BlockedQueryResponse(
            query_id=query_id,
            block_reason="compliance_violation",
            message="This query cannot be processed as it may involve regulatory compliance issues. "
                    "Please contact your compliance team.",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ── 3. Hybrid Retrieval ────────────────────────────────────────────────
    t0 = time.time()
    with app_state.metrics.track_retrieval(retrieval_type="hybrid"):
        retrieval_results, retrieval_meta = await app_state.retriever.retrieve_with_metadata(
            query=safe_question,
            user_role=user.role.value,
            top_k=10,
            use_hybrid=app_state.bm25_index._built,
        )
    timings["retrieval_ms"] = (time.time() - t0) * 1000

    if not retrieval_results:
        return QueryResponse(
            query_id=query_id,
            question=safe_question,
            answer="I couldn't find relevant information in our documentation to answer your question. "
                   "Please contact your loan officer for personalized assistance.",
            sources=[],
            guardrails=GuardrailStatus(
                pii_detected=pii_result.was_modified,
                pii_redacted=pii_result.was_modified,
                compliance_passed=True,
                disclaimers_added=False,
                is_grounded=False,
            ),
            performance=PerformanceMetrics(
                total_latency_ms=(time.time() - start_time) * 1000,
                tokens_used=0,
                prompt_tokens=0,
                completion_tokens=0,
                estimated_cost_usd=0.0,
                documents_retrieved=0,
                documents_after_rerank=0,
            ),
            session_id=request_body.session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_used="none",
            user_role=user.role,
        )

    # ── 4. Reranking ───────────────────────────────────────────────────────
    t0 = time.time()
    reranked = await app_state.reranker.rerank(
        query=safe_question,
        results=retrieval_results,
        top_n=min(user.max_context_chunks, 5),
    )
    timings["rerank_ms"] = (time.time() - t0) * 1000

    # ── 5. Context Assembly ────────────────────────────────────────────────
    context = format_context(reranked, max_tokens=6000)
    document_ids = [r.doc_id for r in reranked]

    # ── 6. LLM Generation ─────────────────────────────────────────────────
    t0 = time.time()
    messages, system_prompt = build_rag_prompt(
        question=safe_question,
        context=context,
        user_role=user.role.value,
        conversation_history=request_body.conversation_history,
    )

    with app_state.metrics.track_generation(
        model=app_state.generator.model,
        user_role=user.role.value,
    ):
        gen_result = await app_state.generator.generate(
            messages=messages,
            system_prompt=system_prompt,
            context=context,
        )
    timings["generation_ms"] = (time.time() - t0) * 1000

    # ── 7. Output Compliance & PII Check ──────────────────────────────────
    t0 = time.time()
    output_compliance = app_state.compliance_checker.check_response(gen_result.answer)
    output_pii = app_state.pii_redactor.redact(gen_result.answer)

    final_answer = gen_result.answer
    if output_compliance.modified_text:
        final_answer = output_compliance.modified_text
    if output_pii.was_modified:
        final_answer = output_pii.redacted_text

    timings["output_guard_ms"] = (time.time() - t0) * 1000

    # If output fails compliance, return safe fallback
    if output_compliance.should_block:
        final_answer = (
            "I'm unable to provide a complete answer to this question at this time. "
            "Please contact your loan officer or compliance team for guidance."
        )
        logger.warning(
            "output_blocked_compliance",
            query_id=query_id,
            violations=[v.violation_type.value for v in output_compliance.violations],
        )

    # ── 8. Build Response ──────────────────────────────────────────────────
    total_latency = (time.time() - start_time) * 1000

    sources = [
        SourceCitation(
            source_file=r.source_file.split("/")[-1],
            document_category=r.document_category,
            section_title=r.section_title,
            regulatory_tags=r.regulatory_tags,
            relevance_score=round(r.score, 3),
            chunk_preview=r.content[:200] + "..." if len(r.content) > 200 else r.content,
        )
        for r in reranked
    ]

    guardrails = GuardrailStatus(
        pii_detected=pii_result.was_modified,
        pii_redacted=pii_result.was_modified,
        compliance_passed=not output_compliance.should_block,
        disclaimers_added=output_compliance.disclaimer_added,
        is_grounded=gen_result.is_grounded,
        pii_types_found=[e.entity_type.value for e in pii_result.entities_found],
        compliance_violations=[v.violation_type.value for v in output_compliance.violations],
    )

    performance = PerformanceMetrics(
        total_latency_ms=round(total_latency, 1),
        retrieval_latency_ms=round(timings.get("retrieval_ms", 0), 1),
        generation_latency_ms=round(timings.get("generation_ms", 0), 1),
        tokens_used=gen_result.total_tokens,
        prompt_tokens=gen_result.prompt_tokens,
        completion_tokens=gen_result.completion_tokens,
        estimated_cost_usd=gen_result.cost_estimate_usd,
        cache_hit=False,
        documents_retrieved=len(retrieval_results),
        documents_after_rerank=len(reranked),
    )

    # ── 9. Audit Log ───────────────────────────────────────────────────────
    app_state.audit_logger.log_query(
        user_id=user.user_id,
        user_role=user.role.value,
        query=safe_question,
        response=final_answer,
        query_id=query_id,
        session_id=request_body.session_id or "no-session",
        documents_accessed=document_ids,
        latency_ms=total_latency,
        tokens_used=gen_result.total_tokens,
        cost_usd=gen_result.cost_estimate_usd,
        pii_types=[e.entity_type.value for e in pii_result.entities_found],
        compliance_violations=[v.violation_type.value for v in output_compliance.violations],
        ip_address=ip_address,
    )

    # ── 10. Record Metrics ─────────────────────────────────────────────────
    app_state.metrics.record_query_result(
        user_role=user.role.value,
        document_category=reranked[0].document_category if reranked else "general",
        success=True,
        tokens_input=gen_result.prompt_tokens,
        tokens_output=gen_result.completion_tokens,
        model=gen_result.model,
        cost_usd=gen_result.cost_estimate_usd,
        retrieval_scores=[r.score for r in reranked],
    )

    logger.info(
        "query_complete",
        query_id=query_id,
        total_ms=round(total_latency, 1),
        tokens=gen_result.total_tokens,
        sources=len(sources),
        grounded=gen_result.is_grounded,
    )

    return QueryResponse(
        query_id=query_id,
        question=safe_question,
        answer=final_answer,
        sources=sources,
        guardrails=guardrails,
        performance=performance,
        session_id=request_body.session_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_used=gen_result.model,
        user_role=user.role,
    )


# ─── Feedback Route ───────────────────────────────────────────────────────────

@router.post("/feedback", tags=["Query"])
async def submit_feedback(
    feedback: FeedbackRequest,
    user=Depends(get_current_user),
):
    """Submit feedback on a query response for continuous improvement."""
    logger.info(
        "feedback_received",
        query_id=feedback.query_id,
        user_id=user.user_id,
        rating=feedback.rating,
        helpful=feedback.helpful,
        accurate=feedback.accurate,
    )
    # In production: store in feedback DB for RLHF/fine-tuning pipeline
    return {"status": "accepted", "message": "Thank you for your feedback!"}


# ─── Admin/Internal Routes ────────────────────────────────────────────────────

@router.post("/ingest", tags=["Admin"])
async def ingest_document(
    file: UploadFile = File(...),
    ingest_params: IngestRequest = Depends(),
    user=Depends(get_current_user),
):
    """
    Ingest a new document into the RAG pipeline.
    Requires: loan_officer role or above.
    """
    from src.api.main import app_state

    # RBAC check
    allowed_roles = [UserRole.LOAN_OFFICER, UserRole.UNDERWRITER, UserRole.COMPLIANCE_OFFICER, UserRole.ADMIN]
    if user.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Document ingestion requires loan_officer role or above",
        )

    import tempfile, os
    from src.ingestion.document_loader import DocumentLoader
    from src.ingestion.chunker import DocumentChunker
    from src.embeddings.embedder import DocumentEmbedder

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load
        loader = DocumentLoader()
        doc = loader.load_file(
            tmp_path,
            access_level=ingest_params.access_level,
            effective_date=ingest_params.effective_date,
            department=ingest_params.department,
        )

        # Chunk
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc)

        # Embed
        doc_embedder = DocumentEmbedder(app_state.embedder)
        embedded_pairs = await doc_embedder.embed_chunks(chunks)

        chunk_objects = [pair[0] for pair in embedded_pairs]
        vectors = [pair[1] for pair in embedded_pairs]

        # Store
        app_state.vector_store.upsert_chunks(chunk_objects, vectors)

        # Update BM25 index
        chunk_dicts = [c.to_dict() for c in chunk_objects]
        existing = [{"chunk_id": cid, "content": ""} for cid in app_state.bm25_index.chunk_ids]
        app_state.bm25_index.build(existing + chunk_dicts)

        app_state.metrics.documents_ingested.labels(
            file_type=doc.metadata.file_type,
            document_category=doc.metadata.document_category,
        ).inc()

        logger.info(
            "document_ingested",
            doc_id=doc.doc_id,
            filename=file.filename,
            chunks=len(chunks),
            ingested_by=user.user_id,
        )

        return {
            "status": "success",
            "doc_id": doc.doc_id,
            "filename": file.filename,
            "chunks_created": len(chunks),
            "document_category": doc.metadata.document_category,
            "regulatory_tags": doc.metadata.regulatory_tags,
        }

    finally:
        os.unlink(tmp_path)


@router.get("/collection/stats", response_model=CollectionStatsResponse, tags=["Admin"])
async def collection_stats(user=Depends(get_current_user)):
    """Get vector store collection statistics. Requires admin role."""
    from src.api.main import app_state

    if user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")

    stats = app_state.vector_store.get_collection_stats()
    return CollectionStatsResponse(
        collection_name=stats["collection_name"],
        total_chunks=stats["points_count"],
        total_documents=None,
        index_status=stats["status"],
        vector_dimension=stats["vector_size"],
    )


@router.post("/evaluate", response_model=EvaluationResponse, tags=["Admin"])
async def run_evaluation(user=Depends(get_current_user)):
    """
    Run RAGAS evaluation on the built-in banking eval dataset.
    Compliance Officer or Admin only.
    """
    from src.api.main import app_state
    from src.monitoring.evaluator import RAGASEvaluator, BANKING_EVAL_DATASET

    if user.role not in [UserRole.COMPLIANCE_OFFICER, UserRole.ADMIN]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Compliance or Admin only")

    # Run a mini eval with first 3 samples
    evaluator = RAGASEvaluator(llm_provider="openai")
    samples = BANKING_EVAL_DATASET[:3]

    # Generate responses using the pipeline
    rag_responses = []
    for sample in samples:
        try:
            # Simulate pipeline response (in production: call the full pipeline)
            rag_responses.append({
                "answer": f"Demo evaluation answer for: {sample.question[:50]}...",
                "contexts": ["Sample context from retrieved documents"],
                "latency_ms": 1500,
                "tokens": 500,
                "cost_usd": 0.002,
            })
        except Exception:
            rag_responses.append({"answer": "", "contexts": [], "latency_ms": 0})

    metrics = await evaluator.evaluate(samples, rag_responses)
    eval_id = str(uuid.uuid4())

    return EvaluationResponse(
        eval_id=eval_id,
        overall_score=metrics.overall_score,
        passes_production_bar=metrics.passes_production_bar,
        metrics={
            "faithfulness": metrics.faithfulness,
            "answer_relevancy": metrics.answer_relevancy,
            "context_precision": metrics.context_precision,
            "context_recall": metrics.context_recall,
            "compliance_pass_rate": metrics.compliance_pass_rate,
            "pii_leak_rate": metrics.pii_leak_rate,
            "hallucination_rate": metrics.hallucination_rate,
            "avg_latency_ms": metrics.avg_latency_ms,
        },
        report=metrics.to_report(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
