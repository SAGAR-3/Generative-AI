"""
src/api/main.py
================
FastAPI application entry point for the banking RAG pipeline.

Features:
- JWT authentication with RBAC
- Rate limiting per user/role
- Request/response validation
- Prometheus metrics endpoint
- Health & readiness probes
- Structured logging with request IDs
- CORS configuration
- Graceful shutdown
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

logger = structlog.get_logger(__name__)

# ─── Application State ────────────────────────────────────────────────────────

class AppState:
    """Holds initialized pipeline components shared across requests."""
    embedder = None
    vector_store = None
    retriever = None
    reranker = None
    generator = None
    pii_redactor = None
    compliance_checker = None
    jwt_handler = None
    audit_logger = None
    rate_limiter = None
    metrics = None
    bm25_index = None


app_state = AppState()


# ─── Lifespan (startup/shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup, clean up on shutdown."""
    import os
    logger.info("bankrag_starting_up")

    # Load settings
    from config.settings import get_settings
    settings = get_settings()

    # Initialize metrics
    from src.monitoring.metrics import get_metrics
    app_state.metrics = get_metrics()

    # Initialize security
    from src.security.auth import JWTHandler, InMemoryRateLimiter
    app_state.jwt_handler = JWTHandler(
        secret_key=settings.security.jwt_secret_key,
        algorithm=settings.security.jwt_algorithm,
        access_token_expire_minutes=settings.security.access_token_expire_minutes,
    )
    app_state.rate_limiter = InMemoryRateLimiter()

    # Initialize audit logger
    from src.security.audit_logger import AuditLogger
    app_state.audit_logger = AuditLogger(
        signing_key=settings.security.jwt_secret_key,
        log_file="logs/audit.jsonl" if settings.monitoring.enable_audit_log else None,
    )

    # Initialize guardrails
    from src.guardrails.pii_redactor import PIIRedactor
    from src.guardrails.compliance_checker import ComplianceChecker
    app_state.pii_redactor = PIIRedactor(use_presidio=False)  # Set True if presidio installed
    app_state.compliance_checker = ComplianceChecker()

    # Initialize embedder
    from src.embeddings.embedder import EmbedderFactory
    app_state.embedder = EmbedderFactory.create(
        provider=settings.embedding.provider,
        api_key=getattr(settings.llm, f"{settings.embedding.provider}_api_key", None),
        model=settings.embedding.model,
    )

    # Initialize vector store
    from src.embeddings.vector_store import QdrantVectorStore
    app_state.vector_store = QdrantVectorStore(
        host=settings.vector_store.host,
        port=settings.vector_store.port,
        api_key=settings.vector_store.api_key,
        collection_name=settings.vector_store.collection_name,
        vector_dimension=settings.embedding.dimension,
    )
    try:
        app_state.vector_store.ensure_collection_exists()
    except Exception as e:
        logger.warning("vector_store_init_failed", error=str(e))

    # Initialize retriever
    from src.retrieval.retriever import HybridRetriever, BM25Index
    app_state.bm25_index = BM25Index()
    app_state.retriever = HybridRetriever(
        vector_store=app_state.vector_store,
        embedder=app_state.embedder,
        bm25_index=app_state.bm25_index,
    )

    # Initialize reranker
    from src.retrieval.reranker import RerankerFactory
    app_state.reranker = RerankerFactory.create(
        provider="cohere",
        api_key=settings.retrieval.cohere_api_key,
        top_n=settings.retrieval.rerank_top_n,
    )

    # Initialize LLM generator
    from src.generation.generator import GeneratorFactory
    app_state.generator = GeneratorFactory.create(
        provider=settings.llm.provider,
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_output_tokens,
    )

    logger.info(
        "bankrag_startup_complete",
        environment=settings.environment,
        llm_provider=settings.llm.provider,
        llm_model=settings.llm.model,
        embedding_model=settings.embedding.model,
        vector_store_host=settings.vector_store.host,
    )

    yield  # Application runs

    # Shutdown cleanup
    logger.info("bankrag_shutting_down")


# ─── FastAPI App ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="BankRAG - Home Lending AI Assistant",
        description="""
Production-grade RAG pipeline for banking and home lending.

## Features
- 🔍 Hybrid retrieval (dense + BM25 + reranking)
- 🛡️ Banking regulatory compliance (RESPA, TILA, ECOA, FCRA)
- 🔒 Role-based access control (customer → compliance officer)
- 🕵️ PII detection and redaction
- 📊 RAGAS evaluation metrics
- 🚨 Real-time audit logging

## Roles
- **customer**: Public docs only, 10 req/min
- **loan_officer**: Internal docs, 60 req/min
- **underwriter**: Confidential docs, 60 req/min
- **compliance_officer**: All docs, 100 req/min
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ──────────────────────────────────────────────────────────
    from config.settings import get_settings
    settings = get_settings()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Request ID & Logging Middleware ─────────────────────────────────────
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.time()

        # Bind request context for all log messages
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            ip=request.client.host if request.client else "unknown",
        )

        response = await call_next(request)

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "request_complete",
            status_code=response.status_code,
            duration_ms=round(duration_ms, 1),
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(duration_ms, 1))

        structlog.contextvars.clear_contextvars()
        return response

    # ── Exception Handlers ──────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_exception", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. This has been logged.",
                "request_id": request.headers.get("X-Request-ID"),
            },
        )

    # ── Routes ──────────────────────────────────────────────────────────────
    from src.api.routes import router
    app.include_router(router, prefix="/api/v1")

    # ── Prometheus metrics endpoint ──────────────────────────────────────────
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # ── Health Endpoints ─────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health_check():
        """Basic liveness probe."""
        return {"status": "healthy", "service": "bankrag-api"}

    @app.get("/ready", tags=["System"])
    async def readiness_check():
        """Readiness probe — checks all dependencies."""
        checks = {}
        status_code = 200

        # Check vector store
        if app_state.vector_store:
            try:
                checks["vector_store"] = await app_state.vector_store.health_check()
            except Exception:
                checks["vector_store"] = False
                status_code = 503

        # Check LLM (basic)
        checks["llm"] = app_state.generator is not None
        checks["embedder"] = app_state.embedder is not None
        checks["guardrails"] = app_state.pii_redactor is not None

        all_healthy = all(checks.values())
        return JSONResponse(
            content={
                "status": "ready" if all_healthy else "degraded",
                "checks": checks,
            },
            status_code=status_code,
        )

    @app.get("/", tags=["System"])
    async def root():
        return {
            "service": "BankRAG - Home Lending AI Assistant",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    from config.settings import get_settings
    settings = get_settings()

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else 4,
        log_level=settings.monitoring.log_level.lower(),
    )
