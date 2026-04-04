"""
src/api/models.py
==================
Pydantic request and response models for the banking RAG API.

All models include:
- Strict type validation
- Field documentation (shows in OpenAPI/Swagger)
- Example values for API docs
- Security: sensitive fields excluded from responses
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ─── Enums ────────────────────────────────────────────────────────────────────

class UserRole(str, Enum):
    CUSTOMER = "customer"
    LOAN_OFFICER = "loan_officer"
    UNDERWRITER = "underwriter"
    COMPLIANCE_OFFICER = "compliance_officer"
    ADMIN = "admin"


class QueryCategory(str, Enum):
    ELIGIBILITY = "eligibility"
    RATES = "rates"
    PROCESS = "process"
    COMPLIANCE = "compliance"
    LOAN_STATUS = "loan_status"
    GENERAL = "general"


class LoanType(str, Enum):
    CONVENTIONAL = "conventional"
    FHA = "fha"
    VA = "va"
    USDA = "usda"
    JUMBO = "jumbo"
    HELOC = "heloc"
    REFINANCE = "refinance"


# ─── Request Models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Main query request to the RAG pipeline."""

    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The user's question about home lending or banking",
        examples=["What is the minimum credit score for an FHA loan?"],
    )
    user_role: UserRole = Field(
        default=UserRole.CUSTOMER,
        description="User's role determines document access level and response style",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for multi-turn conversations",
    )
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        max_length=6,
        description="Previous conversation turns for context (max 6)",
    )
    loan_type_filter: Optional[LoanType] = Field(
        default=None,
        description="Filter results to a specific loan type",
    )
    category_filter: Optional[QueryCategory] = Field(
        default=None,
        description="Filter to a specific document category",
    )
    stream: bool = Field(
        default=False,
        description="Stream the response token by token",
    )

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        """Basic input sanitization."""
        # Strip leading/trailing whitespace
        v = v.strip()
        # Remove null bytes
        v = v.replace("\x00", "")
        # Limit repeated characters (basic spam prevention)
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is the minimum credit score for an FHA loan?",
                    "user_role": "customer",
                    "session_id": None,
                    "stream": False,
                }
            ]
        }
    }


class FeedbackRequest(BaseModel):
    """User feedback on a RAG response."""
    query_id: str = Field(..., description="ID of the query to provide feedback on")
    rating: int = Field(..., ge=1, le=5, description="Rating 1-5 stars")
    helpful: bool = Field(..., description="Was the response helpful?")
    accurate: bool = Field(..., description="Was the response accurate?")
    comment: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional free-text feedback",
    )


class IngestRequest(BaseModel):
    """Request to ingest a document (admin/compliance officer only)."""
    document_url: Optional[str] = Field(
        default=None,
        description="URL to fetch document from",
    )
    document_category: Optional[str] = Field(
        default=None,
        description="Override detected category",
    )
    access_level: str = Field(
        default="internal",
        description="Security classification: public | internal | confidential | restricted",
    )
    effective_date: Optional[str] = Field(
        default=None,
        description="Document effective date (YYYY-MM-DD)",
    )
    department: Optional[str] = Field(
        default=None,
        description="Owning department",
    )


class LoginRequest(BaseModel):
    """Authentication request."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    mfa_token: Optional[str] = Field(
        default=None,
        description="MFA token if 2FA is enabled",
    )


# ─── Response Models ──────────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    """A cited source document chunk."""
    source_file: str = Field(..., description="Source document filename")
    document_category: str = Field(..., description="Document category")
    section_title: Optional[str] = Field(None, description="Section title if detected")
    regulatory_tags: List[str] = Field(default_factory=list, description="Regulatory tags")
    relevance_score: float = Field(..., description="Relevance score (0-1)")
    chunk_preview: str = Field(..., description="First 200 chars of the source chunk")


class GuardrailStatus(BaseModel):
    """Status of all guardrail checks."""
    pii_detected: bool = Field(..., description="Was PII found in the query?")
    pii_redacted: bool = Field(..., description="Was PII successfully redacted?")
    compliance_passed: bool = Field(..., description="Did response pass compliance checks?")
    disclaimers_added: bool = Field(..., description="Were required disclaimers added?")
    is_grounded: bool = Field(..., description="Is the answer grounded in retrieved context?")
    pii_types_found: List[str] = Field(default_factory=list, description="Types of PII found (if any)")
    compliance_violations: List[str] = Field(default_factory=list, description="Compliance violations (if any)")


class PerformanceMetrics(BaseModel):
    """Performance metrics for the query."""
    total_latency_ms: float = Field(..., description="Total end-to-end latency in ms")
    retrieval_latency_ms: Optional[float] = Field(None, description="Retrieval phase latency")
    generation_latency_ms: Optional[float] = Field(None, description="LLM generation latency")
    embedding_latency_ms: Optional[float] = Field(None, description="Embedding latency")
    tokens_used: int = Field(..., description="Total tokens consumed")
    prompt_tokens: int = Field(..., description="Input prompt tokens")
    completion_tokens: int = Field(..., description="Output tokens")
    estimated_cost_usd: float = Field(..., description="Estimated API cost in USD")
    cache_hit: bool = Field(default=False, description="Was this query served from cache?")
    documents_retrieved: int = Field(..., description="Number of documents retrieved")
    documents_after_rerank: int = Field(..., description="Documents after reranking")


class QueryResponse(BaseModel):
    """Full RAG pipeline response."""
    query_id: str = Field(..., description="Unique query identifier for feedback/audit")
    question: str = Field(..., description="Original question (after PII redaction)")
    answer: str = Field(..., description="AI-generated answer")
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="Source documents cited in the answer",
    )
    guardrails: GuardrailStatus = Field(..., description="Guardrail check results")
    performance: PerformanceMetrics = Field(..., description="Performance metrics")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn")
    timestamp: str = Field(..., description="Response timestamp (UTC ISO)")
    model_used: str = Field(..., description="LLM model that generated the answer")
    user_role: UserRole = Field(..., description="Role used for this query")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query_id": "q_abc123",
                    "question": "What is the minimum credit score for an FHA loan?",
                    "answer": "The minimum FICO score for an FHA loan is 580 for a 3.5% down payment. [Source: fha_guidelines.pdf]\n\n*For personalized guidance, please speak with your loan officer.",
                    "sources": [
                        {
                            "source_file": "fha_guidelines.pdf",
                            "document_category": "loan_product",
                            "section_title": "FHA Credit Score Requirements",
                            "regulatory_tags": [],
                            "relevance_score": 0.92,
                            "chunk_preview": "FHA loan requirements: Minimum FICO score of 580 for 3.5% down...",
                        }
                    ],
                    "guardrails": {
                        "pii_detected": False,
                        "pii_redacted": False,
                        "compliance_passed": True,
                        "disclaimers_added": False,
                        "is_grounded": True,
                        "pii_types_found": [],
                        "compliance_violations": [],
                    },
                    "performance": {
                        "total_latency_ms": 1240.5,
                        "tokens_used": 850,
                        "prompt_tokens": 720,
                        "completion_tokens": 130,
                        "estimated_cost_usd": 0.00325,
                        "cache_hit": False,
                        "documents_retrieved": 10,
                        "documents_after_rerank": 5,
                    },
                    "timestamp": "2024-11-15T10:30:00Z",
                    "model_used": "gpt-4o",
                    "user_role": "customer",
                }
            ]
        }
    }


class BlockedQueryResponse(BaseModel):
    """Response when a query is blocked by guardrails."""
    query_id: str
    blocked: bool = True
    block_reason: str
    message: str
    timestamp: str


class AuthTokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_role: UserRole


class CollectionStatsResponse(BaseModel):
    """Vector store collection statistics."""
    collection_name: str
    total_chunks: int
    total_documents: Optional[int]
    index_status: str
    vector_dimension: int


class EvaluationResponse(BaseModel):
    """Evaluation run results."""
    eval_id: str
    overall_score: float
    passes_production_bar: bool
    metrics: Dict[str, Any]
    report: str
    timestamp: str
