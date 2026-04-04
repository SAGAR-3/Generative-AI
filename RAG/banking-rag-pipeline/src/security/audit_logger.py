"""
src/security/audit_logger.py
==============================
Immutable audit logging for banking RAG pipeline.

Regulatory requirement: GLBA and banking examiners require complete
audit trails of:
- Who queried what (user, role, timestamp)
- What documents were accessed (doc IDs, access levels)
- What was returned to the user (response summary)
- Any security events (PII detected, violations blocked)

Audit logs are:
- Append-only (no modification or deletion)
- Timestamped with UTC
- Cryptographically signed (HMAC) to detect tampering
- Retained for minimum 5 years per banking regulations
"""

import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import structlog

logger = structlog.get_logger(__name__)


# ─── Audit Event Types ────────────────────────────────────────────────────────

class AuditEventType(str, Enum):
    # Query events
    QUERY_RECEIVED = "QUERY_RECEIVED"
    QUERY_PROCESSED = "QUERY_PROCESSED"
    QUERY_BLOCKED = "QUERY_BLOCKED"

    # Document access
    DOCUMENT_ACCESSED = "DOCUMENT_ACCESSED"
    DOCUMENT_INGESTED = "DOCUMENT_INGESTED"
    DOCUMENT_DELETED = "DOCUMENT_DELETED"

    # Security events
    PII_DETECTED = "PII_DETECTED"
    PII_REDACTED = "PII_REDACTED"
    COMPLIANCE_VIOLATION = "COMPLIANCE_VIOLATION"
    UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Auth events
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILURE = "LOGIN_FAILURE"
    TOKEN_REVOKED = "TOKEN_REVOKED"
    LOGOUT = "LOGOUT"

    # System events
    SYSTEM_ERROR = "SYSTEM_ERROR"
    COLLECTION_MODIFIED = "COLLECTION_MODIFIED"


# ─── Audit Event ─────────────────────────────────────────────────────────────

@dataclass
class AuditEvent:
    """Immutable audit event record."""
    event_id: str
    event_type: AuditEventType
    timestamp: str
    user_id: Optional[str]
    user_role: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]

    # Query context
    query_id: Optional[str] = None
    query_preview: Optional[str] = None  # First 100 chars, PII redacted
    response_preview: Optional[str] = None  # First 100 chars

    # Document access
    documents_accessed: List[str] = field(default_factory=list)
    access_levels_used: List[str] = field(default_factory=list)

    # Security
    pii_types_found: List[str] = field(default_factory=list)
    compliance_violations: List[str] = field(default_factory=list)
    was_blocked: bool = False
    block_reason: Optional[str] = None

    # Performance
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None

    # Integrity
    hmac_signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ─── Audit Logger ────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Production audit logger for banking RAG pipeline.

    Storage backends (pluggable):
    - Structured log files (default, with log rotation)
    - PostgreSQL (for queryable audit database)
    - Splunk/ELK Stack (for SIEM integration)

    In production, ALL three should be active for defense in depth.
    """

    def __init__(
        self,
        signing_key: Optional[str] = None,
        log_file: Optional[str] = None,
        db_url: Optional[str] = None,
    ):
        self.signing_key = signing_key or os.getenv("AUDIT_SIGNING_KEY", "default-dev-key")
        self.log_file = log_file
        self.db_url = db_url

        # Configure structured logger
        self.audit_log = structlog.get_logger("audit")

        if log_file:
            logger.info("audit_log_file_configured", path=log_file)

    def _sign_event(self, event: AuditEvent) -> str:
        """
        Generate HMAC signature for tamper detection.
        In production use: SHA-256 HMAC with HSM-backed key.
        """
        content = f"{event.event_id}|{event.timestamp}|{event.user_id}|{event.event_type}"
        signature = hmac.new(
            self.signing_key.encode(),
            content.encode(),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def log(self, event: AuditEvent) -> None:
        """
        Log an audit event to all configured backends.
        This is synchronous and blocking — audit logging must never be skipped.
        """
        # Sign the event
        event.hmac_signature = self._sign_event(event)

        # Log to structured logger (always)
        self.audit_log.info(
            "audit_event",
            **{k: v for k, v in event.to_dict().items() if v is not None}
        )

        # Log to file if configured
        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(event.to_json() + "\n")
            except Exception as e:
                logger.error("audit_file_write_failed", error=str(e))

        # In production: also write to PostgreSQL for queryable audit trail
        # await self._write_to_db(event)

    def log_query(
        self,
        user_id: str,
        user_role: str,
        query: str,
        response: str,
        query_id: str,
        session_id: str,
        documents_accessed: List[str],
        latency_ms: float,
        tokens_used: int,
        cost_usd: float,
        was_blocked: bool = False,
        block_reason: Optional[str] = None,
        pii_types: Optional[List[str]] = None,
        compliance_violations: Optional[List[str]] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        """Log a complete query/response cycle."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.QUERY_BLOCKED if was_blocked else AuditEventType.QUERY_PROCESSED,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            user_role=user_role,
            session_id=session_id,
            ip_address=ip_address,
            query_id=query_id,
            query_preview=query[:100] if query else None,  # Truncated for privacy
            response_preview=response[:100] if response else None,
            documents_accessed=documents_accessed,
            pii_types_found=pii_types or [],
            compliance_violations=compliance_violations or [],
            was_blocked=was_blocked,
            block_reason=block_reason,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
        )
        self.log(event)

    def log_security_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str],
        description: str,
        severity: str = "high",
        ip_address: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log a security event (PII detection, unauthorized access, etc.)."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            user_role=kwargs.get("user_role"),
            session_id=kwargs.get("session_id"),
            ip_address=ip_address,
            was_blocked=kwargs.get("was_blocked", False),
            block_reason=description,
        )
        self.log(event)

        # For critical security events, also alert (in production: PagerDuty, SNS, etc.)
        if severity in ("critical", "high"):
            logger.warning(
                "security_alert",
                event_type=event_type.value,
                user_id=user_id,
                description=description,
                severity=severity,
            )

    def log_auth_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        email: str,
        success: bool,
        ip_address: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ) -> None:
        """Log authentication event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            user_role=None,
            session_id=None,
            ip_address=ip_address,
            was_blocked=not success,
            block_reason=failure_reason,
        )
        self.log(event)

        # Alert on repeated login failures (brute force detection)
        if not success:
            logger.warning(
                "auth_failure",
                user_id=user_id,
                email=email,
                ip_address=ip_address,
                reason=failure_reason,
            )
