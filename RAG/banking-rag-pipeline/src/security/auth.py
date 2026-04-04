"""
src/security/auth.py
=====================
JWT-based authentication and Role-Based Access Control (RBAC)
for the banking RAG pipeline.

Banking roles and their document access:
- customer:            public documents only
- loan_officer:        public + internal documents
- underwriter:         public + internal + confidential
- compliance_officer:  all documents including restricted
- admin:               all documents + system management

Security features:
- JWT tokens with short expiry (30 min access, 7 day refresh)
- Role hierarchy enforcement
- Rate limiting per role
- Session management
- Audit trail for all access
"""

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional, Set
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


# ─── Roles & Permissions ──────────────────────────────────────────────────────

class UserRole(str, Enum):
    CUSTOMER = "customer"
    LOAN_OFFICER = "loan_officer"
    UNDERWRITER = "underwriter"
    COMPLIANCE_OFFICER = "compliance_officer"
    ADMIN = "admin"


# Access levels each role can see
ROLE_ACCESS_LEVELS: dict[UserRole, List[str]] = {
    UserRole.CUSTOMER: ["public"],
    UserRole.LOAN_OFFICER: ["public", "internal"],
    UserRole.UNDERWRITER: ["public", "internal", "confidential"],
    UserRole.COMPLIANCE_OFFICER: ["public", "internal", "confidential", "restricted"],
    UserRole.ADMIN: ["public", "internal", "confidential", "restricted"],
}

# Rate limits per role (requests per minute)
ROLE_RATE_LIMITS: dict[UserRole, int] = {
    UserRole.CUSTOMER: 10,
    UserRole.LOAN_OFFICER: 60,
    UserRole.UNDERWRITER: 60,
    UserRole.COMPLIANCE_OFFICER: 100,
    UserRole.ADMIN: 200,
}

# Maximum context window per role (prevents data leakage)
ROLE_MAX_CONTEXT_CHUNKS: dict[UserRole, int] = {
    UserRole.CUSTOMER: 3,
    UserRole.LOAN_OFFICER: 5,
    UserRole.UNDERWRITER: 7,
    UserRole.COMPLIANCE_OFFICER: 10,
    UserRole.ADMIN: 10,
}


# ─── Token Models ─────────────────────────────────────────────────────────────

@dataclass
class TokenPayload:
    sub: str           # User ID
    email: str
    role: UserRole
    exp: datetime
    iat: datetime
    jti: str           # JWT ID for revocation
    session_id: str


@dataclass
class User:
    user_id: str
    email: str
    role: UserRole
    is_active: bool
    branch_id: Optional[str] = None
    department: Optional[str] = None
    created_at: Optional[datetime] = None

    @property
    def access_levels(self) -> List[str]:
        return ROLE_ACCESS_LEVELS.get(self.role, ["public"])

    @property
    def rate_limit_per_minute(self) -> int:
        return ROLE_RATE_LIMITS.get(self.role, 10)

    @property
    def max_context_chunks(self) -> int:
        return ROLE_MAX_CONTEXT_CHUNKS.get(self.role, 3)


# ─── JWT Token Handler ────────────────────────────────────────────────────────

class JWTHandler:
    """
    JWT token creation and validation.
    
    Security:
    - HS256 algorithm (use RS256 in multi-service production)
    - Short access token expiry (30 min)
    - JTI (JWT ID) for revocation support
    - Audience and issuer validation
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
        issuer: str = "bankrag-api",
        audience: str = "bankrag-clients",
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_expire = timedelta(minutes=access_token_expire_minutes)
        self.refresh_expire = timedelta(days=refresh_token_expire_days)
        self.issuer = issuer
        self.audience = audience

        # In-memory revocation set (use Redis in production)
        self._revoked_jtis: Set[str] = set()

    def create_access_token(self, user: User, session_id: str) -> str:
        """Create a signed JWT access token."""
        try:
            from jose import jwt
        except ImportError:
            raise ImportError("python-jose required: pip install python-jose[cryptography]")

        now = datetime.now(timezone.utc)
        jti = secrets.token_urlsafe(16)

        payload = {
            "sub": user.user_id,
            "email": user.email,
            "role": user.role.value,
            "iat": now,
            "exp": now + self.access_expire,
            "jti": jti,
            "session_id": session_id,
            "iss": self.issuer,
            "aud": self.audience,
            # Banking-specific claims
            "access_levels": user.access_levels,
            "branch_id": user.branch_id,
            "department": user.department,
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(
            "access_token_created",
            user_id=user.user_id,
            role=user.role.value,
            expires_in_minutes=self.access_expire.total_seconds() / 60,
        )
        return token

    def create_refresh_token(self, user: User, session_id: str) -> str:
        """Create a long-lived refresh token."""
        try:
            from jose import jwt
        except ImportError:
            raise ImportError("python-jose required: pip install python-jose[cryptography]")

        now = datetime.now(timezone.utc)
        payload = {
            "sub": user.user_id,
            "session_id": session_id,
            "type": "refresh",
            "iat": now,
            "exp": now + self.refresh_expire,
            "jti": secrets.token_urlsafe(16),
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> TokenPayload:
        """
        Verify and decode a JWT token.

        Raises:
            ValueError: If token is invalid, expired, or revoked
        """
        try:
            from jose import jwt, JWTError
        except ImportError:
            raise ImportError("python-jose required")

        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                audience=self.audience,
                issuer=self.issuer,
                options={"verify_exp": True},
            )
        except JWTError as e:
            logger.warning("token_verification_failed", error=str(e))
            raise ValueError(f"Invalid token: {e}")

        # Check revocation
        jti = payload.get("jti")
        if jti and jti in self._revoked_jtis:
            raise ValueError("Token has been revoked")

        return TokenPayload(
            sub=payload["sub"],
            email=payload.get("email", ""),
            role=UserRole(payload.get("role", "customer")),
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            jti=payload.get("jti", ""),
            session_id=payload.get("session_id", ""),
        )

    def revoke_token(self, jti: str) -> None:
        """Revoke a token by JTI. In production, store in Redis with TTL."""
        self._revoked_jtis.add(jti)
        logger.info("token_revoked", jti=jti)


# ─── Password Hashing ─────────────────────────────────────────────────────────

class PasswordHasher:
    """bcrypt password hashing with configurable work factor."""

    def __init__(self, rounds: int = 12):
        self.rounds = rounds

    def hash(self, password: str) -> str:
        try:
            from passlib.context import CryptContext
            ctx = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=self.rounds)
            return ctx.hash(password)
        except ImportError:
            raise ImportError("passlib[bcrypt] required: pip install passlib[bcrypt]")

    def verify(self, password: str, hashed: str) -> bool:
        try:
            from passlib.context import CryptContext
            ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
            return ctx.verify(password, hashed)
        except ImportError:
            raise ImportError("passlib[bcrypt] required")


# ─── Rate Limiter ─────────────────────────────────────────────────────────────

class InMemoryRateLimiter:
    """
    Simple sliding window rate limiter.
    Use Redis-based limiter in production for distributed deployments.
    """

    def __init__(self):
        self._windows: dict = {}  # {key: [timestamps]}

    def is_allowed(self, key: str, limit: int, window_seconds: int = 60) -> tuple[bool, int]:
        """
        Check if request is within rate limit.

        Args:
            key: Unique identifier (user_id + endpoint)
            limit: Max requests per window
            window_seconds: Time window in seconds

        Returns:
            (is_allowed, remaining_requests)
        """
        import time
        now = time.time()
        window_start = now - window_seconds

        if key not in self._windows:
            self._windows[key] = []

        # Remove expired timestamps
        self._windows[key] = [t for t in self._windows[key] if t > window_start]

        if len(self._windows[key]) >= limit:
            return False, 0

        self._windows[key].append(now)
        remaining = limit - len(self._windows[key])
        return True, remaining


# ─── FastAPI Dependencies ─────────────────────────────────────────────────────

def create_auth_dependency(jwt_handler: JWTHandler):
    """Factory for FastAPI authentication dependency."""

    async def get_current_user(token: str) -> User:
        """
        FastAPI dependency: extracts and validates JWT from Authorization header.
        
        Usage:
            @router.get("/query")
            async def query(user: User = Depends(get_current_user)):
                ...
        """
        try:
            payload = jwt_handler.verify_token(token)
        except ValueError as e:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )

        return User(
            user_id=payload.sub,
            email=payload.email,
            role=payload.role,
            is_active=True,
            created_at=payload.iat,
        )

    return get_current_user


def require_role(*allowed_roles: UserRole):
    """FastAPI dependency factory: enforce role-based access."""
    from fastapi import HTTPException, status

    def role_checker(user: User) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role.value}' not authorized for this endpoint. "
                       f"Required: {[r.value for r in allowed_roles]}",
            )
        return user

    return role_checker
