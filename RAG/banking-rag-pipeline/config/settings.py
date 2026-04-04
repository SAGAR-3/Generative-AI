"""
config/settings.py
==================
Centralized configuration management using Pydantic Settings.
All settings are loaded from environment variables with validation.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM Provider Configuration"""
    provider: str = Field(default="openai", alias="LLM_PROVIDER")
    model: str = Field(default="gpt-4o", alias="LLM_MODEL")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    max_output_tokens: int = Field(default=1024, alias="MAX_OUTPUT_TOKENS")
    temperature: float = Field(default=0.1, alias="TEMPERATURE")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    request_timeout: int = Field(default=30, alias="REQUEST_TIMEOUT")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class EmbeddingSettings(BaseSettings):
    """Embedding Model Configuration"""
    model: str = Field(default="text-embedding-3-large", alias="EMBEDDING_MODEL")
    provider: str = Field(default="openai", alias="EMBEDDING_PROVIDER")
    dimension: int = Field(default=3072, alias="EMBEDDING_DIMENSION")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class VectorStoreSettings(BaseSettings):
    """Qdrant Vector Database Configuration"""
    host: str = Field(default="localhost", alias="QDRANT_HOST")
    port: int = Field(default=6333, alias="QDRANT_PORT")
    api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    collection_name: str = Field(default="home_lending_docs", alias="QDRANT_COLLECTION_NAME")
    https: bool = Field(default=False, alias="QDRANT_HTTPS")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class RetrievalSettings(BaseSettings):
    """Retrieval & Reranking Configuration"""
    top_k: int = Field(default=10, alias="RETRIEVAL_TOP_K")
    rerank_top_n: int = Field(default=5, alias="RERANK_TOP_N")
    hybrid_alpha: float = Field(default=0.7, alias="HYBRID_ALPHA")
    min_similarity_score: float = Field(default=0.65, alias="MIN_SIMILARITY_SCORE")
    max_context_tokens: int = Field(default=8000, alias="MAX_CONTEXT_TOKENS")
    cohere_api_key: Optional[str] = Field(default=None, alias="COHERE_API_KEY")
    reranker_model: str = Field(default="rerank-english-v3.0", alias="RERANKER_MODEL")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class SecuritySettings(BaseSettings):
    """Security & Auth Configuration"""
    jwt_secret_key: str = Field(default="dev-secret-key-change-in-production", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, alias="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    encryption_key: Optional[str] = Field(default=None, alias="ENCRYPTION_KEY")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class GuardrailSettings(BaseSettings):
    """Guardrails Configuration"""
    pii_detection_enabled: bool = Field(default=True, alias="PII_DETECTION_ENABLED")
    toxicity_detection_enabled: bool = Field(default=True, alias="TOXICITY_DETECTION_ENABLED")
    compliance_check_enabled: bool = Field(default=True, alias="COMPLIANCE_CHECK_ENABLED")
    hallucination_check_enabled: bool = Field(default=True, alias="HALLUCINATION_CHECK_ENABLED")
    max_pii_risk_level: str = Field(default="medium", alias="MAX_PII_RISK_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class IngestionSettings(BaseSettings):
    """Document Ingestion Configuration"""
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    max_document_size_mb: int = Field(default=50, alias="MAX_DOCUMENT_SIZE_MB")
    supported_formats: str = Field(default="pdf,docx,txt,xlsx", alias="SUPPORTED_FORMATS")

    @property
    def supported_format_list(self) -> List[str]:
        return [f.strip() for f in self.supported_formats.split(",")]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class MonitoringSettings(BaseSettings):
    """Monitoring & Observability Configuration"""
    prometheus_port: int = Field(default=9090, alias="PROMETHEUS_PORT")
    enable_tracing: bool = Field(default=True, alias="ENABLE_TRACING")
    otel_endpoint: str = Field(default="http://localhost:4317", alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    enable_audit_log: bool = Field(default=True, alias="ENABLE_AUDIT_LOG")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Settings(BaseSettings):
    """Master Settings - aggregates all sub-settings"""
    # App
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    rate_limit_per_minute: int = Field(default=60, alias="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=500, alias="RATE_LIMIT_PER_HOUR")
    cors_origins: List[str] = Field(default=["http://localhost:3000"], alias="CORS_ORIGINS")

    # Sub-settings (instantiated separately for clean organization)
    @property
    def llm(self) -> LLMSettings:
        return LLMSettings()

    @property
    def embedding(self) -> EmbeddingSettings:
        return EmbeddingSettings()

    @property
    def vector_store(self) -> VectorStoreSettings:
        return VectorStoreSettings()

    @property
    def retrieval(self) -> RetrievalSettings:
        return RetrievalSettings()

    @property
    def security(self) -> SecuritySettings:
        return SecuritySettings()

    @property
    def guardrails(self) -> GuardrailSettings:
        return GuardrailSettings()

    @property
    def ingestion(self) -> IngestionSettings:
        return IngestionSettings()

    @property
    def monitoring(self) -> MonitoringSettings:
        return MonitoringSettings()

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton - call this everywhere."""
    return Settings()
