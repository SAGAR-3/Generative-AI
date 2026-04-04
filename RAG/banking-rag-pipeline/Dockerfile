# ══════════════════════════════════════════════════════════════════
# BankRAG Dockerfile — Multi-stage production build
# ══════════════════════════════════════════════════════════════════
# Stages:
#   base:        Python with system deps
#   builder:     Install Python packages
#   production:  Lean runtime image
# ══════════════════════════════════════════════════════════════════

# ── Stage 1: Base ──────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Security: Run as non-root user
RUN groupadd -r bankrag && useradd -r -g bankrag bankrag

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: Builder ───────────────────────────────────────────────
FROM base AS builder

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy model for NER (PII detection)
RUN python -m spacy download en_core_web_sm

# ── Stage 3: Production ────────────────────────────────────────────
FROM base AS production

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=bankrag:bankrag . .

# Create required directories
RUN mkdir -p logs data && chown -R bankrag:bankrag /app

# Security: Switch to non-root user
USER bankrag

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Labels
LABEL maintainer="banking-engineering@yourbank.com"
LABEL version="1.0.0"
LABEL description="BankRAG - Production RAG Pipeline for Home Lending"

# Start command
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--log-level", "info"]
