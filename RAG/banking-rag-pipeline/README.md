# 🏦 BankRAG — Production-Grade RAG Pipeline for Home Lending & Banking

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-1.11-DC143C?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen?style=flat-square)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluated-blueviolet?style=flat-square)

**An end-to-end, production-ready Retrieval-Augmented Generation (RAG) system built for the banking and home lending domain. Covers the full pipeline from document ingestion to LLM response — with compliance guardrails, RBAC security, PII protection, and RAGAS evaluation baked in.**

[Problem Statement](#-problem-statement) · [Architecture](#-system-architecture) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Security](#-security--compliance) · [Metrics](#-performance-metrics) · [Contributing](#-contributing)

</div>

---

## 📌 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Why RAG for Banking?](#-why-rag-for-banking)
3. [System Architecture](#-system-architecture)
4. [End-to-End Pipeline Flow](#-end-to-end-pipeline-flow)
5. [Project Structure](#-project-structure)
6. [Technology Stack](#-technology-stack)
7. [Prerequisites](#-prerequisites)
8. [Quick Start](#-quick-start)
9. [Configuration Reference](#-configuration-reference)
10. [Module Deep-Dive](#-module-deep-dive)
11. [API Reference](#-api-reference)
12. [Role-Based Access Control](#-role-based-access-control)
13. [Security & Compliance](#-security--compliance)
14. [Performance Metrics](#-performance-metrics)
15. [Running Tests](#-running-tests)
16. [Evaluation with RAGAS](#-evaluation-with-ragas)
17. [Deployment Guide](#-deployment-guide)
18. [Troubleshooting](#-troubleshooting)
19. [Contributing](#-contributing)

---

## 🎯 Problem Statement

A major retail bank's **mortgage and home-lending division** receives over **10,000 customer and loan-officer queries daily** across web chat, call centers, and email portals. These queries span:

- **Loan Eligibility** — "Do I qualify for an FHA loan with a 600 credit score?"
- **Rate Information** — "What are today's 30-year fixed rates?"
- **Documentation Requirements** — "What documents do I need for a conventional loan?"
- **Regulatory Compliance** — "What disclosures are required under RESPA within 3 days?"
- **Underwriting Guidelines** — "What is the maximum DTI for a jumbo loan?"
- **Loan Status** — "What conditions are still outstanding on my file?"

**Current Pain Points (without RAG):**
- ❌ Loan officers spend 40% of time answering repetitive policy questions
- ❌ Inconsistent answers across branches — different people citing different guideline versions
- ❌ Customers wait hours or days for responses via email
- ❌ Compliance violations due to outdated or inaccurate information being shared
- ❌ No audit trail of what information was given to whom and when

**BankRAG solves all of this** with an AI assistant that retrieves answers from the bank's own authoritative documents — with citations, compliance guardrails, PII protection, and a full audit trail.

---

## 💡 Why RAG for Banking?

Unlike general-purpose chatbots, RAG is uniquely suited to banking because:

| Challenge | Why RAG Wins |
|-----------|-------------|
| **Accuracy requirements** | Answers are grounded in retrieved source documents, not LLM hallucinations |
| **Frequent policy changes** | Update the document store — no model retraining needed |
| **Regulatory citations** | Sources attached to every answer for audit and compliance teams |
| **Proprietary knowledge** | Bank's internal guidelines never leave the organization |
| **Access control** | Different roles see different documents — enforced at the retrieval layer |
| **Explainability** | Examiners can trace every answer back to a source document |

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         BANKRAG SYSTEM ARCHITECTURE                            ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                        DOCUMENT INGESTION PIPELINE                          │
  │                                                                              │
  │  [PDF Policies]  [DOCX Guidelines]  [TXT Regulations]  [XLSX Rate Sheets]   │
  │         │               │                  │                   │            │
  │         └───────────────┴──────────────────┴───────────────────┘            │
  │                                    │                                         │
  │                      ┌─────────────▼──────────────┐                         │
  │                      │  DocumentLoader             │  SHA-256 dedup          │
  │                      │  • Format + size validate   │  Auto-category detect   │
  │                      │  • MIME type security check │  Regulatory tag extract │
  │                      └─────────────┬──────────────┘                         │
  │                                    │                                         │
  │                      ┌─────────────▼──────────────┐                         │
  │                      │  BankingTextSplitter        │  Chunk size: 512 chars  │
  │                      │  • Section-aware chunking   │  Overlap: 50 chars      │
  │                      │  • Paragraph boundaries     │  Min length: 50 chars   │
  │                      └─────────────┬──────────────┘                         │
  │                                    │                                         │
  │                      ┌─────────────▼──────────────┐                         │
  │                      │  OpenAI Embedder            │  text-embedding-3-large │
  │                      │  • Async batch embedding    │  3072 dimensions        │
  │                      │  • Retry + in-memory cache  │  $0.13 / 1M tokens      │
  │                      └─────────────┬──────────────┘                         │
  │                                    │                                         │
  │              ┌─────────────────────▼──────────────────────┐                 │
  │              │            Qdrant Vector Store              │                 │
  │              │  HNSW index (m=16, ef_construct=200)        │                 │
  │              │  Payload indexes: access_level, category    │                 │
  │              │  Cosine similarity + BM25 (in-memory)       │                 │
  │              └────────────────────────────────────────────┘                 │
  └──────────────────────────────────────────────────────────────────────────────┘

                                        │
                                (documents stored)
                                        │
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                              QUERY PIPELINE                                 │
  │                                                                              │
  │  USER QUERY                                                                  │
  │      │                                                                       │
  │      ▼                                                                       │
  │  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────────────┐  │
  │  │  FastAPI Layer   │──▶│ INPUT GUARDRAILS  │──▶│   Hybrid Retrieval      │  │
  │  │  • JWT Auth      │   │ • PII Detection   │   │   • Dense (Qdrant HNSW) │  │
  │  │  • Rate Limiting │   │ • Compliance Chk  │   │   • BM25 Sparse Search  │  │
  │  │  • RBAC enforce  │   │ • Block / Redact  │   │   • RRF Score Fusion    │  │
  │  └──────────────────┘   └──────────────────┘   └────────────┬────────────┘  │
  │                                                              │               │
  │                                                 ┌────────────▼────────────┐  │
  │                                                 │  Cross-Encoder Reranker │  │
  │                                                 │  Cohere Rerank v3       │  │
  │                                                 │  (top-5 after rerank)   │  │
  │                                                 └────────────┬────────────┘  │
  │                                                              │               │
  │                                                 ┌────────────▼────────────┐  │
  │                                                 │  LLM Generation         │  │
  │                                                 │  GPT-4o / Claude        │  │
  │                                                 │  Role-specific prompts  │  │
  │                                                 │  T=0.1, citations reqd  │  │
  │                                                 └────────────┬────────────┘  │
  │                                                              │               │
  │  ┌──────────────────┐   ┌──────────────────┐                │               │
  │  │  Audit Logger    │◀──│ OUTPUT GUARDRAILS │◀───────────────┘               │
  │  │  HMAC-signed     │   │ • PII Scan        │                                │
  │  │  Immutable log   │   │ • Compliance Chk  │                                │
  │  │  GLBA-compliant  │   │ • Disclaimer Add  │                                │
  │  └──────────────────┘   └────────┬──────────┘                                │
  │                                  │                                           │
  │  ┌──────────────────┐            ▼                                           │
  │  │  Prometheus      │     FINAL RESPONSE                                     │
  │  │  + Grafana       │  (answer + sources + performance metrics)              │
  │  └──────────────────┘                                                        │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 End-to-End Pipeline Flow

Every query passes through **10 sequential stages**:

```
Stage  1  →  JWT Authentication + Rate Limiting
Stage  2  →  PII Detection & Redaction (on raw query)
Stage  3  →  Compliance Screening (ECOA/UDAAP/TILA check on query)
Stage  4  →  Embed Query  (OpenAI text-embedding-3-large, ~50ms)
Stage  5  →  Dense Vector Search  (Qdrant HNSW, top-20, access_level filtered)
Stage  6  →  BM25 Sparse Search  (keyword match, top-20, same access filter)
Stage  7  →  Reciprocal Rank Fusion  (Dense×0.7 + Sparse×0.3)
Stage  8  →  Cross-Encoder Reranking  (Cohere Rerank v3 → top-5)
Stage  9  →  LLM Generation  (GPT-4o, T=0.1, role-specific system prompt)
Stage 10  →  Output PII Scan + Compliance Check + Disclaimer Injection
           →  Groundedness Verification
           →  HMAC-signed Audit Logging (GLBA-compliant)
           →  Prometheus Metrics Recording
           →  Structured Response Returned to Client
```

**Latency Budget:**

| Stage | Typical Time |
|-------|-------------|
| Auth + Rate Limit | ~5 ms |
| Input PII + Compliance | ~8 ms |
| Query Embedding | ~50 ms |
| Vector Search (Qdrant) | ~30 ms |
| BM25 Sparse Search | ~5 ms |
| Reranking (Cohere API) | ~200 ms |
| LLM Generation (GPT-4o) | ~800 ms |
| Output Guardrails | ~10 ms |
| Audit + Metrics | ~5 ms |
| **Total P50** | **~1,100 ms** |
| **Total P95** | **< 3,000 ms** |

---

## 🗂️ Project Structure

```
banking-rag-pipeline/
│
├── 📄 README.md                         ←  This file — full documentation
├── 📄 requirements.txt                  ←  All Python dependencies, pinned
├── 📄 .env.example                      ←  Environment variable template
├── 📄 docker-compose.yml                ←  Full infrastructure stack (6 services)
├── 📄 Dockerfile                        ←  Multi-stage production build
│
├── 📁 src/                              ←  Core application source code
│   │
│   ├── 📁 ingestion/                    ←  Document loading & chunking
│   │   ├── __init__.py
│   │   ├── document_loader.py           ←  PDF/DOCX/TXT/XLSX loader + security validation
│   │   └── chunker.py                   ←  Banking-aware recursive text splitter
│   │
│   ├── 📁 embeddings/                   ←  Vector representations
│   │   ├── __init__.py
│   │   ├── embedder.py                  ←  OpenAI + HuggingFace embedding models
│   │   └── vector_store.py              ←  Qdrant CRUD + filtered semantic search
│   │
│   ├── 📁 retrieval/                    ←  Retrieval engine
│   │   ├── __init__.py
│   │   ├── retriever.py                 ←  Hybrid Dense+BM25 retrieval + RRF fusion
│   │   └── reranker.py                  ←  Cohere Rerank + local cross-encoder fallback
│   │
│   ├── 📁 generation/                   ←  LLM response generation
│   │   ├── __init__.py
│   │   ├── generator.py                 ←  GPT-4o / Claude generator + streaming support
│   │   └── prompts.py                   ←  Role-specific banking system prompt templates
│   │
│   ├── 📁 guardrails/                   ←  Safety & compliance layer
│   │   ├── __init__.py
│   │   ├── pii_redactor.py              ←  13-entity PII detection & redaction
│   │   └── compliance_checker.py        ←  ECOA/TILA/RESPA/UDAAP violation checks
│   │
│   ├── 📁 security/                     ←  Authentication & audit
│   │   ├── __init__.py
│   │   ├── auth.py                      ←  JWT handler + 5-level RBAC + rate limiter
│   │   └── audit_logger.py              ←  HMAC-signed immutable audit trail (GLBA)
│   │
│   ├── 📁 monitoring/                   ←  Observability
│   │   ├── __init__.py
│   │   ├── metrics.py                   ←  25+ Prometheus metrics (latency/quality/security)
│   │   └── evaluator.py                 ←  RAGAS evaluation framework + banking eval set
│   │
│   └── 📁 api/                          ←  HTTP API layer
│       ├── __init__.py
│       ├── main.py                      ←  FastAPI app factory + startup/shutdown lifespan
│       ├── routes.py                    ←  All endpoint handlers (query/ingest/eval/auth)
│       └── models.py                    ←  Pydantic request/response schemas
│
├── 📁 config/                           ←  Application configuration
│   ├── __init__.py
│   ├── settings.py                      ←  Pydantic Settings — all env-var driven
│   └── prometheus.yml                   ←  Prometheus scrape targets config
│
├── 📁 tests/                            ←  Automated test suite
│   ├── __init__.py
│   ├── test_ingestion.py                ←  DocumentLoader + Chunker unit tests
│   └── test_guardrails.py               ←  PII + compliance guardrail tests (critical)
│
├── 📁 scripts/                          ←  CLI tools
│   ├── ingest_documents.py              ←  Batch document ingestion with progress reporting
│   └── run_evaluation.py                ←  RAGAS evaluation runner + CI/CD gate
│
├── 📁 data/
│   └── 📁 sample/
│       └── home_lending_policy.txt      ←  Sample FHA/VA/Conventional policy doc for testing
│
├── 📁 docs/
│   └── ARCHITECTURE.md                  ←  Detailed architecture + security guide
│
└── 📁 logs/
    └── .gitkeep                         ←  Audit logs written here at runtime (gitignored)
```

---

## 🛠️ Technology Stack

| Layer | Technology | Version | Why This Choice |
|-------|-----------|---------|-----------------|
| **API Framework** | FastAPI | 0.115 | Async-native, auto OpenAPI docs, Pydantic validation |
| **Vector Database** | Qdrant | 1.11 | Best performance/cost, native payload filtering, RBAC-ready |
| **Embedding (Primary)** | OpenAI text-embedding-3-large | latest | Best accuracy (3072 dims), $0.13/1M tokens |
| **Embedding (Alt)** | BAAI/bge-large-en-v1.5 | HuggingFace | Free local alternative, 1024 dims |
| **LLM (Primary)** | GPT-4o | OpenAI | Best reasoning, 128K context, low hallucination rate |
| **LLM (Alt)** | Claude 3.5 Sonnet | Anthropic | Strong alternative for instruction following |
| **Sparse Search** | BM25 (rank_bm25) | 0.2.2 | Ideal for banking codes ("Reg Z", "FICO 580", "DTI 43%") |
| **Reranker** | Cohere Rerank v3 | API | Best cross-encoder accuracy for passage relevance |
| **Reranker (Alt)** | ms-marco-MiniLM-L-6-v2 | HuggingFace | Local, free, good for development |
| **PII Detection** | Presidio + Custom Regex | 2.2 | Layered: NLP for unstructured + regex for structured PII |
| **Authentication** | JWT via python-jose | 3.3 | Stateless, standard, supports token revocation |
| **Password Hashing** | bcrypt via passlib | 1.7 | Industry standard, configurable work factor |
| **Cache** | Redis | 7.4 | Sub-millisecond query caching + rate limit state |
| **Audit Database** | PostgreSQL | 16 | ACID-compliant, queryable, long-term retention |
| **Metrics** | Prometheus | 2.54 | Pull-based metrics, rich PromQL query language |
| **Dashboards** | Grafana | 11.2 | Best-in-class metrics visualization |
| **Evaluation** | RAGAS | 0.1 | LLM-as-judge evaluation, industry standard for RAG |
| **Containerization** | Docker + Compose | latest | Reproducible environments, one-command startup |
| **Logging** | structlog | 24.4 | Structured JSON logging with context binding |

---

## ✅ Prerequisites

| Requirement | Minimum Version | Check Command |
|-------------|----------------|---------------|
| Python | 3.11+ | `python --version` |
| Docker | 24.0+ | `docker --version` |
| Docker Compose | 2.20+ | `docker compose version` |
| Git | 2.0+ | `git --version` |
| OpenAI API Key | — | Required for embeddings + LLM generation |
| Cohere API Key | — | Optional — for production-quality reranking |

> **No API keys?** Set `EMBEDDING_PROVIDER=huggingface` and `LLM_PROVIDER=anthropic` (or use a local Ollama model). The system works fully without paid APIs using HuggingFace models for embeddings and a local cross-encoder for reranking — ideal for development.

---

## 🚀 Quick Start

### Option A: Full Docker Stack (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/banking-rag-pipeline.git
cd banking-rag-pipeline

# 2. Configure environment
cp .env.example .env
# Open .env and set at minimum:
#   OPENAI_API_KEY=sk-...
#   JWT_SECRET_KEY=your-random-32-char-secret

# 3. Start all infrastructure (Qdrant + Redis + Postgres + Prometheus + Grafana + API)
docker-compose up -d

# Wait ~30 seconds for services to become healthy
docker-compose ps

# 4. Ingest the sample banking policy document
python scripts/ingest_documents.py \
  --source data/sample/ \
  --collection home_lending_docs \
  --access-level internal

# 5. Authenticate and get a token
curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "officer@bank.com", "password": "any"}' \
  | python3 -c "import sys,json; print('TOKEN:', json.load(sys.stdin)['access_token'][:40]+'...')"

# 6. Query the RAG pipeline
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the minimum credit score for an FHA loan?",
    "user_role": "loan_officer"
  }'
```

### Option B: Local Development (No Docker)

```bash
# 1. Clone and enter
git clone https://github.com/your-org/banking-rag-pipeline.git
cd banking-rag-pipeline

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# .\venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # PII NLP model

# 4. Start Qdrant only (minimum required)
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant:v1.11.3

# 5. Configure environment
cp .env.example .env
# Minimum required settings:
#   OPENAI_API_KEY=sk-...
#   QDRANT_HOST=localhost
#   JWT_SECRET_KEY=dev-secret-minimum-32-characters-long

# 6. Launch the API
uvicorn src.api.main:app --reload --port 8000 --log-level info

# 7. Open interactive API docs
open http://localhost:8000/docs
```

### Verify Everything Works

```bash
# Health check
curl http://localhost:8000/health
# → {"status": "healthy", "service": "bankrag-api"}

# Readiness check (all dependencies)
curl http://localhost:8000/ready
# → {"status": "ready", "checks": {"vector_store": true, "llm": true, ...}}

# Run test suite
pytest tests/ -v

# Run evaluation
python scripts/run_evaluation.py --no-fail
```

---

## ⚙️ Configuration Reference

All configuration is managed through environment variables. Copy `.env.example` to `.env`.

### Required Settings

```bash
# ── LLM Provider ─────────────────────────────────────────────────
OPENAI_API_KEY=sk-your-openai-api-key-here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o

# ── Security (MUST change before production) ─────────────────────
JWT_SECRET_KEY=your-cryptographically-random-secret-minimum-32-chars

# ── Vector Database ───────────────────────────────────────────────
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Optional but Recommended

```bash
# ── Reranking (significantly improves response quality) ───────────
COHERE_API_KEY=your-cohere-key
RERANKER_MODEL=rerank-english-v3.0

# ── Caching (reduces latency 30-50% for repeated queries) ─────────
REDIS_HOST=localhost
REDIS_PASSWORD=your-redis-password

# ── Audit Database (required for GLBA compliance) ─────────────────
DATABASE_URL=postgresql+asyncpg://bankrag:password@localhost:5432/bankrag_db
ENABLE_AUDIT_LOG=true
```

### Retrieval Tuning

```bash
RETRIEVAL_TOP_K=10         # Candidates from vector + BM25 search
RERANK_TOP_N=5             # Final chunks after cross-encoder reranking
HYBRID_ALPHA=0.7           # Dense weight: 0.7 dense + 0.3 sparse (banking-optimized)
MIN_SIMILARITY_SCORE=0.65  # Drop results below this threshold
MAX_CONTEXT_TOKENS=8000    # Context budget passed to LLM
CHUNK_SIZE=512             # Characters per chunk (512 works for banking paragraphs)
CHUNK_OVERLAP=50           # Overlap to preserve cross-chunk context
```

> **Tuning Tips:**
> - Dense policy docs → try `CHUNK_SIZE=600`
> - FAQ-style docs → try `CHUNK_SIZE=300`
> - Legal/regulatory text → try `HYBRID_ALPHA=0.5` (equal weight to exact term matching)
> - General customer queries → default `HYBRID_ALPHA=0.7` works best

---

## 📖 Module Deep-Dive

### 1. Document Ingestion

**`src/ingestion/document_loader.py`** — Loads and validates banking documents.

**Supported formats:**

| Format | Use Case | Loader |
|--------|----------|--------|
| `.pdf` | Policy manuals, compliance documents | pypdf |
| `.docx` | Underwriting guidelines, scripts | python-docx |
| `.txt` | Regulatory text, FAQs | native |
| `.xlsx` | Rate tables, fee schedules | openpyxl |

**Security validations on every file:**
1. Extension whitelist check (`.pdf`, `.docx`, `.txt`, `.xlsx`)
2. MIME type verification (defense against extension spoofing)
3. File size limit (configurable, default 50MB)
4. SHA-256 hash computed for deduplication

**Auto-detection features:**
- **Document category** — Analyzes content to assign: `rate_sheet`, `compliance_policy`, `loan_product`, `underwriting_guideline`, `fee_schedule`, or `faq_guide`
- **Regulatory tags** — Detects `RESPA`, `TILA`, `ECOA`, `FCRA`, `HMDA`, `GLBA`, `BSA` mentions automatically

```python
from src.ingestion.document_loader import DocumentLoader

loader = DocumentLoader()
doc = loader.load_file(
    "data/fha_guidelines.pdf",
    access_level="internal",          # Security classification
    effective_date="2024-01-01",
    department="Home Lending Ops",
)

print(doc.doc_id)                      # "doc_a3f9b2c1..."  (SHA-256 derived)
print(doc.metadata.document_category)  # "loan_product"
print(doc.metadata.regulatory_tags)    # ["TILA", "RESPA"]
```

---

**`src/ingestion/chunker.py`** — Splits documents into embedding-ready chunks.

The `BankingTextSplitter` uses a recursive separator hierarchy that respects natural document structure:

```
Priority 1: \n\n\n  →  Major section breaks
Priority 2: \n\n    →  Paragraph breaks
Priority 3: \n      →  Line breaks
Priority 4: ". "    →  Sentence boundaries
Priority 5: "; "    →  Clause boundaries
Priority 6: ", "    →  Phrase boundaries
Priority 7: " "     →  Word boundaries
Priority 8: ""      →  Character level (last resort)
```

Every chunk carries full provenance metadata: `doc_id`, `source_file`, `access_level`, `section_title`, `regulatory_tags`, `char_start`, `char_end`, `token_estimate`.

---

### 2. Embedding Layer

**`src/embeddings/embedder.py`** — Creates dense vector representations.

```python
from src.embeddings.embedder import EmbedderFactory

# OpenAI (production)
embedder = EmbedderFactory.create(
    provider="openai",
    model="text-embedding-3-large",   # 3072 dimensions
)

# HuggingFace (free / offline)
embedder = EmbedderFactory.create(
    provider="huggingface",
    model="BAAI/bge-large-en-v1.5",   # 1024 dimensions, CPU/GPU
)

# Batch embedding for ingestion
from src.embeddings.embedder import DocumentEmbedder
doc_embedder = DocumentEmbedder(embedder, batch_size=50)
embedded_pairs = await doc_embedder.embed_chunks(chunks)
# Returns: [(DocumentChunk, [float, ...]), ...]
```

---

**`src/embeddings/vector_store.py`** — Qdrant integration with RBAC-enforced search.

**HNSW configuration for production:**
- `m=16` — edges per node (higher = better recall, more memory)
- `ef_construct=200` — build quality (higher = better index, slower build)
- Payload indexes on `access_level`, `document_category`, `regulatory_tags`, `doc_id` for O(1) filtering

```python
# Search always enforces access_level — cannot be bypassed
results = await store.search(
    query_vector=embedding,
    top_k=10,
    access_levels=["public", "internal"],   # From user's JWT role
    document_categories=["loan_product"],    # Optional filter
)
```

---

### 3. Hybrid Retrieval

**`src/retrieval/retriever.py`** — Combines dense + sparse search for best of both worlds.

**Why hybrid search matters for banking:**

| Query | Dense Only | BM25 Only | Hybrid |
|-------|-----------|-----------|--------|
| "first-time buyer requirements" | ✅ | ❌ | ✅ |
| "FICO 580 3.5% down payment" | ❌ | ✅ | ✅ |
| "regulation z section 1026.19" | ❌ | ✅ | ✅ |
| "how does mortgage insurance work" | ✅ | ❌ | ✅ |
| "DTI 43 percent conventional" | ⚠️ | ✅ | ✅ |

**Reciprocal Rank Fusion formula:**
```
RRF(document) = Σ weight_i / (60 + rank_i)

where weight_dense = 0.7 and weight_bm25 = 0.3
```

The constant `60` prevents highly-ranked items from dominating — a document ranked #1 in dense and #5 in BM25 will beat a document ranked #2 in dense only.

```python
results, meta = await retriever.retrieve_with_metadata(
    query="What documents do I need for a VA loan?",
    user_role="loan_officer",      # Automatically maps to access levels
    top_k=10,
    use_hybrid=True,
)
# Access levels for loan_officer: ["public", "internal"]
```

---

### 4. Reranking

**`src/retrieval/reranker.py`** — Cross-encoder reranking for precision.

Initial retrieval maximizes **recall** (don't miss relevant docs). Reranking maximizes **precision** (put the best docs first). Cross-encoders jointly process (query, document) pairs and are 10-20% more accurate than bi-encoders for relevance ranking — at the cost of higher latency, which is acceptable since we're only reranking 10-20 candidates.

```python
from src.retrieval.reranker import RerankerFactory

reranker = RerankerFactory.create(provider="cohere", top_n=5)   # Production
reranker = RerankerFactory.create(provider="local", top_n=5)    # Development (free)

reranked = await reranker.rerank(
    query="minimum FHA credit score",
    results=retrieval_results,   # 10 candidates in
    top_n=5,                     # 5 best results out
)
```

---

### 5. Generation Engine

**`src/generation/generator.py`** — LLM generation with banking-specific constraints.

**Key design decisions:**
- **Temperature 0.1** — Near-deterministic for factual banking answers
- **System prompt enforces citations** — Every answer must include `[Source: filename]`
- **Groundedness check** — Post-generation heuristic verifies answer terms appear in context
- **Retry with exponential backoff** — 3 attempts on API errors (1s → 2s → 4s)
- **Token usage tracking** — Every generation records prompt/completion tokens + cost estimate

```python
generator = GeneratorFactory.create(provider="openai", model="gpt-4o")

result = await generator.generate(messages, system_prompt, context)
print(result.answer)             # Answer with [Source: ...] citations
print(result.is_grounded)        # True — grounded in retrieved context
print(result.total_tokens)       # 847
print(result.cost_estimate_usd)  # 0.00425
```

---

**`src/generation/prompts.py`** — Role-differentiated system prompts.

Each role receives a different system prompt that controls language complexity, jargon level, disclaimer requirements, and what kinds of guidance are appropriate:

| Aspect | customer | loan_officer | underwriter | compliance_officer |
|--------|----------|-------------|-------------|-------------------|
| Language | Simple, empathetic | Technical | Technical | Regulatory |
| Jargon | Explained | Allowed | Full | Citation-heavy |
| Disclaimer | Always | As needed | Minimal | Not required |
| Focus | "What do I need to do?" | Full guidelines | Risk thresholds | Exam-ready |

---

### 6. Guardrails

**`src/guardrails/pii_redactor.py`** — Banking-grade PII protection.

Uses a **two-layer approach** for maximum coverage:
1. **Regex patterns** — Structured PII: SSN, credit cards, account numbers, routing numbers
2. **Microsoft Presidio** — Unstructured PII: names, addresses (via spaCy NLP)

**13 entity types across 3 risk tiers:**

| Risk Level | Entity Types | Action |
|-----------|-------------|--------|
| **HIGH** | SSN, Credit Card, Account#, Routing#, Loan#, Tax ID, Driver's License | Always redact |
| **MEDIUM** | DOB, Phone, Email | Redact by default |
| **LOW** | Name, Address, IP Address | Redact based on config |

**Block policy:** Queries containing 3+ HIGH-risk entities are blocked entirely — they likely contain raw financial data that should never be processed by an AI system.

---

**`src/guardrails/compliance_checker.py`** — Regulatory violation detection.

Scans both queries (input) and LLM responses (output) for violations across 4 categories:

| Regulation | Violation Type | Severity | Action |
|-----------|---------------|---------|--------|
| ECOA (Reg B) | Discriminatory language re: protected class | CRITICAL | Block + Alert |
| TILA (Reg Z) | Guaranteed specific interest rate | HIGH | Block |
| UDAAP | "Guaranteed approval", "no fees ever", "lowest rate" | HIGH | Block |
| Unauthorized Advice | Tax/legal advice beyond scope | MEDIUM | Disclaim |
| Rate Mentions | Any rate discussion | LOW | Auto-add TILA disclaimer |

Every violation includes: `violation_type`, `severity`, `matched_text`, `regulation` (with CFR citation), and `recommendation`.

---

### 7. Security & Auth

**`src/security/auth.py`** — JWT authentication with 5-level RBAC.

**Token lifecycle:**
```
Login → Access Token (30 min, HS256) + Refresh Token (7 days)
Request → Authorization: Bearer <token> → JWT verify → Role extract → RBAC check
Logout → JTI added to in-memory revocation set (Redis in production)
```

**RBAC access matrix:**

| Role | Public | Internal | Confidential | Restricted | Rate Limit | Context Chunks |
|------|:------:|:--------:|:------------:|:----------:|:----------:|:--------------:|
| `customer` | ✅ | ❌ | ❌ | ❌ | 10/min | 3 |
| `loan_officer` | ✅ | ✅ | ❌ | ❌ | 60/min | 5 |
| `underwriter` | ✅ | ✅ | ✅ | ❌ | 60/min | 7 |
| `compliance_officer` | ✅ | ✅ | ✅ | ✅ | 100/min | 10 |
| `admin` | ✅ | ✅ | ✅ | ✅ | 200/min | 10 |

**Critical security detail:** The access level filter is enforced directly in the Qdrant query payload filter — not just at the API layer. Even a compromised API token cannot access documents above the user's clearance level, because the vector database itself rejects those results.

---

**`src/security/audit_logger.py`** — GLBA-compliant immutable audit trail.

Every event is:
1. Assigned a UUID `event_id`
2. Timestamped in UTC ISO 8601
3. Assigned an HMAC-SHA256 signature (using `JWT_SECRET_KEY`) for tamper detection
4. Written synchronously to structured logs (and optionally PostgreSQL)

**Logged for every query:**
- `user_id`, `user_role`, `session_id`, `ip_address`
- `query_preview` (first 100 chars, **PII already redacted**)
- `documents_accessed` (list of doc IDs)
- `pii_types_found`, `compliance_violations`
- `was_blocked`, `block_reason`
- `latency_ms`, `tokens_used`, `cost_usd`
- `hmac_signature`

> **Retention:** Banking regulations (GLBA, OCC guidelines) require minimum 5-year audit log retention. Configure your log rotation and archiving accordingly.

---

### 8. Monitoring & Evaluation

**`src/monitoring/metrics.py`** — Prometheus instrumentation.

Access live metrics: `http://localhost:9090` (Prometheus) · `http://localhost:3000` (Grafana, admin/bankrag-grafana-pass)

Key metrics exposed:

```
# Latency
bankrag_query_duration_seconds{user_role, query_category, cache_hit}
bankrag_retrieval_duration_seconds{retrieval_type}
bankrag_generation_duration_seconds{model, user_role}

# Quality
bankrag_retrieval_score{user_role}
bankrag_faithfulness_score
bankrag_hallucination_detected_total{model, user_role}

# Security
bankrag_pii_detections_total{pii_type, location}
bankrag_compliance_violations_total{violation_type, severity, regulation}
bankrag_blocked_queries_total{block_reason}
bankrag_auth_failures_total{failure_type}

# Business
bankrag_queries_total{user_role, document_category, success}
bankrag_token_usage_total{model, token_type}
bankrag_cost_usd_total{model, operation}
```

---

## 📡 API Reference

Full interactive documentation: **`http://localhost:8000/docs`** (Swagger UI) or **`http://localhost:8000/redoc`** (ReDoc).

### Demo Credentials

| Email | Role | Access Level |
|-------|------|-------------|
| `customer@bank.com` | customer | Public only |
| `officer@bank.com` | loan_officer | Public + Internal |
| `underwriter@bank.com` | underwriter | + Confidential |
| `compliance@bank.com` | compliance_officer | All documents |
| `admin@bank.com` | admin | All + system |

Any password is accepted in the demo. Replace with real auth in production.

---

### `POST /api/v1/auth/login`

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "officer@bank.com", "password": "any"}'
```

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user_role": "loan_officer"
}
```

---

### `POST /api/v1/query` — Main RAG Endpoint

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | ✅ | Natural language question (5-1000 chars) |
| `user_role` | enum | ❌ | Override role for context (default: token role) |
| `session_id` | string | ❌ | For multi-turn conversations |
| `conversation_history` | array | ❌ | Previous turns, max 6 |
| `loan_type_filter` | enum | ❌ | Filter to: fha, va, conventional, jumbo, heloc |
| `stream` | bool | ❌ | Enable SSE streaming (default: false) |

**Example request:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the minimum credit score for an FHA loan?",
    "user_role": "loan_officer",
    "loan_type_filter": "fha"
  }'
```

**Example response:**
```json
{
  "query_id": "q_f3a9b2c1d4e5",
  "question": "What is the minimum credit score for an FHA loan?",
  "answer": "The minimum FICO credit score for an FHA loan is **580** for a 3.5% down payment. Borrowers with scores between 500-579 may qualify with a 10% minimum down payment. Scores below 500 are not eligible. [Source: home_lending_policy.txt, Section: FHA Loan Requirements]\n\n*For personalized guidance, please speak with your loan officer.*",
  "sources": [
    {
      "source_file": "home_lending_policy.txt",
      "document_category": "loan_product",
      "section_title": "FHA Loan Requirements",
      "regulatory_tags": [],
      "relevance_score": 0.941,
      "chunk_preview": "The minimum FICO credit score for FHA loan approval with a 3.5% down payment is 580..."
    }
  ],
  "guardrails": {
    "pii_detected": false,
    "pii_redacted": false,
    "compliance_passed": true,
    "disclaimers_added": false,
    "is_grounded": true,
    "pii_types_found": [],
    "compliance_violations": []
  },
  "performance": {
    "total_latency_ms": 1247.3,
    "retrieval_latency_ms": 287.1,
    "generation_latency_ms": 832.4,
    "tokens_used": 872,
    "prompt_tokens": 740,
    "completion_tokens": 132,
    "estimated_cost_usd": 0.00448,
    "cache_hit": false,
    "documents_retrieved": 10,
    "documents_after_rerank": 5
  },
  "session_id": null,
  "timestamp": "2024-11-15T10:30:15.234Z",
  "model_used": "gpt-4o",
  "user_role": "loan_officer"
}
```

---

### `POST /api/v1/ingest` — Upload Document

Requires `loan_officer` role or above.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@docs/fha_guidelines_2024.pdf" \
  -F "access_level=internal" \
  -F "effective_date=2024-01-01" \
  -F "department=Home Lending Ops"
```

---

### `POST /api/v1/feedback`

```bash
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "q_f3a9b2c1d4e5",
    "rating": 5,
    "helpful": true,
    "accurate": true,
    "comment": "Exact guideline cited, very helpful."
  }'
```

---

### System Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | None | Liveness probe |
| `/ready` | GET | None | Readiness probe (checks all deps) |
| `/metrics` | GET | None | Prometheus metrics scrape |
| `/docs` | GET | None | Swagger UI |
| `/api/v1/collection/stats` | GET | Admin | Vector store statistics |
| `/api/v1/evaluate` | POST | Compliance/Admin | Run RAGAS evaluation |

---

## 🛡️ Security & Compliance

### Defense-in-Depth Security Model

```
Layer 1  Network      TLS 1.3 in transit, Docker network isolation between services
Layer 2  Auth         JWT (HS256), 30-min access tokens, revocation support via JTI
Layer 3  RBAC         5 roles, enforced at API layer AND at vector store query layer
Layer 4  Rate Limit   Per-user sliding window (60s), per-role configurable limits
Layer 5  Input Guard  PII redaction + compliance screening before any processing
Layer 6  Retrieval    Access level payload filter in Qdrant (cannot be bypassed)
Layer 7  Output Guard PII scan + compliance check on every LLM response
Layer 8  Audit Trail  Immutable HMAC-signed event log, every action traceable
Layer 9  Encryption   AES-256 at rest (Qdrant), secrets only in env vars
```

### Regulatory Compliance Coverage

| Regulation | Full Name | Scope | Implementation |
|-----------|-----------|-------|---------------|
| **RESPA** | Real Estate Settlement Procedures Act | Settlement disclosures | Anti-kickback language detection |
| **TILA** | Truth in Lending Act (Regulation Z) | Rate/APR disclosure | Rate guarantee blocking + auto-disclaimer |
| **ECOA** | Equal Credit Opportunity Act (Regulation B) | Anti-discrimination | Protected class language detection |
| **FCRA** | Fair Credit Reporting Act | Credit report usage | Confidentiality enforcement in responses |
| **HMDA** | Home Mortgage Disclosure Act | Reporting | Regulatory tag awareness in responses |
| **GLBA** | Gramm-Leach-Bliley Act | Data privacy | PII protection pipeline + audit logging |
| **BSA/AML** | Bank Secrecy Act | Anti-money laundering | Document tagging + awareness |

---

## 📊 Performance Metrics

### Production Quality Gates

All thresholds enforced by `scripts/run_evaluation.py` — pipeline exits with code 1 (blocks deployment) if any fail.

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Faithfulness** | ≥ 0.85 | RAGAS LLM-as-judge: are all claims in the answer supported by retrieved context? |
| **Answer Relevancy** | ≥ 0.80 | RAGAS embedding similarity between question and answer |
| **Context Precision** | ≥ 0.75 | RAGAS: are retrieved chunks actually relevant? |
| **Context Recall** | ≥ 0.80 | RAGAS: did we retrieve all relevant chunks? |
| **Compliance Pass Rate** | ≥ 99.0% | Compliance checker applied to all eval responses |
| **PII Leak Rate** | ≤ 0.10% | PII scanner on final responses — no raw PII should ever reach users |
| **Hallucination Rate** | ≤ 2.00% | Groundedness check: non-grounded claims / total claims |
| **P95 Latency** | ≤ 3,000 ms | End-to-end from request received to response sent |
| **Citation Rate** | ≥ 85% | Presence of `[Source: ...]` attribution in responses |

### Cost Benchmarks (at 10,000 queries/day)

| Component | Model | Cost per Query | Daily Cost |
|-----------|-------|---------------|-----------|
| Query embedding | text-embedding-3-large | ~$0.00013 | ~$1.30 |
| LLM generation | GPT-4o | ~$0.0042 | ~$42.00 |
| Reranking | Cohere Rerank v3 | ~$0.0001 | ~$1.00 |
| **Total** | | **~$0.0044** | **~$44/day** |

---

## 🧪 Running Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report (opens HTML report)
pytest tests/ -v --cov=src --cov-report=html && open htmlcov/index.html

# Run critical safety tests only (fast, ~5 seconds)
pytest tests/test_guardrails.py -v

# Run a single test class
pytest tests/test_guardrails.py::TestPIIRedactor -v

# Run a single test
pytest tests/test_guardrails.py::TestPIIRedactor::test_ssn_detection -v

# Run ingestion tests
pytest tests/test_ingestion.py -v
```

### Coverage Targets by Module

| Module | Required Coverage | Why |
|--------|-----------------|-----|
| `guardrails/` | ≥ 95% | Critical safety code — every edge case must be tested |
| `security/` | ≥ 90% | Auth bugs are security vulnerabilities |
| `ingestion/` | ≥ 85% | Data quality gate for the entire pipeline |
| `generation/` | ≥ 80% | Correctness of LLM interaction |
| `api/` | ≥ 75% | Integration surface |

### Tests Must Pass Before Deployment

```bash
# These are blocking CI gates
pytest tests/test_guardrails.py -v        # PII + compliance safety
pytest tests/test_ingestion.py -v         # Document loading + chunking
python scripts/run_evaluation.py --samples 50  # RAGAS quality gates
```

---

## 📈 Evaluation with RAGAS

### What is RAGAS?

[RAGAS](https://docs.ragas.io/) (Retrieval Augmented Generation Assessment) is an evaluation framework that uses an LLM-as-judge to automatically assess RAG pipeline quality across four dimensions: faithfulness, answer relevancy, context precision, and context recall.

### Built-in Banking Evaluation Dataset

Shipped in `src/monitoring/evaluator.py` with 5 expert-curated banking questions:

| ID | Question | Category |
|----|----------|---------|
| eval_001 | Minimum credit score for FHA loan? | Eligibility |
| eval_002 | Documents required for conventional loan? | Process |
| eval_003 | Maximum DTI for conventional loan? | Eligibility |
| eval_004 | RESPA disclosures within 3 business days? | Compliance |
| eval_005 | Current conforming loan limits? | Rates |

### Running Evaluation

```bash
# Standard run — 5 samples, saves JSON + report
python scripts/run_evaluation.py

# Extended run for CI/CD — 50 samples
python scripts/run_evaluation.py --samples 50

# CI/CD: exits with code 1 if below threshold
python scripts/run_evaluation.py --samples 50
echo "Exit code: $?"    # 0 = pass, 1 = fail → blocks deployment

# Just see the report, no file saving
python scripts/run_evaluation.py --no-save --no-fail
```

### Sample Report Output

```
╔══════════════════════════════════════════════════════════════════╗
║           BankRAG Pipeline Evaluation Report                     ║
╚══════════════════════════════════════════════════════════════════╝

Overall Status: ✅ PASS
Overall Score:  0.852 / 1.000
Model:          gpt-4o

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAGAS QUALITY METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Faithfulness:       0.891  (target ≥ 0.850) ✅
  Answer Relevancy:   0.834  (target ≥ 0.800) ✅
  Context Precision:  0.776  (target ≥ 0.750) ✅
  Context Recall:     0.812  (target ≥ 0.800) ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BANKING SAFETY METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Compliance Pass Rate: 1.000  (target ≥ 0.990) ✅
  PII Leak Rate:        0.0000 (target ≤ 0.001) ✅
  Hallucination Rate:   0.000  (target ≤ 0.020) ✅
  Citation Rate:        0.800

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Avg Latency:  1,300 ms
  P95 Latency:  2,100 ms  (target ≤ 3,000 ms) ✅
  Avg Cost:     $0.0042/query
```

### CI/CD Integration

```yaml
# .github/workflows/quality-gate.yml
name: RAG Quality Gate
on: [push, pull_request]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run RAG Evaluation
        run: python scripts/run_evaluation.py --samples 50
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          QDRANT_HOST: localhost
        # Exits with code 1 on failure → blocks PR merge
```

---

## 🚢 Deployment Guide

### Pre-Deployment Checklist

- [ ] `JWT_SECRET_KEY` is cryptographically random (32+ chars)
- [ ] `DEBUG=false` and `ENVIRONMENT=production`
- [ ] Qdrant configured with API key authentication
- [ ] Redis password configured
- [ ] CORS origins restricted to actual domains
- [ ] TLS termination configured at load balancer
- [ ] All evaluation gates pass (`python scripts/run_evaluation.py`)
- [ ] Audit log rotation configured (5-year retention for GLBA)
- [ ] Alertmanager configured for critical compliance violations
- [ ] Grafana admin password changed from default

### Production Start

```bash
# Build and start
docker-compose -f docker-compose.yml up -d --build

# Scale API horizontally (Qdrant and Redis are shared)
docker-compose up --scale bankrag=4 -d

# Health check all services
docker-compose ps
curl http://localhost:8000/ready
```

### Performance Tuning for Scale

```bash
# For > 1M vectors — increase HNSW precision
# In vector_store.py: hnsw_config=HnswConfigDiff(m=32, ef_construct=400)

# Reduce embedding cost (1536 dims = 50% cheaper, ~3% quality loss)
EMBEDDING_DIMENSION=1536
EMBEDDING_MODEL=text-embedding-3-large   # Still use large, just truncate

# Switch to smaller embedding model (85% cheaper)
EMBEDDING_MODEL=text-embedding-3-small

# Speed up with Redis cache (reduces LLM calls for repeated queries)
REDIS_HOST=your-redis-host
REDIS_PASSWORD=your-password
```

---

## 🔧 Troubleshooting

**Qdrant connection refused:**
```bash
docker-compose ps qdrant          # Is it running?
curl http://localhost:6333/health  # Is it healthy?
docker-compose logs qdrant         # Any errors?
```

**"No relevant documentation found" on all queries:**
```bash
# Check if documents are actually ingested
curl http://localhost:8000/api/v1/collection/stats \
  -H "Authorization: Bearer $ADMIN_TOKEN"
# points_count should be > 0

# Re-ingest with dry-run to debug
python scripts/ingest_documents.py --source data/sample/ --dry-run
```

**High latency (> 5 seconds):**
```bash
# Check Prometheus to find the slow stage:
# bankrag_retrieval_duration_seconds → Qdrant is slow (check disk I/O)
# bankrag_generation_duration_seconds → LLM API is slow (check OpenAI status)
# bankrag_embedding_duration_seconds → Embedding API slow

# Quick wins:
# 1. Enable Redis caching
# 2. Reduce RETRIEVAL_TOP_K from 10 to 5
# 3. Use local cross-encoder (eliminates Cohere API round-trip)
```

**JWT token expired (401 errors):**
```bash
# Increase expiry for development
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=120

# Or re-login to get a fresh token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d '{"email":"officer@bank.com","password":"any"}'
```

**PII false positives blocking valid queries:**
```bash
# Reduce sensitivity — increase min_confidence in pii_redactor.py:
# Change: min_confidence=0.75  →  min_confidence=0.85

# Or raise the block threshold:
# Change: should_block() condition from >= 3 to >= 5 HIGH-risk entities
```

**Compliance violations on valid responses:**
```bash
# Check which rule is triggering
result = checker.check_response(answer)
print([(v.violation_type, v.matched_text) for v in result.violations])
# Review the UDAAP/TILA patterns in compliance_checker.py
# Some financial terms may need to be whitelisted for your institution
```

---

## 📁 Quick Reference

| Task | File | What to change |
|------|------|---------------|
| Add new document format | `src/ingestion/document_loader.py` | Add to `LOADERS` dict |
| Change chunking strategy | `src/ingestion/chunker.py` | Modify `BankingTextSplitter.separators` |
| Add a new PII entity | `src/guardrails/pii_redactor.py` | Add to `PII_PATTERNS` list |
| Add a compliance rule | `src/guardrails/compliance_checker.py` | Add regex to pattern lists |
| Change role permissions | `src/security/auth.py` | Modify `ROLE_ACCESS_LEVELS` |
| Add a new API endpoint | `src/api/routes.py` | Add route handler |
| Add a Prometheus metric | `src/monitoring/metrics.py` | Add to `BankRAGMetrics.__init__` |
| Change system prompt | `src/generation/prompts.py` | Modify `ROLE_SYSTEM_PROMPTS` |
| Add evaluation questions | `src/monitoring/evaluator.py` | Append to `BANKING_EVAL_DATASET` |
| Tune retrieval weighting | `src/retrieval/retriever.py` | Modify `DENSE_WEIGHT` / `SPARSE_WEIGHT` |

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Setup

```bash
git clone https://github.com/your-org/banking-rag-pipeline.git
cd banking-rag-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # Fill in your API keys
```

### Contribution Process

1. **Fork** and create a branch: `git checkout -b feature/your-feature-name`
2. **Write tests** — especially for guardrails (95% coverage required)
3. **Run tests:** `pytest tests/ -v --cov=src`
4. **Run evaluation:** `python scripts/run_evaluation.py --no-fail`
5. **Update docs** — update this README if adding new modules or changing behavior
6. **Submit PR** — include a clear description and link to any related issues

### Areas We'd Love Help With

- 🌐 **Multi-language support** — Spanish/Mandarin for diverse banking customers
- 🔄 **Real-time rate ingestion** — Webhook integration for live rate updates
- 📊 **Pre-built Grafana dashboards** — JSON configs for one-click import
- 🧪 **Larger eval dataset** — More banking Q&A pairs with ground truth answers
- 🔌 **Additional LLM providers** — Azure OpenAI, AWS Bedrock, Google Vertex AI
- 📱 **Client SDKs** — Python, JavaScript, iOS, Android
- 🏗️ **Kubernetes manifests** — Helm charts for K8s deployment

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with outstanding open-source tools:

- [FastAPI](https://fastapi.tiangolo.com/) — The API framework
- [Qdrant](https://qdrant.tech/) — Vector database
- [RAGAS](https://docs.ragas.io/) — RAG evaluation framework
- [Microsoft Presidio](https://microsoft.github.io/presidio/) — PII detection
- [rank_bm25](https://github.com/dorianbrown/rank_bm25) — BM25 implementation
- [Cohere](https://cohere.com/) — Reranking API
- [OpenAI](https://openai.com/) — Embeddings and generation
- [Anthropic](https://anthropic.com/) — Claude generation alternative

---

<div align="center">

**Built for the banking industry. Designed with security, compliance, and accuracy as first-class concerns.**

*Every query audited. Every response grounded. Every customer protected.*

[⭐ Star this repo](https://github.com/your-org/banking-rag-pipeline) · [🐛 Report a Bug](https://github.com/your-org/banking-rag-pipeline/issues) · [💡 Request a Feature](https://github.com/your-org/banking-rag-pipeline/issues)

</div>
