# BankRAG Architecture & Security Guide

## 🏗️ End-to-End Pipeline Flow

```
USER QUERY
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  API GATEWAY (FastAPI)                               │
│  • JWT Authentication (HS256, 30-min expiry)        │
│  • Role-Based Access Control (5 roles)              │
│  • Rate Limiting (per user/role, sliding window)    │
│  • Request ID injection for tracing                 │
│  • CORS enforcement                                  │
└─────────────────────┬────────────────────────────────┘
                      │
    ┌─────────────────▼─────────────────┐
    │  INPUT GUARDRAILS                 │
    │  ① PII Detection & Redaction      │
    │     - SSN, CC#, Account#, DOB    │
    │     - Regex + Presidio NLP        │
    │     - Block if > 3 HIGH-risk PII  │
    │  ② Compliance Check (Query)       │
    │     - ECOA/anti-discrimination    │
    │     - UDAAP screening             │
    │     - Block on critical violation │
    └─────────────────┬─────────────────┘
                      │
    ┌─────────────────▼─────────────────┐
    │  HYBRID RETRIEVAL                 │
    │  ① Dense Search (Qdrant HNSW)    │
    │     - text-embedding-3-large      │
    │     - Cosine similarity, top-20   │
    │     - Access-level filter (RBAC)  │
    │  ② Sparse Search (BM25)          │
    │     - Keyword/term matching       │
    │     - Banking terminology         │
    │  ③ Score Fusion (RRF)            │
    │     - Dense:0.7, Sparse:0.3       │
    │     - Reciprocal Rank Fusion      │
    └─────────────────┬─────────────────┘
                      │
    ┌─────────────────▼─────────────────┐
    │  CROSS-ENCODER RERANKING          │
    │  • Cohere rerank-english-v3.0     │
    │  • Joint (query, doc) scoring     │
    │  • Top-5 after rerank             │
    └─────────────────┬─────────────────┘
                      │
    ┌─────────────────▼─────────────────┐
    │  CONTEXT ASSEMBLY                 │
    │  • Source attribution headers     │
    │  • Regulatory tag metadata        │
    │  • Token budget enforcement       │
    │  • Role-appropriate context size  │
    └─────────────────┬─────────────────┘
                      │
    ┌─────────────────▼─────────────────┐
    │  LLM GENERATION (GPT-4o/Claude)   │
    │  • Role-specific system prompt    │
    │  • Citation injection prompting   │
    │  • Low temperature (0.1)          │
    │  • Anti-hallucination constraints │
    │  • Multi-turn conversation support│
    └─────────────────┬─────────────────┘
                      │
    ┌─────────────────▼─────────────────┐
    │  OUTPUT GUARDRAILS                │
    │  ① PII check on response          │
    │  ② Compliance check               │
    │     - Rate guarantee detection    │
    │     - Approval guarantee detect.  │
    │     - UDAAP deceptive check       │
    │     - ECOA discrimination check   │
    │  ③ Disclaimer injection           │
    │  ④ Groundedness check             │
    └─────────────────┬─────────────────┘
                      │
    ┌─────────────────▼─────────────────┐
    │  AUDIT & MONITORING               │
    │  • Immutable audit log (JSONL)    │
    │  • HMAC signature per event       │
    │  • Prometheus metrics             │
    │  • Grafana dashboards             │
    │  • OpenTelemetry tracing          │
    └─────────────────┬─────────────────┘
                      │
                      ▼
              STRUCTURED RESPONSE
              (answer + sources + metrics)
```

---

## 🔒 Security Architecture

### Authentication & Authorization

```
Token Flow:
  Login → JWT Access Token (30min) + Refresh Token (7 days)
  Every Request → Bearer Token → JWT Verify → User Object → RBAC Check

Role Hierarchy:
  customer          → ["public"]
  loan_officer      → ["public", "internal"]
  underwriter       → ["public", "internal", "confidential"]
  compliance_officer→ ["public", "internal", "confidential", "restricted"]
  admin             → all

RBAC is enforced at THREE layers:
  1. API route level (FastAPI Depends)
  2. Retrieval level (Qdrant filter on access_level)
  3. Context size limit per role (prevents data exfil)
```

### Data Security

```
At Rest:
  - Vector payloads encrypted by Qdrant (AES-256)
  - Audit logs cryptographically signed (HMAC-SHA256)
  - Passwords hashed with bcrypt (cost factor 12)
  - JWT secrets stored in environment variables (not code)

In Transit:
  - HTTPS/TLS 1.3 for all external traffic
  - Internal service mesh (docker network isolation)
  - Qdrant: HTTPS mode for production cloud

PII Protection (GLBA Compliance):
  - SSN, account numbers, credit cards → NEVER stored in logs
  - All queries PII-redacted before storage
  - Audit logs contain only redacted query previews
  - Customer data never cross-contaminated between sessions
```

### Regulatory Guardrails

```
ECOA (Reg B): Block any response mentioning protected class in lending context
TILA (Reg Z): Detect rate guarantees → add required disclosures
RESPA (Reg X): Prevent kickback-adjacent language
UDAAP:        Block deceptive/unfair language patterns
FCRA:         Enforce credit report confidentiality
GLBA:         PII redaction pipeline + audit logging
```

---

## 📊 Performance Targets

| Metric | Target | How Measured |
|--------|--------|-------------|
| P50 Latency | < 1.5s | Prometheus histogram |
| P95 Latency | < 3.0s | Prometheus histogram |
| P99 Latency | < 5.0s | Prometheus histogram |
| Faithfulness | > 0.85 | RAGAS (LLM-as-judge) |
| Answer Relevancy | > 0.80 | RAGAS (embedding similarity) |
| Context Precision | > 0.75 | RAGAS |
| Context Recall | > 0.80 | RAGAS |
| PII Leak Rate | < 0.1% | Output PII scan |
| Compliance Pass | > 99% | Compliance checker |
| Hallucination Rate | < 2% | Groundedness check |
| Cache Hit Rate | > 30% | Redis cache metrics |
| Throughput | 100 req/s | Load testing |

---

## 🚨 Incident Response

### High-Risk PII Detected
1. Query blocked immediately
2. User notified to contact loan officer
3. Security event logged with user_id + IP
4. Alert sent to security team (Slack/PagerDuty)

### Compliance Violation in Output
1. Response replaced with safe fallback
2. Violation logged with full context
3. Compliance team notified for CRITICAL violations
4. Affected prompts flagged for review

### Unusual Query Patterns (Potential Exfiltration)
1. Rate limiter activates
2. Account temporarily suspended (automatic)
3. Audit log reviewed by security team
4. IP blocked if pattern confirmed

---

## 🧪 Evaluation Pipeline (CI/CD)

```bash
# Run in CI/CD before deployment:
python scripts/run_evaluation.py --samples 50

# Exit codes:
# 0 = All metrics pass production bar
# 1 = One or more metrics below threshold → BLOCK DEPLOYMENT
```

Minimum gate for production deployment:
- faithfulness ≥ 0.85
- answer_relevancy ≥ 0.80  
- compliance_pass_rate ≥ 0.99
- pii_leak_rate ≤ 0.001
- p95_latency ≤ 3000ms

---

## 📦 Technology Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Vector DB | Qdrant | Best performance/cost, native payload filtering |
| Embeddings | text-embedding-3-large | Best accuracy, 3072 dims, $0.13/1M tokens |
| LLM | GPT-4o | Best reasoning, low hallucination rate |
| Sparse | BM25 (rank_bm25) | Excellent for banking terminology/regulation codes |
| Reranker | Cohere Rerank v3 | Best cross-encoder accuracy, fast API |
| Evaluation | RAGAS | Industry standard, LLM-as-judge pipeline |
| Auth | JWT (python-jose) | Stateless, production-proven |
| PII | Presidio + Regex | Layered defense, banking-specific patterns |
| Metrics | Prometheus | Industry standard, Grafana native |
| API | FastAPI | Async, auto-docs, Pydantic validation |
