# GenAI Test Case Generator

An end-to-end tool that accepts a Business Requirement Document (BRD) and generates
comprehensive, traceable test cases using an LLM + RAG pipeline.

---

## Architecture overview

```
BRD file (PDF/DOCX/TXT)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 — Document ingestion                               │
│  Parser → Chunker → Metadata extractor                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   Embedding model             NER / Requirement
   (text-embedding-3           parser (actors,
    or MiniLM)                 flows, rules)
          │                           │
          ▼                           ▼
    Vector store               Prompt builder
   (ChromaDB/Pinecone)         (few-shot + CoT)
          │                           │
          └─────────────┬─────────────┘
                        ▼
              LLM (Claude Sonnet 4)
              chain-of-thought generation
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
     Functional     Edge/Negative  Non-functional
     test cases     test cases     test cases
                        │
                        ▼
              Human review portal
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
         CSV           JSON        Jira CSV
```

---

## Project structure

```
genai-test-tool/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── models/
│   │   └── schemas.py             # Pydantic models (TestCase, BRDDocument, etc.)
│   ├── routes/
│   │   ├── upload.py              # POST /api/upload
│   │   ├── generate.py            # POST /api/generate
│   │   └── export.py              # GET  /api/export/{id}?format=csv|json|jira
│   └── services/
│       ├── document_parser.py     # PDF/DOCX/HTML text extraction + chunking
│       ├── vector_store.py        # ChromaDB embeddings + RAG retrieval
│       └── llm_service.py         # Prompt engineering + Anthropic API call
│
├── frontend/
│   ├── index.html
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── package.json
│   ├── Dockerfile
│   ├── nginx.conf
│   └── src/
│       ├── main.tsx               # React entry point
│       ├── App.tsx                # Root component + page routing
│       ├── pages/
│       │   ├── UploadPage.tsx     # Drag-and-drop BRD upload + test type selector
│       │   └── ResultsPage.tsx    # Test case review table + export buttons
│       └── utils/
│           └── api.ts             # Axios client + TypeScript types
│
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Quick start (local development)

### Prerequisites
- Python 3.11+
- Node.js 20+
- An Anthropic API key → https://console.anthropic.com

### 1. Clone and configure

```bash
git clone <repo>
cd genai-test-tool
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Start the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API will be at http://localhost:8000
Interactive docs at http://localhost:8000/docs

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

---

## Docker (full stack)

```bash
cp .env.example .env
# Set your ANTHROPIC_API_KEY in .env

docker compose up --build
```

- Frontend → http://localhost:3000
- Backend API → http://localhost:8000
- API docs → http://localhost:8000/docs

---

## API reference

### POST /api/upload/
Upload a BRD file. Returns a `BRDDocument` with an `id` for subsequent calls.

**Body:** `multipart/form-data` with field `file`
**Accepted:** `.pdf`, `.docx`, `.txt`, `.md`, `.html`

```json
{
  "id": "abc123",
  "filename": "payment-service-brd.pdf",
  "content": "...",
  "chunks": ["chunk1", "chunk2"]
}
```

---

### POST /api/generate/
Generate test cases for an uploaded document.

```json
{
  "document_id": "abc123",
  "test_types": ["functional", "edge", "negative", "security"],
  "max_cases": 50,
  "model": "claude-sonnet-4-20250514"
}
```

Returns a `GenerateResponse`:
```json
{
  "document_id": "abc123",
  "total": 34,
  "coverage_gaps": ["REQ-012 lacks validation criteria"],
  "test_cases": [
    {
      "id": "TC-A1B2C3D4",
      "title": "Successful payment with valid card",
      "test_type": "functional",
      "priority": "P0 - Critical",
      "preconditions": ["User is logged in", "Cart has items"],
      "steps": [
        { "step_number": 1, "action": "Click Checkout", "expected": "Payment form shown" }
      ],
      "expected_result": "Payment processed, confirmation email sent",
      "linked_requirement": "REQ-005",
      "tags": ["payment", "checkout"],
      "status": "draft"
    }
  ]
}
```

---

### POST /api/generate/review
Submit a QA engineer's review decision.

```json
{
  "test_case_id": "TC-A1B2C3D4",
  "action": "approve",
  "feedback": null
}
```

Actions: `approve` | `reject` | `edit` (pass `edited_case` for edit)

---

### GET /api/export/{document_id}?format=csv|json|jira
Download test cases in the chosen format.
- `csv` — Standard spreadsheet, importable everywhere
- `json` — Full structured data
- `jira` — Jira bulk import CSV format

---

## Configuration

| Environment variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `OPENAI_API_KEY` | No | Use OpenAI embeddings instead of local MiniLM |

---

## Extending the tool

### Swap the LLM
In `backend/services/llm_service.py`, change the `model` parameter:
- `claude-opus-4-6` — highest quality, slower
- `claude-sonnet-4-6` — balanced (default)
- Any OpenAI model by swapping the `anthropic` client for `openai`

### Use a production vector store
In `backend/services/vector_store.py`, replace the ChromaDB client:
```python
# Pinecone example
import pinecone
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("brd-chunks")
```

### Connect to Jira via API
Replace the Jira CSV export in `export.py` with direct API calls:
```python
import requests
requests.post(
    f"{JIRA_BASE}/rest/api/3/issue",
    json={"fields": {"summary": tc.title, "issuetype": {"name": "Test"}}},
    auth=(JIRA_EMAIL, JIRA_TOKEN),
)
```

### Add fine-tuning feedback loop
Collect approved/rejected cases from the review endpoint and periodically
fine-tune a smaller model (e.g. Mistral 7B) on the approved examples using
the Anthropic fine-tuning API or any PEFT framework.

---

## Test case schema

Each generated test case follows this structure:

| Field | Type | Description |
|---|---|---|
| `id` | string | Auto-generated unique ID (TC-XXXXXXXX) |
| `title` | string | Short descriptive title |
| `test_type` | enum | functional / edge / negative / performance / security / accessibility / regression / uat |
| `priority` | enum | P0 Critical / P1 High / P2 Medium / P3 Low |
| `preconditions` | string[] | Prerequisites before test execution |
| `steps` | Step[] | Numbered steps with action + expected result |
| `expected_result` | string | Overall pass condition |
| `linked_requirement` | string | Traceability link to BRD section/ID |
| `tags` | string[] | Free-form tags for filtering |
| `status` | enum | draft / approved / rejected |


Code Understanding :-
Backend (Python / FastAPI)

<img width="907" height="618" alt="image" src="https://github.com/user-attachments/assets/856c1468-53b4-4cf4-81b8-ba8da64449e3" />





