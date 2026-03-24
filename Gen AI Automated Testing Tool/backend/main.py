"""
GenAI Test Case Generator — FastAPI Backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import upload, generate, export

app = FastAPI(
    title="GenAI Test Case Generator",
    description="Generates test cases from Business Requirement Documents using LLMs + RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router,   prefix="/api/upload",   tags=["Upload"])
app.include_router(generate.router, prefix="/api/generate", tags=["Generate"])
app.include_router(export.router,   prefix="/api/export",   tags=["Export"])

@app.get("/health")
def health():
    return {"status": "ok"}
