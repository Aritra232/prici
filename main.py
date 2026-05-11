"""
main.py — Hobby & Industry Search API (OpenAI-Powered)
========================================================
Run:
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...
    python main.py

API Endpoints:
    GET /hobbies/search?q=cycling&limit=10
    GET /industries/search?q=healthcare&limit=10
    GET /health
    GET /docs      ← Swagger UI
    GET /redoc     ← ReDoc UI

Requires: OPENAI_API_KEY environment variable
"""

import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env into environment
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from service import search_hobbies, search_industries, search_job_titles

# ---------------------------------------------------------------------------
# Startup check
# ---------------------------------------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "\n\n  OPENAI_API_KEY is not set.\n"
        "  Export it before starting:\n"
        "      export OPENAI_API_KEY=sk-...\n"
    )

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Hobby & Industry Search API",
    description=(
        "AI-powered search for hobbies and industries using OpenAI. "
        "Handles partial input ('cycl') and typos ('helthcre') automatically."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class HobbyResult(BaseModel):
    hobby: str
    reason: str


class IndustryResult(BaseModel):
    industry: str
    reason: str


class JobTitleResult(BaseModel):
    job_title: str
    reason: str


class HobbySearchResponse(BaseModel):
    query: str
    total: int
    results: list[HobbyResult]


class IndustrySearchResponse(BaseModel):
    query: str
    total: int
    results: list[IndustryResult]


class JobTitleSearchResponse(BaseModel):
    query: str
    total: int
    results: list[JobTitleResult]


class HealthResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return {"status": "ok", "message": "Hobby & Industry Search API is running."}


@app.get(
    "/hobbies/search",
    response_model=HobbySearchResponse,
    tags=["Hobbies"],
    summary="Search hobbies by name or keyword (AI-powered)",
    description="Uses OpenAI to return hobbies related to the query. Handles typos and partial input automatically.",
)
def hobby_search(
    q: str = Query(..., min_length=1, max_length=100, description="e.g. 'cycling', 'cycl', 'mountan biking'"),
    limit: int = Query(default=10, ge=1, le=50),
):
    try:
        results = search_hobbies(q, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI search failed: {str(e)}")

    if not results:
        raise HTTPException(status_code=404, detail=f"No hobbies found for: '{q}'")

    return {"query": q, "total": len(results), "results": results}


@app.get(
    "/industries/search",
    response_model=IndustrySearchResponse,
    tags=["Industries"],
    summary="Search industries by name or keyword (AI-powered)",
    description="Uses OpenAI to return industries related to the query. Handles typos and partial input automatically.",
)
def industry_search(
    q: str = Query(..., min_length=1, max_length=100, description="e.g. 'healthcare', 'helthcre', 'tech'"),
    limit: int = Query(default=10, ge=1, le=50),
):
    try:
        results = search_industries(q, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI search failed: {str(e)}")

    if not results:
        raise HTTPException(status_code=404, detail=f"No industries found for: '{q}'")

    return {"query": q, "total": len(results), "results": results}


@app.get(
    "/job-titles/search",
    response_model=JobTitleSearchResponse,
    tags=["Job Titles"],
    summary="Search job titles by name or keyword (AI-powered)",
    description="Uses OpenAI to return job titles related to the query. Handles typos and partial input automatically.",
)
def job_title_search(
    q: str = Query(..., min_length=1, max_length=100, description="e.g. 'dietitian', 'nutritionist', 'medical'"),
    limit: int = Query(default=10, ge=1, le=50),
):
    try:
        results = search_job_titles(q, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI search failed: {str(e)}")

    if not results:
        raise HTTPException(status_code=404, detail=f"No job titles found for: '{q}'")

    return {"query": q, "total": len(results), "results": results}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)