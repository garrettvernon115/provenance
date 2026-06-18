"""Provenance API + UI (Phase 6).

One FastAPI service ties the whole system together: hybrid retrieval → trained
re-ranker → grounded cited answer, plus a source viewer and the eval dashboard.
The retrieval/re-rank/answer code is imported from the same modules the CLIs and
eval harness use — one code path everywhere.

Run locally:
    backend/.venv/Scripts/python -m uvicorn app:app --reload --port 8000
(from the backend/ directory)
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

import answer as answer_mod
import db
import reranker
import retrieval

log = logging.getLogger("backend.app")

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
EVAL_RESULTS = REPO_ROOT / "data" / "eval" / "results.json"
CANDIDATE_POOL = 50

_state: dict = {"pool": None, "answerer": None, "answer_error": None}


def _get_answerer() -> Optional[answer_mod.AnswerGenerator]:
    """Lazily build the answer generator; tolerate a missing API key."""
    if _state["answerer"] is None and _state["answer_error"] is None:
        try:
            _state["answerer"] = answer_mod.get_answer_generator()
        except Exception as exc:  # noqa: BLE001
            _state["answer_error"] = str(exc)
            log.warning("answer layer unavailable: %s", exc)
    return _state["answerer"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.load_env()
    pool = ConnectionPool(db.database_url(), configure=register_vector,
                          min_size=1, max_size=4, open=True)
    _state["pool"] = pool
    log.info("DB pool ready; re-ranker available: %s", reranker.model_available())
    try:
        yield
    finally:
        pool.close()


app = FastAPI(title="Provenance", version="1.0", lifespan=lifespan)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True


class Source(BaseModel):
    index: int
    chunk_id: int
    accession: str
    company: str
    form: str
    section: Optional[str]
    char_start: int
    char_end: int
    text: str
    rerank_score: Optional[float] = None
    cited: bool = False


class QueryResponse(BaseModel):
    query: str
    answer: Optional[str]
    answer_available: bool
    note: Optional[str]
    reranked: bool
    sources: list[Source]


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is empty")
    pool: ConnectionPool = _state["pool"]
    with pool.connection() as conn:
        candidates = retrieval.hybrid_search(conn, req.query, limit=CANDIDATE_POOL,
                                             candidate_limit=CANDIDATE_POOL)
    reranked = False
    if req.rerank and reranker.model_available():
        results = reranker.rerank(req.query, candidates, top_k=req.k)
        reranked = True
    else:
        results = candidates[: req.k]

    answerer = _get_answerer()
    note = None
    answer_text = None
    cited: set[int] = set()
    if answerer is not None:
        result = answerer.generate(req.query, results)
        answer_text = result.text
        cited = set(result.citations)
    else:
        note = ("Answer layer disabled (no ANTHROPIC_API_KEY). Retrieval and "
                "re-ranking still work — showing the cited-quality passages below.")

    sources = [
        Source(
            index=i, chunk_id=r.chunk_id, accession=r.accession, company=r.company,
            form=r.form, section=r.section, char_start=r.char_start,
            char_end=r.char_end, text=r.text, rerank_score=r.rerank_score,
            cited=(i in cited),
        )
        for i, r in enumerate(results, start=1)
    ]
    return QueryResponse(
        query=req.query, answer=answer_text, answer_available=answerer is not None,
        note=note, reranked=reranked, sources=sources,
    )


class Document(BaseModel):
    accession: str
    company: str
    form: str
    filed: str
    full_text: str


@app.get("/api/documents/{accession}", response_model=Document)
def get_document(accession: str) -> Document:
    """Full document text so the UI can show a citation in its exact context."""
    pool: ConnectionPool = _state["pool"]
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT accession, company, form, filed::text, full_text "
            "FROM documents WHERE accession = %s",
            (accession,),
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="document not found")
    return Document(accession=row[0], company=row[1], form=row[2], filed=row[3],
                    full_text=row[4])


@app.get("/api/eval")
def get_eval():
    """Serve the Phase 5 three-way comparison for the dashboard."""
    if not EVAL_RESULTS.is_file():
        raise HTTPException(status_code=404,
                            detail="no eval results yet (run eval/run_eval.py)")
    return json.loads(EVAL_RESULTS.read_text(encoding="utf-8"))


@app.get("/healthz")
def healthz():
    return {"status": "ok", "reranker": reranker.model_available(),
            "answer_layer": _get_answerer() is not None}


# --------------------------------------------------------------------------
# UI (static, server-served — no build step)
# --------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
