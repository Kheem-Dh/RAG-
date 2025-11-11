"""FastAPI application entry point for the document Q&A service."""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

from fastapi import Body, Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from . import database
from .services import ingestion, rag, retrieval


# Configuration via environment variables
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data.db"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
RATE_LIMIT_COUNT = int(os.getenv("RATE_LIMIT_COUNT", "10"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))

# Initialise database and retrieval index
database.init_db(DB_PATH)
retrieval_index = retrieval.RetrievalIndex()
# Build initial index from any existing chunks
existing_chunks = database.fetch_all_chunks(DB_PATH)
retrieval_index.build(existing_chunks)

# Caches and rate limiting
_query_cache: Dict[Tuple[str, str], Tuple[float, dict]] = {}
_requests_log: Dict[str, List[float]] = {}


def check_rate_limit(client_ip: str) -> None:
    """Raise an HTTPException if the client has exceeded the rate limit."""
    now = time.time()
    history = _requests_log.setdefault(client_ip, [])
    # Drop timestamps older than one minute
    one_minute_ago = now - 60
    history[:] = [t for t in history if t > one_minute_ago]
    if len(history) >= RATE_LIMIT_COUNT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_COUNT} queries per minute.",
        )
    history.append(now)


def get_cached_result(query: str, mode: str) -> Optional[dict]:
    """Retrieve a cached query result if present and valid."""
    key = (query, mode)
    entry = _query_cache.get(key)
    if not entry:
        return None
    timestamp, result = entry
    if time.time() - timestamp > CACHE_TTL:
        # Expired
        _query_cache.pop(key, None)
        return None
    return result


def set_cached_result(query: str, mode: str, result: dict) -> None:
    """Store a query result in the cache."""
    _query_cache[(query, mode)] = (time.time(), result)


app = FastAPI(title="Document Q&A Service")


@app.post("/ingest")
async def ingest_endpoint(
    files: Optional[List[UploadFile]] = File(default=None),
    text_inputs: Optional[List[str]] = Body(default=None),
) -> JSONResponse:
    """Ingest uploaded files or raw text into the index.

    Files should be plain text or markdown.  Raw text can be provided as a list
    in the JSON body.  New documents are split into chunks and stored in the
    database and index.
    """
    if not files and not text_inputs:
        raise HTTPException(status_code=400, detail="No files or text provided.")
    ingested_docs: List[int] = []
    # Process file uploads
    if files:
        for uploaded in files:
            content_bytes = await uploaded.read()
            try:
                content = content_bytes.decode("utf-8")
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode file {uploaded.filename}: {exc}",
                )
            content = ingestion.clean_text(content)
            chunks = ingestion.chunk_text(content, CHUNK_SIZE)
            doc_id = database.insert_document(uploaded.filename, content, DB_PATH)
            database.insert_chunks(doc_id, list(enumerate(chunks)), DB_PATH)
            ingested_docs.append(doc_id)
    # Process raw text inputs
    if text_inputs:
        for text in text_inputs:
            content = ingestion.clean_text(text)
            chunks = ingestion.chunk_text(content, CHUNK_SIZE)
            doc_id = database.insert_document(None, content, DB_PATH)
            database.insert_chunks(doc_id, list(enumerate(chunks)), DB_PATH)
            ingested_docs.append(doc_id)
    # Rebuild index after ingestion
    all_chunks = database.fetch_all_chunks(DB_PATH)
    retrieval_index.build(all_chunks)
    return JSONResponse(status_code=201, content={"ingested_ids": ingested_docs})


@app.get("/documents")
async def list_documents(offset: int = 0, limit: int = 20) -> List[dict]:
    """Return a paginated list of documents."""
    docs = database.fetch_documents(offset=offset, limit=limit, db_path=DB_PATH)
    return [
        {
            "id": doc_id,
            "filename": filename,
            "content": content,
            "created_at": created_at,
        }
        for doc_id, filename, content, created_at in docs
    ]


@app.post("/query")
async def query_endpoint(
    request: Request,
    body: dict = Body(...),
) -> dict:
    """Answer a user question using the selected retrieval mode."""
    # Rate limiting
    client_ip = request.client.host if request.client else "anonymous"
    check_rate_limit(client_ip)
    # Extract parameters
    query: str = body.get("query")
    if not query or not isinstance(query, str):
        raise HTTPException(status_code=400, detail="Field 'query' must be a non‑empty string.")
    top_k: int = int(body.get("top_k", 3))
    mode: str = body.get("mode", "baseline").lower()
    if mode not in {"baseline", "vector", "hybrid"}:
        raise HTTPException(status_code=400, detail="Mode must be 'baseline', 'vector' or 'hybrid'.")
    # Check cache
    cached = get_cached_result(query, mode)
    if cached:
        return cached
    # Perform retrieval
    if mode == "baseline":
        retrieved = retrieval_index.query_bm25(query, top_k=top_k)
    elif mode == "vector":
        retrieved = retrieval_index.query_tfidf(query, top_k=top_k)
    else:
        retrieved = retrieval_index.query_hybrid(query, top_k=top_k)
    # Compose answer
    result = rag.compose_answer(query, retrieved, lambda ids: database.fetch_chunks_by_ids(ids, DB_PATH))
    # Add retrieval mode and top_k for transparency
    result["mode"] = mode
    result["top_k"] = top_k
    # Cache result
    set_cached_result(query, mode, result)
    return result


@app.get("/health")
async def health() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> dict:
    """Return basic service metrics."""
    docs_count = database.count_documents(DB_PATH)
    chunks_count = database.count_chunks(DB_PATH)
    return {"documents": docs_count, "chunks": chunks_count}