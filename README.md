# AI Backend Document Q&A Microservice

This repository contains a small question–answering service built with **FastAPI**.  
The service ingests plain text or markdown files, indexes them using both a sparse BM25‑style scoring scheme and dense TF–IDF vectors, and then answers user questions via a simple retrieval‑augmented generation pipeline.  
It is intentionally lightweight: no external dependencies are downloaded at runtime and everything (documents, index and metadata) is stored in a local SQLite database.  

## Features

- **Document ingestion** – upload one or more text/markdown files or provide raw text directly.  
  Documents are stored in a SQLite database, split into small chunks (configurable via an environment variable) and prepared for indexing.
- **Indexing** – the service builds two kinds of indices across all document chunks:
  - A **BM25**‑style index that scores chunks based on term frequency and inverse document frequency.  This approach is widely used in search and information retrieval systems because it rewards terms that are rare across the corpus while down‑weighting very common terms【876197337910835†L64-L112】.
  - A **TF‑IDF vector** index built with scikit‑learn.  Each chunk is transformed into a vector in a high‑dimensional space and queries are compared against these vectors using cosine similarity.  The foundation of BM25 is the term frequency–inverse document frequency (TF‑IDF) family of techniques【876197337910835†L64-L112】, so TF‑IDF serves as a simple dense embedding in this implementation.
- **Retrieval modes** – at query time you can choose between `baseline` (pure BM25), `vector` (pure TF‑IDF) and `hybrid` (an average of both scores).
- **Answer composition** – for each question the service retrieves the top‑`k` chunks and returns the highest scoring passage as the answer along with additional context snippets.
- **Basic operations endpoints** – list all stored documents, check health and retrieve simple metrics about the index (number of documents and chunks).
- **Rate limiting and caching** – queries from the same IP are limited to a configurable number per minute, and recent query results are cached for a short period to avoid recomputing identical requests.
- **Container and unit tests** – a `Dockerfile` is provided to build the service image and run it with `uvicorn`.  The `tests` folder includes a few PyTest tests demonstrating ingestion and querying.

## Usage

### Running locally

From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn scikit-learn
uvicorn app.main:app --reload
```

Navigate to `http://localhost:8000/docs` to explore the automatically generated OpenAPI docs and test the endpoints.

### Environment variables

- `DB_PATH` – path to the SQLite database file (default: `./data.db`).
- `CHUNK_SIZE` – approximate number of tokens per chunk when splitting documents (default: `200`).
- `RATE_LIMIT_COUNT` – number of `/query` requests allowed per client IP per minute (default: `10`).
- `CACHE_TTL` – time‑to‑live in seconds for cached query results (default: `300`).

### API Endpoints

#### `POST /ingest`

Ingest one or more documents into the index.  Documents may be uploaded as files (`multipart/form-data`) or passed as raw text in the request body.  Example using `curl`:

```bash
curl -X POST "http://localhost:8000/ingest" \
     -F files=@/path/to/file1.md \
     -F files=@/path/to/file2.txt
```

#### `GET /documents`

Return a paginated list of all ingested documents.  Accepts `offset` and `limit` query parameters.

#### `POST /query`

Query the indexed documents.  The body must include a `query` string.  Optional parameters:

- `top_k` – number of chunks to return (default: `3`).
- `mode` – one of `baseline`, `vector` or `hybrid` (default: `baseline`).

The response contains an `answer` (the top‑scoring chunk) and a list of `contexts` with their scores.

#### `GET /health`

Return a simple JSON object confirming that the service is running.

#### `GET /metrics`

Return basic metrics: number of documents and number of chunks in the index.

## Sample data

The `sample_data` directory contains three markdown files used during development.  You can ingest these via the `/ingest` endpoint to populate the database quickly.

## Design decisions

Because external internet access is disabled in this environment, it is not possible to download large pretrained transformer models.  Therefore the **vector** retrieval mode uses scikit‑learn’s `TfidfVectorizer` as a simple dense embedding representation.  The service still offers a `hybrid` mode which combines BM25 scores and TF‑IDF cosine similarity to rank passages.

BM25 is implemented manually based on the standard formula, which adjusts term frequencies by document length and inverse document frequency.  The algorithm rewards terms that appear frequently in a document but rarely across the corpus【876197337910835†L64-L112】.  See `app/services/retrieval.py` for implementation details.

## Testing

To run the unit tests:

```bash
pytest -q
```

The tests perform ingestion, listing of documents and simple query operations to ensure the endpoints behave as expected.
