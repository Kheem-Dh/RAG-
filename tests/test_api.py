import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def setup_module(module):
    """Remove any existing database file before tests run."""
    from app import main as app_main  # noqa: import for side‑effects
    db_path = app_main.DB_PATH
    # Remove the file if it exists
    try:
        os.remove(db_path)
    except FileNotFoundError:
        pass
    # Reinitialise database
    app_main.database.init_db(db_path)
    app_main.retrieval_index.build([])


def test_ingestion_and_query():
    from app import main as app_main
    client = TestClient(app_main.app)
    # Ingest two simple documents via raw text
    payload = {
        "text_inputs": [
            "Apples are a type of fruit. They are red, green or yellow.",
            "Bananas are yellow and rich in potassium."
        ]
    }
    response = client.post("/ingest", json=payload)
    assert response.status_code == 201
    assert len(response.json().get("ingested_ids")) == 2
    # List documents
    resp_docs = client.get("/documents")
    assert resp_docs.status_code == 200
    docs_list = resp_docs.json()
    assert len(docs_list) >= 2
    # Query about apples
    query_body = {"query": "What colour are apples?", "mode": "baseline", "top_k": 2}
    resp_query = client.post("/query", json=query_body)
    assert resp_query.status_code == 200
    data = resp_query.json()
    assert "answer" in data
    assert "apples" in data["answer"].lower()
    # Metrics
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()
    assert metrics["documents"] >= 2
    assert metrics["chunks"] >= 2
    # Health check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json().get("status") == "ok"