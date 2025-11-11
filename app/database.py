"""Simple SQLite helper for storing documents and chunks.

This module wraps the `sqlite3` standard library to provide a minimal set of
functions for creating tables, inserting documents and chunks, and fetching
records.  It intentionally avoids SQLAlchemy to keep dependencies light.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

# Name of the environment variable that defines the database location.  The
# default is relative to the project root.
DB_ENV_VAR = "DB_PATH"
DEFAULT_DB_PATH = os.getenv(DB_ENV_VAR, os.path.join(os.path.dirname(__file__), "..", "data.db"))


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialise the database.

    This function creates the database file if it does not exist and sets up
    the required tables.  It is idempotent – repeated calls will simply
    ensure the tables exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        # Enable foreign key constraints
        cur.execute("PRAGMA foreign_keys = ON;")
        # Create documents table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                content TEXT,
                created_at TEXT
            );
            """
        )
        # Create chunks table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER,
                text TEXT
            );
            """
        )
        conn.commit()


@contextmanager
def get_conn(db_path: str = DEFAULT_DB_PATH):
    """Context manager yielding a connection with foreign keys enabled."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        conn.close()


def insert_document(filename: Optional[str], content: str, db_path: str = DEFAULT_DB_PATH) -> int:
    """Insert a document and return its generated ID."""
    created_at = datetime.utcnow().isoformat()
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (filename, content, created_at) VALUES (?, ?, ?)",
            (filename, content, created_at),
        )
        doc_id = cur.lastrowid
        conn.commit()
    return doc_id


def insert_chunks(doc_id: int, chunks: Iterable[Tuple[int, str]], db_path: str = DEFAULT_DB_PATH) -> None:
    """Insert multiple chunks for a given document.

    Args:
        doc_id: The ID of the parent document.
        chunks: An iterable of `(index, text)` tuples.
    """
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.executemany(
            "INSERT INTO chunks (document_id, chunk_index, text) VALUES (?, ?, ?)",
            [(doc_id, idx, text) for idx, text in chunks],
        )
        conn.commit()


def fetch_documents(offset: int = 0, limit: int = 20, db_path: str = DEFAULT_DB_PATH) -> List[Tuple[int, str, Optional[str], str]]:
    """Fetch a paginated list of documents.

    Returns a list of `(id, filename, content, created_at)` tuples.
    """
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, filename, content, created_at FROM documents ORDER BY id LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return cur.fetchall()


def count_documents(db_path: str = DEFAULT_DB_PATH) -> int:
    """Return the total number of documents."""
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents")
        return cur.fetchone()[0]


def count_chunks(db_path: str = DEFAULT_DB_PATH) -> int:
    """Return the total number of chunks."""
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks")
        return cur.fetchone()[0]


def fetch_all_chunks(db_path: str = DEFAULT_DB_PATH) -> List[Tuple[int, str]]:
    """Return all chunks as a list of `(chunk_id, text)` tuples."""
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, text FROM chunks")
        return cur.fetchall()


def fetch_chunks_by_ids(ids: List[int], db_path: str = DEFAULT_DB_PATH) -> List[Tuple[int, int, int, str]]:
    """Fetch chunk rows by a list of IDs.

    Returns a list of `(id, document_id, chunk_index, text)` tuples ordered by
    the same sequence as the input IDs.
    """
    if not ids:
        return []
    placeholders = ",".join(["?"] * len(ids))
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT id, document_id, chunk_index, text FROM chunks WHERE id IN ({placeholders})",
            ids,
        )
        records = {row[0]: row for row in cur.fetchall()}
        # Preserve order of ids
        return [records[i] for i in ids if i in records]