"""Postgres access for the retrieval layer.

Connection settings come from ``PROVENANCE_DB_URL`` or the ``POSTGRES_*``
variables in the repo-root ``.env`` (host port defaults to 5433). This mirrors
``ingestion/db.py`` deliberately so the backend is runnable on its own; the
schema itself is owned and migrated by ingestion.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import psycopg
from pgvector.psycopg import register_vector

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Load the repo-root .env (no-op if python-dotenv is unavailable)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


def database_url(override: Optional[str] = None) -> str:
    if override:
        return override
    if os.environ.get("PROVENANCE_DB_URL"):
        return os.environ["PROVENANCE_DB_URL"]
    user = os.environ.get("POSTGRES_USER", "provenance")
    password = os.environ.get("POSTGRES_PASSWORD", "provenance")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5433")
    dbname = os.environ.get("POSTGRES_DB", "provenance")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def connect(override: Optional[str] = None) -> psycopg.Connection:
    """Open a connection with the pgvector adapter registered."""
    conn = psycopg.connect(database_url(override))
    register_vector(conn)
    return conn
