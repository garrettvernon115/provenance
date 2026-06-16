"""Postgres access for the training-data pipeline.

Connection settings come from ``PROVENANCE_DB_URL`` or the ``POSTGRES_*``
variables in the repo-root ``.env`` (host port defaults to 5433), mirroring
``ingestion/db.py`` / ``backend/db.py`` so the reranker tooling runs on its own.
The ``generated_questions`` schema is owned by ingestion's migration chain
(``ingestion/sql/003_*``); we reuse its tracked runner here.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Optional

import psycopg

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
    return psycopg.connect(database_url(override))


def apply_migrations(conn: psycopg.Connection) -> None:
    """Apply ingestion's tracked migrations (loaded by path — its module name
    'db' collides with this one)."""
    path = REPO_ROOT / "ingestion" / "db.py"
    spec = importlib.util.spec_from_file_location("ingestion_db", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_migrations(conn)
