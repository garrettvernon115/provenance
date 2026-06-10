"""Postgres access for ingestion: connection, migrations, document upserts.

Connection settings come from ``PROVENANCE_DB_URL`` or the ``POSTGRES_*``
variables in the repo-root ``.env`` (host port defaults to 5433 — see
infra/docker-compose.yml).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import psycopg

log = logging.getLogger("ingest.db")

REPO_ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = Path(__file__).resolve().parent / "sql"


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
    """Apply ingestion/sql/*.sql in name order, tracked in schema_migrations."""
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version TEXT PRIMARY KEY,"
            " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        conn.commit()
        for path in sorted(SQL_DIR.glob("*.sql")):
            cur.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s", (path.name,)
            )
            if cur.fetchone():
                continue
            log.info("applying migration %s", path.name)
            cur.execute(path.read_text(encoding="utf-8"))
            cur.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)", (path.name,)
            )
            conn.commit()


def existing_sha256(conn: psycopg.Connection, accession: str) -> Optional[str]:
    """sha256 of the already-ingested source document, or None if absent."""
    with conn.cursor() as cur:
        cur.execute("SELECT sha256 FROM documents WHERE accession = %s", (accession,))
        row = cur.fetchone()
    return row[0] if row else None


def replace_document(
    conn: psycopg.Connection,
    meta: dict,
    full_text: str,
    parser: str,
    source_file: str,
    chunks: list,
) -> int:
    """Insert (or fully replace) one document and its chunks. Caller commits."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE accession = %s", (meta["accession"],))
        cur.execute(
            """
            INSERT INTO documents
              (accession, cik, company, form, filed, source_file, source_url,
               sha256, parser, full_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                meta["accession"], meta["cik"], meta["company"], meta["form"],
                meta["filed"], source_file, meta.get("source_url"),
                meta.get("sha256"), parser, full_text,
            ),
        )
        doc_id = cur.fetchone()[0]
        cur.executemany(
            """
            INSERT INTO chunks
              (document_id, chunk_index, section, text, char_start, char_end,
               token_estimate)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (doc_id, c.chunk_index, c.section, c.text, c.char_start,
                 c.char_end, c.token_estimate)
                for c in chunks
            ],
        )
    return doc_id
