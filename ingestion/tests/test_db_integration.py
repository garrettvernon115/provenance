"""Round-trip test against the local Postgres (skips if it isn't running).

Everything except the idempotent schema migration happens inside one
transaction that is rolled back, so the test leaves no rows behind.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import chunker
import db
import parsers
from fixtures import FORM4_XML

TEST_ACCESSION = "0000000000-99-999999"


def _connect_or_skip():
    db.load_env()
    try:
        conn = db.connect()
    except Exception as exc:  # noqa: BLE001 - any connection failure means skip
        pytest.skip(f"Postgres unavailable: {exc}")
    return conn


def test_schema_and_fts_roundtrip():
    conn = _connect_or_skip()
    try:
        db.apply_migrations(conn)

        parsed = parsers.parse_form4_xml(FORM4_XML.encode())
        chunks = chunker.chunk_blocks(parsed.full_text, parsed.blocks)
        meta = {
            "accession": TEST_ACCESSION,
            "cik": 9999999999,
            "company": "INTEGRATION TEST CO",
            "form": "4",
            "filed": "2026-06-08",
            "source_url": None,
            "sha256": "test-sha",
        }
        db.replace_document(
            conn, meta, parsed.full_text, parsed.parser, "raw/4/test/doc.xml", chunks
        )

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.text, c.char_start, c.char_end, d.full_text
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.accession = %s
                  AND c.ts @@ plainto_tsquery('english', 'trading plan')
                """,
                (TEST_ACCESSION,),
            )
            rows = cur.fetchall()
        assert rows, "FTS query found no chunks for the test document"
        text, start, end, full_text = rows[0]
        assert full_text[start:end] == text  # citation invariant survives the DB
    finally:
        conn.rollback()  # discard the test document entirely
        conn.close()
