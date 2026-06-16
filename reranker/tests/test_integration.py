"""End-to-end-ish DB test: migration 003 + a question round trip + negative
mining via the live FTS retriever. Skips if Postgres is unavailable; rolls back."""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import build_training_data as btd
import db


@pytest.fixture(scope="module")
def conn():
    db.load_env()
    try:
        connection = db.connect()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres unavailable: {exc}")
    try:
        db.apply_migrations(connection)  # ensures generated_questions exists
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"migrations failed: {exc}")
    yield connection
    connection.rollback()
    connection.close()


def _first_chunk(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM chunks ORDER BY id LIMIT 1")
        row = cur.fetchone()
    if row is None:
        pytest.skip("no chunks ingested; run ingestion + embedding first")
    return row[0]


def test_generated_questions_roundtrip(conn):
    chunk_id = _first_chunk(conn)
    question = f"integration test question {uuid.uuid4().hex[:8]}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO generated_questions (chunk_id, question, model) "
            "VALUES (%s, %s, %s) RETURNING id",
            (chunk_id, question, "test-model"),
        )
        qid = cur.fetchone()[0]
        cur.execute("SELECT question FROM generated_questions WHERE id = %s", (qid,))
        assert cur.fetchone()[0] == question
    conn.rollback()  # discard the test row


def test_lexical_mining_excludes_gold(conn):
    # A distinctive nonce guarantees the seeded chunk is the top lexical hit, so
    # negative mining must then exclude it.
    nonce = f"zzqq{uuid.uuid4().hex[:6]}"
    text = f"The {nonce} clause governs unusual indemnification obligations."
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents
              (accession, cik, company, form, filed, source_file, parser, full_text)
            VALUES (%s, 0, 'NEG TEST CO', '10-K', '2026-01-01', 't', 't', %s)
            RETURNING id
            """,
            (f"NEGTEST-{uuid.uuid4().hex[:10]}", text),
        )
        doc_id = cur.fetchone()[0]
        cur.execute(
            """
            INSERT INTO chunks
              (document_id, chunk_index, section, text, char_start, char_end)
            VALUES (%s, 0, 'Item 1A', %s, 0, %s) RETURNING id
            """,
            (doc_id, text, len(text)),
        )
        gold_chunk_id = cur.fetchone()[0]

    candidates = btd.retrieval.lexical_search(conn, nonce, limit=10)
    assert any(c.chunk_id == gold_chunk_id for c in candidates)
    negatives = btd.select_hard_negatives(candidates, gold_chunk_id, doc_id, n=5)
    assert all(n.chunk_id != gold_chunk_id for n in negatives)
    assert all(n.document_id != doc_id for n in negatives)
    conn.rollback()  # discard seeded rows
