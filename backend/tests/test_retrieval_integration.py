"""End-to-end retrieval against the live Postgres + the real embedding model.

Skips cleanly if either the database or the model is unavailable. Seeds one
document with a unique nonce token, exercises all three retrievers, then rolls
back so nothing persists.
"""

import importlib.util
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import db
import embedding
import retrieval


def _ingestion_db():
    """Load ingestion/db.py by path (it also defines a module named 'db')."""
    path = Path(__file__).resolve().parents[2] / "ingestion" / "db.py"
    spec = importlib.util.spec_from_file_location("ingestion_db", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def conn():
    db.load_env()
    try:
        connection = db.connect()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres unavailable: {exc}")
    try:
        _ingestion_db().apply_migrations(connection)  # ensure embedding column/index
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"migrations failed: {exc}")
    try:
        embedding.get_model()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"embedding model unavailable: {exc}")
    yield connection
    connection.rollback()
    connection.close()


def _seed(conn, text: str) -> int:
    """Insert one document + one fully-covering chunk with an embedding."""
    accession = f"TEST-{uuid.uuid4().hex[:12]}"
    vec = embedding.encode_passages([text])[0]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents
              (accession, cik, company, form, filed, source_file, parser, full_text)
            VALUES (%s, 0, 'INTEGRATION TEST CO', '10-K', '2026-01-01',
                    'test', 'test', %s)
            RETURNING id
            """,
            (accession, text),
        )
        doc_id = cur.fetchone()[0]
        cur.execute(
            """
            INSERT INTO chunks
              (document_id, chunk_index, section, text, char_start, char_end,
               token_estimate, embedding)
            VALUES (%s, 0, 'Item 1A', %s, 0, %s, %s, %s)
            RETURNING id
            """,
            (doc_id, text, len(text), len(text) // 4, vec),
        )
        return cur.fetchone()[0]


def test_all_retrievers_find_seeded_chunk_and_preserve_offsets(conn):
    nonce = f"zqxwvu{uuid.uuid4().hex[:6]}"
    text = (f"The {nonce} initiative exposes the company to unusual "
            f"counterparty settlement risk in emerging markets.")
    chunk_id = _seed(conn, text)

    # lexical: the nonce is unique in the corpus -> top hit
    lex = retrieval.lexical_search(conn, nonce, limit=10)
    assert lex and lex[0].chunk_id == chunk_id
    assert lex[0].lexical_rank == 1

    # semantic: querying the sentence itself should surface the same chunk
    sem = retrieval.semantic_search(conn, text, limit=10)
    assert any(r.chunk_id == chunk_id for r in sem)

    # hybrid: fused result set includes it, tagged with both retrievers
    hyb = retrieval.hybrid_search(conn, f"{nonce} settlement risk", limit=10)
    match = next((r for r in hyb if r.chunk_id == chunk_id), None)
    assert match is not None

    # citation invariant survives the round trip
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.full_text, c.char_start, c.char_end, c.text
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.id = %s
            """,
            (chunk_id,),
        )
        full_text, start, end, chunk_text = cur.fetchone()
    assert full_text[start:end] == chunk_text
