"""Backfill chunk embeddings (Phase 2).

Embeds every chunk that lacks a vector and writes it back to
``chunks.embedding``. Idempotent: a second run embeds nothing. Use ``--reembed``
after changing the embedding model (which also needs a matching ``vector(N)``
migration). Kept separate from ``ingest.py`` so ingestion stays torch-free and
re-embedding never requires re-parsing.

Usage:
    python embed_chunks.py [--batch-size 64] [--reembed]
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import db
import embedding

log = logging.getLogger("backend.embed_chunks")


def _ingestion_db():
    """Load ingestion/db.py by path (its module name 'db' collides with ours).

    The embedding column ships in ingestion's migration chain (the schema is
    owned there); reuse its tracked migration runner rather than duplicate it.
    """
    path = Path(__file__).resolve().parent.parent / "ingestion" / "db.py"
    spec = importlib.util.spec_from_file_location("ingestion_db", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill chunk embeddings.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="chunks per encode/write batch (default 64)")
    parser.add_argument("--reembed", action="store_true",
                        help="re-embed all chunks, not just those missing a vector")
    parser.add_argument("--db-url", default=None,
                        help="overrides PROVENANCE_DB_URL / POSTGRES_* settings")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # the HF hub / httpx loggers emit a line per file request on model load
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    db.load_env()

    conn = db.connect(args.db_url)
    _ingestion_db().apply_migrations(conn)  # ensures chunks.embedding + HNSW index exist

    where = "" if args.reembed else "WHERE embedding IS NULL"
    # Materialize the id+text worklist up front. The read cursor must be fully
    # consumed before the per-batch commits below — committing invalidates an
    # open server-side cursor — and at this corpus size the texts are only a
    # few MB, so a client-side fetch is simplest and safe.
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, text FROM chunks {where} ORDER BY id")
        worklist = cur.fetchall()

    todo = len(worklist)
    if todo == 0:
        log.info("nothing to embed (all chunks already have vectors)")
        conn.close()
        return 0

    log.info("embedding %d chunks with %s", todo, embedding.MODEL_NAME)
    done = 0
    for start in range(0, todo, args.batch_size):
        done += _flush(conn, worklist[start:start + args.batch_size])
        log.info("  %d/%d", done, todo)

    conn.close()
    log.info("done: embedded %d chunks", done)
    return 0


def _flush(conn, batch: list[tuple[int, str]]) -> int:
    """Embed one batch of (id, text) and persist the vectors. Commits."""
    vectors = embedding.encode_passages([text for _, text in batch])
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE chunks SET embedding = %s WHERE id = %s",
            [(vec, cid) for (cid, _), vec in zip(batch, vectors)],
        )
    conn.commit()
    return len(batch)


if __name__ == "__main__":
    sys.exit(main())
