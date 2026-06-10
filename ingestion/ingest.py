"""Phase 1 CLI: parse fetched filings, chunk them, and load Postgres.

Reads every ``data/raw/<form>/<accession>/metadata.json`` written by
``fetch_edgar.py``, parses the primary document, chunks it with exact source
offsets, and replaces the corresponding ``documents`` + ``chunks`` rows.
Reruns skip filings whose source sha256 is already ingested.

Usage:
    python ingest.py --source edgar --limit 200
    python ingest.py --dry-run        # parse + chunk only, no database
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import chunker
import db
import parsers

log = logging.getLogger("ingest")

REPO_ROOT = Path(__file__).resolve().parent.parent


def discover_filings(data_dir: Path) -> list[dict]:
    """Load all fetch metadata, newest filings first."""
    metas: list[dict] = []
    for meta_path in (data_dir / "raw").glob("*/*/metadata.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("unreadable metadata %s: %s", meta_path, exc)
            continue
        meta["_dir"] = meta_path.parent
        metas.append(meta)
    metas.sort(key=lambda m: (m.get("filed", ""), m.get("accession", "")), reverse=True)
    return metas


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse and chunk fetched filings into the chunks table."
    )
    parser.add_argument("--source", choices=["edgar"], default="edgar",
                        help="corpus source (only 'edgar' so far)")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                        help="data directory containing raw/ (default <repo>/data)")
    parser.add_argument("--limit", type=int, default=None,
                        help="ingest at most N filings (newest first)")
    parser.add_argument("--reingest", action="store_true",
                        help="re-parse and replace filings already in the database")
    parser.add_argument("--dry-run", action="store_true",
                        help="parse + chunk only; do not touch the database")
    parser.add_argument("--db-url", default=None,
                        help="overrides PROVENANCE_DB_URL / POSTGRES_* settings")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    db.load_env()

    metas = discover_filings(args.data_dir)
    if args.limit is not None:
        metas = metas[: args.limit]
    if not metas:
        log.error("no fetched filings under %s — run fetch_edgar.py first",
                  args.data_dir)
        return 1

    conn = None
    if not args.dry_run:
        conn = db.connect(args.db_url)
        db.apply_migrations(conn)

    stats: Counter = Counter()
    total_chunks = 0
    total_chunk_chars = 0
    for meta in metas:
        accession = meta.get("accession", "?")
        form = meta.get("form", "?")
        try:
            if conn is not None and not args.reingest:
                known_sha = db.existing_sha256(conn, accession)
                if known_sha is not None and known_sha == meta.get("sha256"):
                    stats["skipped"] += 1
                    continue
            doc_path = meta["_dir"] / meta.get("document", "")
            if not doc_path.is_file():
                raise parsers.ParserError(f"document file missing: {doc_path}")
            parsed = parsers.parse_filing(doc_path, form)
            chunks = chunker.chunk_blocks(parsed.full_text, parsed.blocks)
            if not chunks:
                raise parsers.ParserError("no chunks produced")
            if conn is not None:
                source_file = doc_path.relative_to(args.data_dir).as_posix()
                db.replace_document(
                    conn, meta, parsed.full_text, parsed.parser, source_file, chunks
                )
                conn.commit()
            stats["ingested"] += 1
            total_chunks += len(chunks)
            total_chunk_chars += sum(len(c.text) for c in chunks)
            log.info("%s %-5s %s  %s  (%d chunks)",
                     "parse " if args.dry_run else "ingest",
                     form, accession, meta.get("company", "?"), len(chunks))
        except parsers.ParserError as exc:
            if conn is not None:
                conn.rollback()
            stats["unparseable"] += 1
            log.warning("skip   %-5s %s — %s", form, accession, exc)
        except Exception:
            if conn is not None:
                conn.rollback()
            stats["failed"] += 1
            log.exception("failed %-5s %s", form, accession)

    if conn is not None:
        conn.close()

    avg_chars = total_chunk_chars // total_chunks if total_chunks else 0
    log.info(
        "done%s: %d ingested, %d skipped (unchanged), %d unparseable, %d failed; "
        "%d chunks (avg %d chars)",
        " (dry run)" if args.dry_run else "",
        stats["ingested"], stats["skipped"], stats["unparseable"], stats["failed"],
        total_chunks, avg_chars,
    )
    return 0 if (stats["ingested"] + stats["skipped"]) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
