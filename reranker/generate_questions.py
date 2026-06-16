"""Generate training questions from chunks with an LLM (Phase 3 positives).

For each selected chunk, the LLM writes the questions that chunk answers; the
(question, chunk) pair is a training positive. Hard negatives are mined later by
``build_training_data.py``. Idempotent: chunks that already have generated
questions are skipped, so reruns extend the set and resume after interruptions.

Usage:
    python generate_questions.py --limit 200 --num-questions 3
    python generate_questions.py --limit 200            # validation sample
    python generate_questions.py --limit 2 --dry-run    # print, don't write
"""

from __future__ import annotations

import argparse
import logging
import sys

import db
import llm

log = logging.getLogger("reranker.generate")

# Form 4 chunks are short structured summaries; skip the tiniest passages, which
# don't yield good questions. 10-K prose easily clears this.
DEFAULT_MIN_CHARS = 400


def select_chunks(conn, limit: int, min_chars: int) -> list[tuple[int, str]]:
    """Chunks with no questions yet, long enough to be worth generating from.

    Random order so a partial sample spans the corpus rather than the first
    few documents; the NOT EXISTS guard keeps it resumable.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.text
            FROM chunks c
            WHERE length(c.text) >= %(min_chars)s
              AND NOT EXISTS (
                  SELECT 1 FROM generated_questions g WHERE g.chunk_id = c.id
              )
            ORDER BY random()
            LIMIT %(limit)s
            """,
            {"min_chars": min_chars, "limit": limit},
        )
        return cur.fetchall()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate training questions from chunks with an LLM."
    )
    parser.add_argument("--limit", type=int, default=200,
                        help="max chunks to generate for this run (default 200)")
    parser.add_argument("--num-questions", type=int, default=3,
                        help="questions to request per chunk (default 3)")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS,
                        help=f"skip chunks shorter than this (default {DEFAULT_MIN_CHARS})")
    parser.add_argument("--provider", default="anthropic",
                        help="LLM provider (default anthropic)")
    parser.add_argument("--model", default=llm.DEFAULT_MODEL,
                        help=f"model id (default {llm.DEFAULT_MODEL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="print generated questions; do not write to the database")
    parser.add_argument("--db-url", default=None,
                        help="overrides PROVENANCE_DB_URL / POSTGRES_* settings")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    db.load_env()

    try:
        generator = llm.get_generator(args.provider, model=args.model)
    except Exception as exc:  # noqa: BLE001 - surface setup errors (e.g. missing key) clearly
        log.error("could not initialize the %s generator: %s", args.provider, exc)
        log.error("set ANTHROPIC_API_KEY in the repo-root .env (see .env.example).")
        return 2

    conn = db.connect(args.db_url)
    db.apply_migrations(conn)
    chunks = select_chunks(conn, args.limit, args.min_chars)
    if not chunks:
        log.info("no eligible chunks need questions (all done, or none >= %d chars)",
                 args.min_chars)
        conn.close()
        return 0

    log.info("generating up to %d questions for %d chunks via %s",
             args.num_questions, len(chunks), args.model)
    chunks_done = 0
    questions_total = 0
    failed = 0
    for chunk_id, text in chunks:
        try:
            questions = generator.generate(text, args.num_questions)
        except Exception:  # noqa: BLE001 - one bad call shouldn't kill the run
            log.exception("generation failed for chunk %d", chunk_id)
            failed += 1
            continue
        if not questions:
            log.warning("no questions returned for chunk %d", chunk_id)
            continue
        if args.dry_run:
            print(f"\n--- chunk {chunk_id} ---")
            for q in questions:
                print(f"  • {q}")
        else:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO generated_questions (chunk_id, question, model) "
                    "VALUES (%s, %s, %s)",
                    [(chunk_id, q, args.model) for q in questions],
                )
            conn.commit()
        chunks_done += 1
        questions_total += len(questions)
        if chunks_done % 25 == 0:
            log.info("  %d/%d chunks, %d questions", chunks_done, len(chunks),
                     questions_total)

    conn.close()
    log.info(
        "done%s: %d chunks, %d questions, %d failed",
        " (dry run)" if args.dry_run else "", chunks_done, questions_total, failed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
