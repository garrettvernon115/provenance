"""Build a held-out gold evaluation set (Phase 5).

Generates questions on chunks that were NOT used to train the re-ranker (i.e.
chunks absent from ``generated_questions``), so the three-way comparison is free
of train/test leakage. Each gold row is (question, the one chunk that answers it).
Reuses the Phase 3 LLM interface; run it with the reranker venv (it has the
Anthropic client).

Usage (from repo root):
    reranker/.venv/Scripts/python eval/build_gold.py --num-chunks 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "reranker"))

import db    # reranker/db.py
import llm   # reranker/llm.py

log = logging.getLogger("eval.build_gold")

DEFAULT_OUT = REPO_ROOT / "data" / "eval" / "gold.jsonl"
DEFAULT_MIN_CHARS = 400


def select_heldout_chunks(conn, n: int, min_chars: int):
    """Long-enough chunks with no generated questions (⇒ never trained on)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.text, c.section, d.accession
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE length(c.text) >= %(min_chars)s
              AND NOT EXISTS (
                  SELECT 1 FROM generated_questions g WHERE g.chunk_id = c.id
              )
            ORDER BY random()
            LIMIT %(n)s
            """,
            {"min_chars": min_chars, "n": n},
        )
        return cur.fetchall()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a held-out gold eval set.")
    parser.add_argument("--num-chunks", type=int, default=50)
    parser.add_argument("--questions-per-chunk", type=int, default=1)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default=llm.DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    db.load_env()

    try:
        generator = llm.get_generator(args.provider, model=args.model)
    except Exception as exc:  # noqa: BLE001
        log.error("could not init %s generator: %s (set ANTHROPIC_API_KEY in .env)",
                  args.provider, exc)
        return 2

    conn = db.connect(args.db_url)
    db.apply_migrations(conn)
    chunks = select_heldout_chunks(conn, args.num_chunks, args.min_chars)
    if not chunks:
        log.error("no held-out chunks found (>= %d chars, not in generated_questions)",
                  args.min_chars)
        conn.close()
        return 1

    log.info("generating gold questions for %d held-out chunks via %s",
             len(chunks), args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for chunk_id, text, section, accession in chunks:
            try:
                questions = generator.generate(text, args.questions_per_chunk)
            except Exception:  # noqa: BLE001
                log.exception("generation failed for chunk %d", chunk_id)
                continue
            for q in questions:
                fh.write(json.dumps({
                    "question": q,
                    "gold_chunk_id": chunk_id,
                    "gold_document": accession,
                    "gold_section": section,
                    "gold_text": text,
                }) + "\n")
                written += 1

    conn.close()
    log.info("wrote %d gold questions to %s", written, args.out.resolve())
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
