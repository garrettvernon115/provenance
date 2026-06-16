"""Assemble (question, +passage, -passages) training examples (Phase 3).

Reads the LLM-generated questions, treats each question's source chunk as the
positive, and mines **hard negatives** with the same BM25/FTS retriever the live
system uses (``backend.retrieval.lexical_search``) — passages that look
lexically relevant to the question but come from elsewhere in the corpus. Writes
one JSON object per question to a JSONL file for Phase 4 cross-encoder training.

Usage:
    python build_training_data.py --num-negatives 3
    python build_training_data.py --out ../data/training/triples.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import db

# Reuse Phase 2's lexical retriever for negative mining (appended so reranker's
# own modules win the name 'db'; retrieval imports 'embedding'/'psycopg', not 'db').
sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))
import retrieval  # noqa: E402  (backend/retrieval.py)

log = logging.getLogger("reranker.build")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "training" / "triples.jsonl"


def select_hard_negatives(
    candidates,
    gold_chunk_id: int,
    gold_document_id: int,
    n: int,
    exclude_same_doc: bool = True,
):
    """Pick up to ``n`` hard negatives from ranked candidates.

    Drops the gold chunk always, and (by default) any chunk from the gold's own
    document — a same-document chunk is the most likely false negative, since it
    may genuinely answer the question. Pure: candidates need ``chunk_id`` and
    ``document_id`` attributes and must already be in rank order.
    """
    out = []
    for c in candidates:
        if c.chunk_id == gold_chunk_id:
            continue
        if exclude_same_doc and c.document_id == gold_document_id:
            continue
        out.append(c)
        if len(out) >= n:
            break
    return out


def load_questions(conn) -> list[dict]:
    """Every generated question with its positive chunk + source document."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT g.id, g.question, c.id, c.document_id, c.text, d.accession
            FROM generated_questions g
            JOIN chunks c    ON c.id = g.chunk_id
            JOIN documents d ON d.id = c.document_id
            ORDER BY g.id
            """
        )
        rows = cur.fetchall()
    return [
        {"question_id": qid, "question": q, "chunk_id": cid,
         "document_id": doc_id, "text": text, "accession": accession}
        for (qid, q, cid, doc_id, text, accession) in rows
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build (question, +passage, -passage) training data."
    )
    parser.add_argument("--num-negatives", type=int, default=3,
                        help="hard negatives per question (default 3)")
    parser.add_argument("--pool", type=int, default=20,
                        help="BM25 candidates to draw negatives from (default 20)")
    parser.add_argument("--include-same-doc", action="store_true",
                        help="allow negatives from the positive's own document")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output JSONL path (default {DEFAULT_OUT})")
    parser.add_argument("--db-url", default=None,
                        help="overrides PROVENANCE_DB_URL / POSTGRES_* settings")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    db.load_env()

    conn = db.connect(args.db_url)
    questions = load_questions(conn)
    if not questions:
        log.error("no generated questions found — run generate_questions.py first")
        conn.close()
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    no_negatives = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for item in questions:
            candidates = retrieval.lexical_search(conn, item["question"], limit=args.pool)
            negatives = select_hard_negatives(
                candidates, item["chunk_id"], item["document_id"],
                args.num_negatives, exclude_same_doc=not args.include_same_doc,
            )
            if not negatives:
                no_negatives += 1
                continue
            fh.write(json.dumps({
                "question": item["question"],
                "positive": {"chunk_id": item["chunk_id"], "text": item["text"]},
                "negatives": [{"chunk_id": n.chunk_id, "text": n.text} for n in negatives],
                "gold_document": item["accession"],
            }) + "\n")
            written += 1

    conn.close()
    log.info("wrote %d training examples to %s (%d questions had no minable negatives)",
             written, args.out.resolve(), no_negatives)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
