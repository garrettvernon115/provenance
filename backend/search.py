"""CLI for baseline retrieval — query in, ranked cited chunks out (Phase 2).

    python search.py "what are the company's supply chain risks?"
    python search.py "insider stock sales" --mode lexical --k 5

A thin wrapper over retrieval.py; the Phase 6 API will call the same functions.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

import db
import retrieval


def _score_label(result: retrieval.Result, mode: str) -> str:
    """Show the score that's meaningful for the chosen retriever."""
    if mode == "semantic":
        return f"cos={result.semantic_score:.4f}" if result.semantic_score is not None else "cos=—"
    if mode == "lexical":
        return f"ts={result.lexical_score:.4f}" if result.lexical_score is not None else "ts=—"
    return f"rrf={result.score:.4f}"


def _format(result: retrieval.Result, mode: str, snippet_chars: int) -> str:
    snippet = " ".join(result.text.split())
    if len(snippet) > snippet_chars:
        snippet = snippet[:snippet_chars].rstrip() + "…"
    ranks = []
    if result.semantic_rank is not None:
        ranks.append(f"sem#{result.semantic_rank}")
    if result.lexical_rank is not None:
        ranks.append(f"lex#{result.lexical_rank}")
    where = result.section or "-"
    head = (f"{result.company}  ·  {result.form}  ·  {where}  "
            f"·  {_score_label(result, mode)}  [{', '.join(ranks) or '-'}]")
    cite = (f"cite: {result.accession} "
            f"[{result.char_start}:{result.char_end}] (chunk {result.chunk_id})")
    body = textwrap.fill(snippet, width=96, initial_indent="    ",
                         subsequent_indent="    ")
    return f"{head}\n    {cite}\n{body}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query the baseline retrieval system.")
    parser.add_argument("query", help="natural-language question")
    parser.add_argument("--mode", choices=["hybrid", "semantic", "lexical"],
                        default="hybrid", help="retriever to use (default hybrid)")
    parser.add_argument("--k", type=int, default=10, help="results to show (default 10)")
    parser.add_argument("--snippet-chars", type=int, default=280,
                        help="max snippet length per result (default 280)")
    parser.add_argument("--db-url", default=None,
                        help="overrides PROVENANCE_DB_URL / POSTGRES_* settings")
    args = parser.parse_args(argv)

    # filings contain real unicode (curly quotes, etc.); avoid mojibake on
    # Windows consoles that default to a legacy code page
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    db.load_env()
    conn = db.connect(args.db_url)
    try:
        results = retrieval.search(conn, args.query, mode=args.mode, limit=args.k)
    finally:
        conn.close()

    print(f"\n{args.mode} search · \"{args.query}\" · {len(results)} results\n")
    for i, result in enumerate(results, start=1):
        print(f"{i:>2}. {_format(result, args.mode, args.snippet_chars)}\n")
    if not results:
        print("(no matches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
