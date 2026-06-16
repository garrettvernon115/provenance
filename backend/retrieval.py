"""Baseline hybrid retrieval: pgvector semantic + Postgres FTS, fused via RRF.

This is the candidate generator the Phase 4 re-ranker reorders and the Phase 5
eval harness measures. The functions are deliberately import-friendly (a
connection in, dataclasses out) so the eval harness and the future API call the
exact same code path.

"BM25" in the project notes is realized here as Postgres full-text search ranked
by ``ts_rank_cd``. Reciprocal Rank Fusion combines *rank positions*, not
calibrated scores, so the precise lexical scorer is not load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import psycopg

import embedding

DEFAULT_LIMIT = 50
RRF_K = 60  # standard RRF damping constant (Cormack et al. 2009)

# Columns every retriever selects, so results are interchangeable before fusion.
_SELECT = """
    SELECT c.id, c.document_id, c.section, c.text, c.char_start, c.char_end,
           d.accession, d.company, d.form
"""


@dataclass
class Result:
    chunk_id: int
    document_id: int
    accession: str
    company: str
    form: str
    section: Optional[str]
    text: str
    char_start: int
    char_end: int
    # populated progressively: per-retriever rank (1-based) and the fused score
    semantic_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    semantic_score: Optional[float] = None
    lexical_score: Optional[float] = None
    score: float = 0.0
    retrievers: list[str] = field(default_factory=list)


def _row_to_result(row) -> Result:
    cid, doc_id, section, text, cstart, cend, accession, company, form = row[:9]
    return Result(
        chunk_id=cid, document_id=doc_id, accession=accession, company=company,
        form=form, section=section, text=text, char_start=cstart, char_end=cend,
    )


def semantic_search(
    conn: psycopg.Connection, query: str, limit: int = DEFAULT_LIMIT
) -> list[Result]:
    """k-NN over chunk embeddings (cosine). Skips chunks not yet embedded."""
    qvec = embedding.encode_query(query)
    sql = _SELECT + """
        , 1 - (c.embedding <=> %(qvec)s) AS cosine_sim
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> %(qvec)s
        LIMIT %(limit)s
    """
    results: list[Result] = []
    with conn.cursor() as cur:
        cur.execute(sql, {"qvec": qvec, "limit": limit})
        for rank, row in enumerate(cur.fetchall(), start=1):
            r = _row_to_result(row)
            r.semantic_rank = rank
            r.semantic_score = float(row[9])
            r.retrievers = ["semantic"]
            results.append(r)
    return results


def lexical_search(
    conn: psycopg.Connection, query: str, limit: int = DEFAULT_LIMIT
) -> list[Result]:
    """BM25-style full-text search over chunks.ts, ranked by ts_rank_cd.

    Terms are OR-combined (term overlap, BM25-like), not AND-combined: a
    natural-language question rarely has *every* term present in a relevant
    passage, so AND semantics (e.g. websearch_to_tsquery) would match almost
    nothing. We let ``plainto_tsquery`` do normalization/stemming/stopword
    removal, then flip its ``&`` operators to ``|`` so ranking — not a hard
    conjunction — decides relevance.
    """
    sql = _SELECT + """
        , ts_rank_cd(c.ts, q.query) AS rank
        FROM chunks c JOIN documents d ON d.id = c.document_id,
             (SELECT replace(plainto_tsquery('english', %(query)s)::text,
                             '&', '|')::tsquery AS query) AS q
        WHERE q.query <> '' AND c.ts @@ q.query
        ORDER BY rank DESC
        LIMIT %(limit)s
    """
    results: list[Result] = []
    with conn.cursor() as cur:
        cur.execute(sql, {"query": query, "limit": limit})
        for rank, row in enumerate(cur.fetchall(), start=1):
            r = _row_to_result(row)
            r.lexical_rank = rank
            r.lexical_score = float(row[9])
            r.retrievers = ["lexical"]
            results.append(r)
    return results


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[Result]],
    k: int = RRF_K,
    limit: int = DEFAULT_LIMIT,
) -> list[Result]:
    """Fuse ranked lists by Reciprocal Rank Fusion: score = Σ 1/(k + rank).

    Pure function (no DB/model). Results are merged by chunk_id; the first
    occurrence is kept and enriched with each list's rank/score so callers can
    see which retrievers contributed.
    """
    merged: dict[int, Result] = {}
    for results in result_lists:
        for rank, r in enumerate(results, start=1):
            existing = merged.get(r.chunk_id)
            if existing is None:
                existing = _row_to_result(
                    (r.chunk_id, r.document_id, r.section, r.text, r.char_start,
                     r.char_end, r.accession, r.company, r.form)
                )
                merged[r.chunk_id] = existing
            existing.score += 1.0 / (k + rank)
            if r.semantic_rank is not None:
                existing.semantic_rank = r.semantic_rank
                existing.semantic_score = r.semantic_score
            if r.lexical_rank is not None:
                existing.lexical_rank = r.lexical_rank
                existing.lexical_score = r.lexical_score
            for name in r.retrievers:
                if name not in existing.retrievers:
                    existing.retrievers.append(name)
    ranked = sorted(merged.values(), key=lambda x: x.score, reverse=True)
    return ranked[:limit]


def hybrid_search(
    conn: psycopg.Connection,
    query: str,
    limit: int = DEFAULT_LIMIT,
    candidate_limit: int = DEFAULT_LIMIT,
) -> list[Result]:
    """Run semantic + lexical retrieval and fuse them with RRF."""
    semantic = semantic_search(conn, query, candidate_limit)
    lexical = lexical_search(conn, query, candidate_limit)
    return reciprocal_rank_fusion([semantic, lexical], limit=limit)


def search(
    conn: psycopg.Connection, query: str, mode: str = "hybrid", limit: int = DEFAULT_LIMIT
) -> list[Result]:
    """Dispatch to one retriever by name: 'hybrid' | 'semantic' | 'lexical'."""
    if mode == "semantic":
        return semantic_search(conn, query, limit)
    if mode == "lexical":
        return lexical_search(conn, query, limit)
    if mode == "hybrid":
        return hybrid_search(conn, query, limit)
    raise ValueError(f"unknown search mode: {mode!r}")
