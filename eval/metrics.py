"""Retrieval metrics for the Phase 5 comparison — pure functions, no deps.

Each gold question has exactly one known relevant chunk (its source passage),
so metrics take the 1-based ``rank`` of that gold chunk in a system's ranked
results, or ``None`` when it isn't retrieved at all. With a single relevant
item the ideal DCG is 1, so nDCG@k reduces to 1/log2(rank+1) for rank ≤ k.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional


def reciprocal_rank(rank: Optional[int]) -> float:
    return 1.0 / rank if rank else 0.0


def recall_at_k(rank: Optional[int], k: int) -> float:
    return 1.0 if (rank is not None and rank <= k) else 0.0


def ndcg_at_k(rank: Optional[int], k: int) -> float:
    """nDCG@k with a single relevant item (ideal DCG = 1)."""
    if rank is not None and rank <= k:
        return 1.0 / math.log2(rank + 1)
    return 0.0


def aggregate(ranks: Iterable[Optional[int]], ks=(1, 5, 10)) -> dict:
    """Mean metrics over a set of per-query gold ranks."""
    ranks = list(ranks)
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    out: dict[str, float] = {"n": n}
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(r, k) for r in ranks) / n
    out["mrr"] = sum(reciprocal_rank(r) for r in ranks) / n
    out["ndcg@10"] = sum(ndcg_at_k(r, 10) for r in ranks) / n
    # how often the gold chunk was retrieved at all (first-stage ceiling)
    out["found"] = sum(1 for r in ranks if r is not None) / n
    return out
