"""Pure tests for the retrieval metrics."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import metrics


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(1) == 1.0
    assert metrics.reciprocal_rank(2) == 0.5
    assert metrics.reciprocal_rank(None) == 0.0


def test_recall_at_k():
    assert metrics.recall_at_k(1, 1) == 1.0
    assert metrics.recall_at_k(2, 1) == 0.0
    assert metrics.recall_at_k(5, 5) == 1.0
    assert metrics.recall_at_k(6, 5) == 0.0
    assert metrics.recall_at_k(None, 10) == 0.0


def test_ndcg_single_relevant():
    assert metrics.ndcg_at_k(1, 10) == 1.0                 # 1/log2(2) = 1
    assert metrics.ndcg_at_k(3, 10) == 1.0 / math.log2(4)  # 0.5
    assert metrics.ndcg_at_k(11, 10) == 0.0                # beyond k
    assert metrics.ndcg_at_k(None, 10) == 0.0
    # monotonic: better rank -> higher ndcg
    assert metrics.ndcg_at_k(2, 10) > metrics.ndcg_at_k(5, 10)


def test_aggregate():
    ranks = [1, 2, None, 11, 5]  # one perfect, one absent, one beyond top-10
    agg = metrics.aggregate(ranks, ks=(1, 5, 10))
    assert agg["n"] == 5
    assert agg["recall@1"] == 1 / 5          # only rank 1
    assert agg["recall@5"] == 3 / 5          # ranks 1,2,5
    assert agg["recall@10"] == 3 / 5         # 11 excluded from recall@10
    # MRR has no rank cutoff: rank 11 still contributes 1/11
    assert math.isclose(agg["mrr"], (1 + 0.5 + 0 + 1 / 11 + 0.2) / 5)
    assert agg["found"] == 4 / 5             # one None
    assert math.isclose(agg["ndcg@10"],
                        (1.0 + 1/math.log2(3) + 0 + 0 + 1/math.log2(6)) / 5)


def test_aggregate_empty():
    assert metrics.aggregate([]) == {"n": 0}
