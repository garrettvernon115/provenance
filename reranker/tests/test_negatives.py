"""Tests for hard-negative selection — pure, no DB or model."""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# build_training_data appends backend/ to sys.path on import (for retrieval);
# that import is exercised here too, but the function under test is pure.
import build_training_data as btd


@dataclass
class FakeResult:
    chunk_id: int
    document_id: int


def cands(*pairs):
    return [FakeResult(cid, doc) for cid, doc in pairs]


def test_excludes_gold_chunk():
    c = cands((1, 10), (2, 20), (3, 30))
    negs = btd.select_hard_negatives(c, gold_chunk_id=1, gold_document_id=10, n=5)
    assert [r.chunk_id for r in negs] == [2, 3]


def test_excludes_same_document_by_default():
    # chunk 2 shares the gold's document (10) -> dropped as a likely false negative
    c = cands((2, 10), (3, 30), (4, 40))
    negs = btd.select_hard_negatives(c, gold_chunk_id=1, gold_document_id=10, n=5)
    assert [r.chunk_id for r in negs] == [3, 4]


def test_include_same_doc_keeps_them():
    c = cands((2, 10), (3, 30))
    negs = btd.select_hard_negatives(
        c, gold_chunk_id=1, gold_document_id=10, n=5, exclude_same_doc=False
    )
    assert [r.chunk_id for r in negs] == [2, 3]


def test_caps_at_n_and_preserves_rank_order():
    c = cands((2, 20), (3, 30), (4, 40), (5, 50))
    negs = btd.select_hard_negatives(c, gold_chunk_id=1, gold_document_id=10, n=2)
    assert [r.chunk_id for r in negs] == [2, 3]


def test_returns_fewer_when_pool_exhausted():
    c = cands((1, 10), (2, 10))  # one is gold, the other same-doc
    negs = btd.select_hard_negatives(c, gold_chunk_id=1, gold_document_id=10, n=3)
    assert negs == []
