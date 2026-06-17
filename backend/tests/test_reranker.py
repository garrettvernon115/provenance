"""Serving-side re-ranker tests. Skip until the ONNX model has been exported."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import reranker
from retrieval import Result


def _result(chunk_id: int, text: str) -> Result:
    return Result(
        chunk_id=chunk_id, document_id=1, accession="acc", company="Co",
        form="10-K", section=None, text=text, char_start=0, char_end=len(text),
    )


pytestmark = pytest.mark.skipif(
    not reranker.model_available(),
    reason="re-ranker ONNX model not exported yet (run train_reranker.py + export_onnx.py)",
)


def test_rerank_orders_relevant_passage_first():
    query = "How much revenue did the company report this year?"
    results = [
        _result(1, "The board declared a quarterly cash dividend payable in March."),
        _result(2, "Total net sales for fiscal 2025 were $1.2 billion, up 8% year over year."),
        _result(3, "Our headquarters relocated to a new campus in Austin, Texas."),
    ]
    ranked = reranker.rerank(query, results)
    assert ranked[0].chunk_id == 2
    assert all(r.rerank_score is not None for r in ranked)
    # scores are sorted descending
    assert ranked == sorted(ranked, key=lambda r: r.rerank_score, reverse=True)


def test_rerank_empty_is_noop():
    assert reranker.rerank("q", []) == []


def test_top_k_truncates():
    query = "revenue"
    results = [_result(i, f"passage {i}") for i in range(5)]
    assert len(reranker.rerank(query, results, top_k=2)) == 2
