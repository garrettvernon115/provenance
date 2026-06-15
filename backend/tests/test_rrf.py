"""Reciprocal Rank Fusion unit tests — pure, no DB or model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import retrieval
from retrieval import Result, reciprocal_rank_fusion


def make(chunk_id: int, retriever: str, rank: int) -> Result:
    r = Result(
        chunk_id=chunk_id, document_id=1, accession="acc", company="Co",
        form="10-K", section=None, text=f"chunk {chunk_id}", char_start=0, char_end=1,
    )
    if retriever == "semantic":
        r.semantic_rank, r.semantic_score = rank, 1.0
    else:
        r.lexical_rank, r.lexical_score = rank, 1.0
    r.retrievers = [retriever]
    return r


def semantic_list(ids):
    return [make(cid, "semantic", i) for i, cid in enumerate(ids, start=1)]


def lexical_list(ids):
    return [make(cid, "lexical", i) for i, cid in enumerate(ids, start=1)]


def test_item_in_both_lists_outranks_singletons():
    # chunk 2 is mid-rank in both lists; chunks 1 and 9 are top of one list only.
    semantic = semantic_list([1, 2, 3])
    lexical = lexical_list([9, 2, 8])
    fused = reciprocal_rank_fusion([semantic, lexical], k=60)
    assert fused[0].chunk_id == 2
    assert {"semantic", "lexical"} == set(fused[0].retrievers)


def test_scores_match_rrf_formula():
    fused = reciprocal_rank_fusion(
        [semantic_list([1, 2]), lexical_list([2, 1])], k=60
    )
    by_id = {r.chunk_id: r for r in fused}
    # chunk 1: rank 1 semantic + rank 2 lexical; chunk 2: the mirror -> equal totals
    assert by_id[1].score == 1 / 61 + 1 / 62
    assert by_id[2].score == 1 / 61 + 1 / 62


def test_dedup_merges_ranks_from_each_retriever():
    fused = reciprocal_rank_fusion(
        [semantic_list([5]), lexical_list([5])], k=60
    )
    assert len(fused) == 1
    assert fused[0].semantic_rank == 1
    assert fused[0].lexical_rank == 1
    assert fused[0].score == 2 * (1 / 61)


def test_smaller_k_rewards_top_ranks_more():
    lists = [semantic_list([1, 2, 3, 4, 5])]
    spread = reciprocal_rank_fusion(lists, k=1)[0].score
    flat = reciprocal_rank_fusion(lists, k=1000)[0].score
    assert spread > flat  # small k amplifies the gap between rank 1 and the rest


def test_limit_truncates():
    fused = reciprocal_rank_fusion([semantic_list([1, 2, 3, 4, 5])], k=60, limit=3)
    assert len(fused) == 3


def test_single_list_preserves_order():
    fused = reciprocal_rank_fusion([semantic_list([7, 8, 9])], k=60)
    assert [r.chunk_id for r in fused] == [7, 8, 9]


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
