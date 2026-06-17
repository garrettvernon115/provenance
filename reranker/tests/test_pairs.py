"""Tests for training-pair construction — pure, no torch/DB."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pairs

EXAMPLES = [
    {"question": "q1", "positive": {"chunk_id": 1, "text": "pos1"},
     "negatives": [{"chunk_id": 2, "text": "neg1a"}, {"chunk_id": 3, "text": "neg1b"}]},
    {"question": "q2", "positive": {"chunk_id": 4, "text": "pos2"},
     "negatives": [{"chunk_id": 5, "text": "neg2a"}]},
]


def test_to_pairs_labels_and_counts():
    rows = pairs.to_pairs(EXAMPLES)
    # 2 positives + 3 negatives
    assert len(rows) == 5
    assert sum(r["label"] for r in rows) == 2.0
    pos = [r for r in rows if r["label"] == 1.0]
    assert {r["passage"] for r in pos} == {"pos1", "pos2"}
    for r in rows:
        assert set(r) == {"query", "passage", "label"}


def test_split_by_question_has_no_leakage():
    # many questions so a 0.5 split is non-trivial
    examples = [
        {"question": f"q{i}", "positive": {"chunk_id": i, "text": f"p{i}"},
         "negatives": [{"chunk_id": 1000 + i, "text": f"n{i}"}]}
        for i in range(10)
    ]
    rows = pairs.to_pairs(examples)
    train, val = pairs.split_by_question(rows, val_frac=0.5, seed=1)
    train_q = {r["query"] for r in train}
    val_q = {r["query"] for r in val}
    assert train_q and val_q
    assert train_q.isdisjoint(val_q)  # a question never spans both splits
    assert train_q | val_q == {f"q{i}" for i in range(10)}


def test_split_is_deterministic():
    rows = pairs.to_pairs(EXAMPLES)
    a = pairs.split_by_question(rows, val_frac=0.5, seed=7)
    b = pairs.split_by_question(rows, val_frac=0.5, seed=7)
    assert [r["query"] for r in a[1]] == [r["query"] for r in b[1]]


def test_load_examples_roundtrip(tmp_path):
    import json
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in EXAMPLES) + "\n", encoding="utf-8")
    assert pairs.load_examples(p) == EXAMPLES
