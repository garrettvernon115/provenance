"""Embedding-model tests. Skipped when the model can't be loaded (e.g. offline)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

import embedding


@pytest.fixture(scope="module")
def model_available():
    try:
        embedding.get_model()
    except Exception as exc:  # noqa: BLE001 - download/load failure -> skip, not fail
        pytest.skip(f"embedding model unavailable: {exc}")


def test_passage_embeddings_shape_and_norm(model_available):
    vecs = embedding.encode_passages(["hello world", "second passage"])
    assert vecs.shape == (2, embedding.DIM)
    assert vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)  # normalized -> unit vectors


def test_query_embedding_shape_and_norm(model_available):
    vec = embedding.encode_query("what are the risk factors?")
    assert vec.shape == (embedding.DIM,)
    np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-3)


def test_query_instruction_changes_the_vector(model_available):
    """bge is asymmetric: a query encodes differently than the same text as a passage."""
    text = "supply chain disruption"
    as_query = embedding.encode_query(text)
    as_passage = embedding.encode_passages([text])[0]
    assert not np.allclose(as_query, as_passage, atol=1e-4)


def test_empty_passage_list_returns_empty(model_available):
    vecs = embedding.encode_passages([])
    assert vecs.shape == (0, embedding.DIM)


def test_relevant_passage_scores_above_unrelated(model_available):
    q = embedding.encode_query("How much revenue did the company report?")
    passages = embedding.encode_passages([
        "Total net sales for fiscal 2025 were $1.2 billion, up 8% year over year.",
        "The cafeteria menu rotates on a weekly basis and includes vegetarian options.",
    ])
    sims = passages @ q  # cosine similarity (all unit vectors)
    assert sims[0] > sims[1]
