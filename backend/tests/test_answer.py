"""Tests for the answer layer — pure prompt/format logic, no network."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import answer
from retrieval import Result


def _result(chunk_id, company, text, section="Item 1A", accession="0000-1"):
    return Result(
        chunk_id=chunk_id, document_id=1, accession=accession, company=company,
        form="10-K", section=section, text=text, char_start=0, char_end=len(text),
    )


def test_assemble_context_numbers_passages():
    results = [
        _result(1, "Acme Inc", "Revenue rose 10%."),
        _result(2, "Beta Corp", "Debt is variable rate."),
    ]
    ctx = answer.assemble_context(results)
    assert "[1] Acme Inc" in ctx
    assert "[2] Beta Corp" in ctx
    assert "Revenue rose 10%." in ctx
    assert "0000-1" in ctx  # accession shown for provenance


def test_build_user_prompt_has_question_and_passages():
    prompt = answer.build_user_prompt("What is revenue?",
                                      [_result(1, "Acme", "Revenue was $5M.")])
    assert "What is revenue?" in prompt
    assert "Revenue was $5M." in prompt


def test_get_answer_generator_unknown_provider():
    with pytest.raises(ValueError):
        answer.get_answer_generator("nope")


def test_answer_result_shape():
    r = answer.AnswerResult(text="hi", citations=[1, 2])
    assert r.text == "hi"
    assert r.citations == [1, 2]
