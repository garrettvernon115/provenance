"""Tests for the LLM interface — pure helpers + factory, no network."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import llm


def test_clean_questions_strips_dedups_and_caps():
    raw = [
        "  What were total revenues?  ",
        "what were total revenues?",          # case-insensitive duplicate
        "",                                    # dropped
        "   ",                                 # whitespace-only, dropped
        "What is the risk?",
        "How many shares were sold?",
    ]
    assert llm.clean_questions(raw, n=3) == [
        "What were total revenues?",
        "What is the risk?",
        "How many shares were sold?",
    ]


def test_clean_questions_caps_to_n():
    assert llm.clean_questions(["a", "b", "c", "d"], n=2) == ["a", "b"]


def test_clean_questions_handles_empty():
    assert llm.clean_questions([], n=3) == []
    assert llm.clean_questions(["", "  "], n=3) == []


def test_build_user_prompt_includes_passage_and_count():
    prompt = llm.build_user_prompt("Acme had revenue of $5M.", 4)
    assert "Acme had revenue of $5M." in prompt
    assert "4 distinct" in prompt


def test_get_generator_unknown_provider_raises():
    with pytest.raises(ValueError):
        llm.get_generator("nope")


def test_get_generator_ollama_not_implemented():
    with pytest.raises(NotImplementedError):
        llm.get_generator("ollama")


def test_anthropic_generator_conforms_to_protocol_without_instantiating():
    # Construction would need the SDK + API key; just assert the protocol shape.
    assert hasattr(llm.AnthropicQuestionGenerator, "generate")
    assert issubclass(llm.AnthropicQuestionGenerator, object)
