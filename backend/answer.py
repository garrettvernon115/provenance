"""Grounded, cited answer layer (Phase 6).

Takes a question plus the re-ranked passages and asks an LLM to answer using
ONLY those passages, citing the ones it used by number. The LLM stays behind an
``AnswerGenerator`` protocol so the provider is swappable (CLAUDE.md convention)
— it is a commodity here; the trained re-ranker is what determines which
passages it sees.

Default provider: Claude Haiku 4.5 via ``messages.parse`` structured output, so
the cited passage numbers come back machine-readable and map straight to chunk
citations (accession + exact char offsets).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from retrieval import Result

DEFAULT_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You answer questions about SEC EDGAR filings (10-K annual reports and Form 4 "
    "insider-trading filings) for a retrieval system.\n"
    "Rules:\n"
    "- Use ONLY the numbered passages provided. Do not use outside knowledge.\n"
    "- Cite every passage you rely on by its number, and cite only passages you "
    "actually used.\n"
    "- If the passages do not contain the answer, say so plainly rather than "
    "guessing.\n"
    "- Be concise and specific; name the company and figures when the passages give them."
)


@dataclass
class AnswerResult:
    text: str
    citations: list[int]  # 1-based passage numbers the model used


def assemble_context(results: "list[Result]") -> str:
    """Render retrieved passages as a numbered context block for the prompt."""
    blocks = []
    for i, r in enumerate(results, start=1):
        where = r.section or "—"
        header = f"[{i}] {r.company} · {r.form} · {where} · ({r.accession})"
        text = " ".join(r.text.split())
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def build_user_prompt(query: str, results: "list[Result]") -> str:
    return (
        f"Question: {query}\n\n"
        f"Passages:\n{assemble_context(results)}\n\n"
        "Answer the question using only these passages and cite the ones you use."
    )


@runtime_checkable
class AnswerGenerator(Protocol):
    def generate(self, query: str, results: "list[Result]") -> AnswerResult:
        ...


class AnthropicAnswerGenerator:
    """Answer generator backed by the Anthropic Messages API (Claude Haiku)."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024):
        import anthropic
        from pydantic import BaseModel, Field

        class _Answer(BaseModel):
            answer: str = Field(description="the grounded answer, citing passage numbers")
            citations: list[int] = Field(
                default_factory=list,
                description="1-based numbers of the passages actually used",
            )

        self._schema = _Answer
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, query: str, results: "list[Result]") -> AnswerResult:
        if not results:
            return AnswerResult(
                text="No relevant passages were retrieved for this question.",
                citations=[],
            )
        resp = self._client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(query, results)}],
            output_format=self._schema,
        )
        parsed = resp.parsed_output
        if parsed is None:
            return AnswerResult(text="(the model returned no answer)", citations=[])
        # keep only citations that refer to a real passage
        valid = [c for c in parsed.citations if 1 <= c <= len(results)]
        return AnswerResult(text=parsed.answer.strip(), citations=valid)


def get_answer_generator(provider: str = "anthropic", **kwargs) -> AnswerGenerator:
    """Factory keeping the answer provider swappable."""
    if provider == "anthropic":
        return AnthropicAnswerGenerator(**kwargs)
    raise ValueError(f"unknown answer provider: {provider!r}")
