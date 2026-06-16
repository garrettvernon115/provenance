"""LLM behind a swappable interface — used ONLY to bootstrap training data.

Per the project thesis the LLM never determines answer quality; it generates the
questions each passage answers, which become training positives. The provider is
kept behind ``QuestionGenerator`` so it can be swapped (hosted Claude now, local
Ollama later) without touching the pipeline.

Default provider: Anthropic Claude Haiku 4.5 (cheap, fast, reliable structured
output) via the official SDK and ``messages.parse`` for schema-validated JSON.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

DEFAULT_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You generate realistic search questions for a retrieval system over SEC "
    "EDGAR filings (10-K annual reports and Form 4 insider-trading filings). "
    "Given one passage, write the questions a user would type that this passage "
    "directly and fully answers.\n"
    "Rules:\n"
    "- Each question must be answerable SOLELY from the passage.\n"
    "- Make them self-contained: name the company/subject rather than using bare "
    "pronouns like 'the company' when the passage names it.\n"
    "- Vary specificity and phrasing (a mix of factual, numeric, and conceptual).\n"
    "- No yes/no questions; no questions about the document's formatting.\n"
    "- Do not invent facts not present in the passage."
)


def build_user_prompt(passage: str, n: int) -> str:
    return (
        f"Write {n} distinct search questions that the following passage answers.\n\n"
        f"PASSAGE:\n\"\"\"\n{passage}\n\"\"\""
    )


def clean_questions(questions: list[str], n: int) -> list[str]:
    """Strip, drop empties, de-duplicate case-insensitively, cap to n."""
    seen: set[str] = set()
    out: list[str] = []
    for q in questions:
        q = (q or "").strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= n:
            break
    return out


@runtime_checkable
class QuestionGenerator(Protocol):
    """Generates the questions a passage answers."""

    def generate(self, passage: str, n: int) -> list[str]:
        ...


class AnthropicQuestionGenerator:
    """Question generator backed by the Anthropic Messages API (Claude Haiku)."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024):
        import anthropic
        from pydantic import BaseModel

        class _Questions(BaseModel):
            questions: list[str]

        self._schema = _Questions
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, passage: str, n: int) -> list[str]:
        # Haiku 4.5: no effort/thinking params; structured output via messages.parse.
        resp = self._client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(passage, n)}],
            output_format=self._schema,
        )
        parsed = resp.parsed_output
        if parsed is None:  # refusal or non-conforming output
            return []
        return clean_questions(parsed.questions, n)


def get_generator(provider: str = "anthropic", **kwargs) -> QuestionGenerator:
    """Factory keeping the provider swappable (CLAUDE.md convention)."""
    if provider == "anthropic":
        return AnthropicQuestionGenerator(**kwargs)
    if provider == "ollama":
        raise NotImplementedError(
            "Ollama generator not wired up yet; pass --provider anthropic. "
            "The QuestionGenerator protocol makes adding it a drop-in."
        )
    raise ValueError(f"unknown provider: {provider!r}")
