# reranker

Phase 3: the training-data pipeline — the clever part. An LLM is used **only** to
bootstrap training data (it never determines answer quality); BM25/FTS mines hard
negatives. The output is (question, +passage, −passage) examples for the Phase 4
cross-encoder. (Phase 4 — training + ONNX export — lands here next.)

## Setup

```bash
cd reranker
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt    # Windows
# .venv/bin/python -m pip install -r requirements.txt      # macOS/Linux
```

Generation uses the Anthropic API (Claude Haiku 4.5). Put `ANTHROPIC_API_KEY` in
the repo-root `.env` (see `.env.example`). The DB must be up and embedded
(Phases 1–2). The provider is swappable behind `llm.QuestionGenerator` — local
Ollama is a drop-in once wired (`--provider ollama`).

## 1. Generate questions (LLM positives)

```bash
python generate_questions.py --limit 200            # validation sample (~200 chunks)
python generate_questions.py --limit 2 --dry-run    # eyeball quality, no DB writes
python generate_questions.py                        # scale up; reruns resume
```

For each eligible chunk (≥ `--min-chars`, default 400 — skips the tiny Form 4
structured chunks), the LLM writes `--num-questions` (default 3) questions the
chunk answers. They're stored in `generated_questions` (migration `003`).
Idempotent: chunks that already have questions are skipped, so reruns extend the
set and resume after interruptions. For the full corpus, the Anthropic Batches
API (50% cheaper) is the natural scale-up.

## 2. Build training data (BM25 hard negatives → JSONL)

```bash
python build_training_data.py --num-negatives 3
```

For each question, the positive is its source chunk; hard negatives are mined
with the **same** `backend.retrieval.lexical_search` (FTS/`ts_rank_cd`) the live
system uses — lexically plausible passages from *other* documents (the gold chunk
and, by default, its whole document are excluded as likely false negatives).
Writes one JSON object per question to `data/training/triples.jsonl`:

```json
{"question": "...", "positive": {"chunk_id": 1, "text": "..."},
 "negatives": [{"chunk_id": 9, "text": "..."}], "gold_document": "0000..."}
```

## Layout

- `llm.py` — `QuestionGenerator` protocol + `AnthropicQuestionGenerator` (Haiku via
  `messages.parse` structured output) + `get_generator(provider)` factory. The only
  file that knows the provider.
- `generate_questions.py` — chunk selection + generation CLI (idempotent).
- `build_training_data.py` — `select_hard_negatives` (pure) + the mining/export CLI.
- `db.py` — env-based connection; applies ingestion's migration chain.

## Tests

```bash
.venv/Scripts/python -m pytest
```

`test_llm.py` (prompt/parsing/factory) and `test_negatives.py` (selection logic)
are pure — no network, no DB. `test_integration.py` runs against the live Postgres
and rolls back (skips if the DB is down). No test spends Anthropic credits.
