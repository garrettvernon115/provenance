# reranker

Phases 3–4: the training-data pipeline and the trained cross-encoder re-ranker —
the project's core. An LLM is used **only** to bootstrap training data (it never
determines answer quality); BM25/FTS mines hard negatives; a cross-encoder is
fine-tuned on the result and exported to ONNX for the backend to serve.

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

## 3. Train the cross-encoder + export ONNX (Phase 4)

```bash
python train_reranker.py                  # fine-tune on data/training/triples.jsonl
python export_onnx.py                     # -> models/reranker.onnx (+ tokenizer)
```

`train_reranker.py` fine-tunes `cross-encoder/ms-marco-MiniLM-L-6-v2` on the triples
as binary (query, passage)→{1,0} pairs (`BinaryCrossEntropyLoss`), splitting
train/val by question so no question straddles the split. Starting from the MS MARCO
base means the *un-fine-tuned* model is Phase 5's "off-the-shelf" baseline — an
apples-to-apples before/after. CPU-friendly at this dataset size; the same script
reproduces on a free Colab/Kaggle GPU when the dataset grows (`--epochs`,
`--batch-size`, `--max-length` are flags).

`export_onnx.py` exports the trained model to `models/reranker.onnx`, saves the
tokenizer, and verifies ONNX Runtime logits match PyTorch (fails if they diverge).
The backend serves this ONNX model — no torch at serving time. See
`backend/reranker.py` and `python backend/search.py "<q>" --rerank`.

## Layout

- `llm.py` — `QuestionGenerator` protocol + `AnthropicQuestionGenerator` (Haiku via
  `messages.parse` structured output) + `get_generator(provider)` factory. The only
  file that knows the provider.
- `generate_questions.py` — chunk selection + generation CLI (idempotent).
- `build_training_data.py` — `select_hard_negatives` (pure) + the mining/export CLI.
- `pairs.py` — triples → labeled (query, passage) pairs + leak-free split (pure).
- `train_reranker.py` — cross-encoder fine-tune; `export_onnx.py` — ONNX export + parity.
- `db.py` — env-based connection; applies ingestion's migration chain.

## Tests

```bash
.venv/Scripts/python -m pytest
```

`test_llm.py`, `test_negatives.py`, `test_pairs.py` are pure — no network, no DB, no
torch. `test_integration.py` runs against the live Postgres and rolls back (skips if
the DB is down). No test spends Anthropic credits. (The serving-side re-ranker test
lives in `backend/tests/test_reranker.py` and skips until the ONNX model is exported.)
