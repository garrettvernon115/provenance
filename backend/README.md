# backend

Phase 2: baseline hybrid retrieval — pgvector semantic search + Postgres full-text
search, fused via Reciprocal Rank Fusion. This is the candidate generator the trained
re-ranker (Phase 4) reorders and the eval harness (Phase 5) measures. The FastAPI answer
API comes in Phase 6; for now the entry point is a CLI.

## Setup

```bash
cd backend
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt    # Windows
# .venv/bin/python -m pip install -r requirements.txt      # macOS/Linux
```

This pulls torch transitively (~2 GB); the embedding model (`BAAI/bge-small-en-v1.5`,
~130 MB) downloads on first use. CPU is fine at the current corpus size. Connection
settings come from the repo-root `.env` (`POSTGRES_*`, host port 5433), shared with
ingestion.

## Embed the corpus

```bash
docker compose -f ../infra/docker-compose.yml up -d db   # if not already running
python embed_chunks.py            # embeds chunks with no vector yet (idempotent)
python embed_chunks.py --reembed  # re-embed everything (after a model change)
```

`embed_chunks.py` applies the `chunks.embedding vector(384)` migration (owned by
ingestion's migration chain) and backfills vectors in batches.

## Search

```bash
python search.py "what are the company's supply chain risks?"
python search.py "insider stock sales" --mode lexical --k 5
python search.py "climate change regulation" --mode semantic
```

`--mode hybrid` (default) fuses semantic + lexical with RRF; `semantic` and `lexical`
run a single retriever. Each result shows company · form · section · fused score, the
contributing retriever ranks, a snippet, and `accession[char_start:char_end]` citation
coordinates.

## Re-ranking (Phase 4)

```bash
python search.py "what are the company's supply chain risks?" --rerank
```

`--rerank` pulls a deeper first-stage pool (`--candidates`, default 50) from hybrid
retrieval, then reorders it with the trained cross-encoder, returning the top `--k`.
Results show the re-ranker score `ce=…` alongside the original `rrf=…`. Requires the
ONNX model — train and export it first (`reranker/train_reranker.py`,
`reranker/export_onnx.py` → `models/reranker.onnx`). Serving is a pure ONNX Runtime
call; no torch is needed at serving time.

## Layout

- `embedding.py` — model wrapper (the only file that knows the model name, the 384-dim
  output, and bge's query instruction; swapping models is a one-file change + re-embed +
  matching migration).
- `retrieval.py` — `semantic_search`, `lexical_search`, `reciprocal_rank_fusion` (pure),
  `hybrid_search`. Import these from the eval harness and API — same code path everywhere.
- `reranker.py` — loads `models/reranker.onnx` + tokenizer (ONNX Runtime); `rerank(query,
  results)` reorders candidates and sets each result's `rerank_score`.
- `db.py` — env-based connection with the pgvector adapter registered.
- `embed_chunks.py`, `search.py` — the two CLIs above.

## Tests

```bash
.venv/Scripts/python -m pytest
```

`test_rrf.py` is pure (always runs). `test_embedding.py` loads the model (skips if it
can't be downloaded). `test_retrieval_integration.py` runs against the live Postgres and
rolls back everything it writes (skips if the DB or model is unavailable).
`test_reranker.py` exercises the serving re-ranker (skips until the ONNX model exists).
