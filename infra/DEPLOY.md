# Deploying Provenance

The app is one Docker image (FastAPI + UI + the trained ONNX re-ranker) plus a
Postgres-with-pgvector database. Everything needed to deploy is here; the steps
that touch your hosting/DB accounts are called out as **[you]**.

## What ships

- **Image** (`infra/Dockerfile`): the API, the static UI, and `models/reranker.onnx`.
  The bge query-encoder is pulled on first request and cached in the image's
  `HF_HOME`. CPU-only.
- **Database**: Postgres 16 + `pgvector`, holding `documents` + `chunks` (with
  embeddings). It must be **populated once** (below) — the image does not contain
  the corpus.
- **Secret**: `ANTHROPIC_API_KEY` for the answer layer. Retrieval + re-ranking work
  without it (the UI just shows cited passages instead of a written answer).

## Run it locally first (full stack)

```bash
docker compose -f infra/docker-compose.yml up --build
# UI + API on http://localhost:8000  (API talks to the db service on the compose net)
```

This reuses the local `pgdata` volume, so the corpus you already ingested is there.

## Deploy to a cheap host (Fly.io path)

Any host that runs a Docker image + gives you a pgvector Postgres works (Render,
Railway, a small VPS). Concrete steps for **Fly.io**:

1. **[you]** Install `flyctl`, `fly auth login`, and from the repo root `fly launch
   --no-deploy` (it'll read `infra/fly.toml`; pick a name/region).
2. **[you]** Provision Postgres with pgvector. Easiest is a free **Neon** or
   **Supabase** project (both ship pgvector); grab its connection string. Or use
   `fly postgres create` and `CREATE EXTENSION vector;` once.
3. **Populate the DB once** (from your machine, pointing the existing tooling at the
   cloud DB):
   ```bash
   export PROVENANCE_DB_URL="postgresql://…cloud…/provenance"
   ingestion/.venv/Scripts/python ingestion/fetch_edgar.py     # or copy data/raw
   ingestion/.venv/Scripts/python ingestion/ingest.py
   backend/.venv/Scripts/python  backend/embed_chunks.py
   ```
   (`ingest.py` applies all migrations, including pgvector, on connect.)
4. **[you]** Set secrets on the host:
   ```bash
   fly secrets set ANTHROPIC_API_KEY=sk-ant-… PROVENANCE_DB_URL="postgresql://…"
   ```
   The app reads `PROVENANCE_DB_URL` directly, or the `POSTGRES_*` pieces.
5. **[you]** `fly deploy`. Health check is `GET /healthz`.

## Notes

- **Image size**: torch ships in the image only for the bge query-encoder. Exporting
  that encoder to ONNX (as we already did for the re-ranker) would drop torch and
  shrink the image substantially — the obvious next optimization if the host is
  size-constrained.
- **One web worker**: the ONNX session and the embedding model load per process;
  scale with replicas, not threads, for a demo.
- **CI** (`.github/workflows/ci.yml`) builds this image on every push, so a broken
  Dockerfile is caught before deploy.
