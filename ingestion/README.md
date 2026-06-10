# ingestion

Phase 0: `fetch_edgar.py` pulls recent SEC EDGAR filings onto disk.
Phase 1: `ingest.py` parses them, chunks them with exact source offsets, and loads
the `documents` + `chunks` tables in Postgres.

## Setup

```bash
cd ingestion
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt    # Windows
# .venv/bin/python -m pip install -r requirements.txt      # macOS/Linux
```

SEC's [fair-access policy](https://www.sec.gov/os/accessing-edgar-data) requires a
descriptive User-Agent with contact info. Copy the repo-root `.env.example` to `.env`
and set `EDGAR_USER_AGENT` (or pass `--user-agent`). The same `.env` carries the
`POSTGRES_*` connection settings (host port defaults to 5433).

## Fetch (Phase 0)

```bash
python fetch_edgar.py                          # 10 recent 10-Ks + 25 recent Form 4s
python fetch_edgar.py --num-10k 50 --num-form4 200
python fetch_edgar.py --force                  # refetch filings already on disk
```

Filings land in `<repo>/data/raw/<form>/<accession>/` as the filing's primary
document (main `.htm` for 10-Ks, raw ownership `.xml` for Form 4s) plus a
`metadata.json` (cik, company, form, filed date, source URL, sha256, …).
Reruns skip anything already fetched.

How it finds "recent" filings: walks EDGAR's daily master indexes backwards from
today, then resolves each filing's primary document via the
`data.sec.gov/submissions` API (full-submission `.txt` as fallback). All requests
go through one throttled session (default 8 req/s, hard-capped at SEC's 10).

## Ingest (Phase 1)

```bash
docker compose -f ../infra/docker-compose.yml up -d db   # once
python ingest.py --source edgar                          # everything under data/raw
python ingest.py --limit 20                              # newest 20 filings only
python ingest.py --dry-run                               # parse + chunk, no database
python ingest.py --reingest                              # replace already-ingested docs
```

Pipeline: `parsers.py` → `chunker.py` → `db.py`, schema in `sql/` (applied
automatically as tracked migrations).

- **Parsers** produce a canonical `full_text` plus offset-tracked blocks.
  10-K HTML is iXBRL-aware (hidden tagging machinery and page-break furniture are
  dropped, `Item 1A`-style section labels attached); Form 4 XML is rendered as a
  faithful readable summary (issuer, owner roles, transactions, footnotes); PDFs
  use the pypdf text layer with an OCR fallback hook (optional
  pytesseract + pypdfium2 + Tesseract binary).
- **Chunker** packs blocks to ~1500 chars with ~200-char overlap and enforces the
  citation invariant: `chunk.text == documents.full_text[char_start:char_end]`.
- **Tables**: `documents` (one row per filing, canonical full text) and `chunks`
  (text, section, offsets, generated `tsvector` + GIN index for full-text search).
  The embedding vector column is deliberately deferred to Phase 2, where the
  embedding model — and so the column dimension — gets chosen.

Reruns skip filings whose source sha256 is already ingested (`--reingest` to force).

## Tests

```bash
.venv/Scripts/python -m pytest
```

Parser/chunker/fetcher tests are fixture-only (no network). The DB round-trip test
uses the local Postgres and skips automatically when it isn't running; it rolls
back everything it writes.
