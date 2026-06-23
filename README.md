# Provenance

Retrieval system that answers natural-language questions over public SEC filings and returns answers with citations to the exact source passage — with retrieval quality driven by a cross-encoder re-ranker trained and evaluated in-house, not an off-the-shelf API.

## Overview

Provenance is a question-answering system over SEC EDGAR filings (10-K annual reports and Form 4 insider-trading filings). A user asks a plain-English question, and the system retrieves candidate passages, re-ranks them with a fine-tuned model, and has an LLM write a grounded answer in which every claim is cited back to the exact source span — viewable, highlighted, in the original document.

The project was built baseline-first. A hybrid retrieval layer established a measurable baseline; a cross-encoder re-ranker was then fine-tuned on LLM-bootstrapped training data to beat it; an evaluation harness proved the improvement on a held-out gold set; and a grounded answer layer plus a web UI turned it into a working application. The design thesis is deliberate: the LLM is used only to bootstrap training data and phrase final answers, while the model that actually determines retrieval quality is trained, measured, and served by hand.

## Key Features

- Natural-language Q&A over SEC EDGAR 10-K and Form 4 filings
- Citations to the exact source passage, with a source viewer that highlights the cited span in the original document
- Hybrid retrieval — pgvector semantic search + Postgres full-text (BM25-style), fused via Reciprocal Rank Fusion
- An in-house fine-tuned cross-encoder re-ranker, exported to ONNX and served without PyTorch at request time
- LLM-bootstrapped training data (LLM-generated questions + BM25 hard-negative mining) — no hand labeling
- Evaluation harness with a three-way comparison (no re-rank vs. off-the-shelf vs. trained) on recall@k, MRR, and nDCG@10
- End-to-end answer faithfulness and citation-accuracy evaluation via an LLM judge
- Grounded, cited answers behind a swappable LLM interface
- In-app evaluation dashboard
- Containerized full stack with PostgreSQL + pgvector and GitHub Actions CI

## Demo

Live Demo: not yet deployed — deploy config and a step-by-step guide are included (`infra/DEPLOY.md`); the full stack runs locally via Docker at http://localhost:8000

Portfolio Case Study: GarrettV.com

## Project Status

Build complete (ingestion through answer layer + UI), CI green, full stack runs in Docker. Ongoing/optional work: cloud deployment, scaling the training set to widen the re-ranker's margin, and slimming the serving image.

## Highlights

- Built a question-answering system over SEC filings that cites the exact source passage for every claim, using Python, FastAPI, PostgreSQL + pgvector, and a fine-tuned cross-encoder served via ONNX.
- Fine-tuned a cross-encoder re-ranker on LLM-bootstrapped training data (LLM-generated questions + BM25 hard-negative mining), improving nDCG@10 from 0.58 (no re-rank) to 0.82 and beating an off-the-shelf MS MARCO baseline on a held-out gold set; the correct passage ranked #1 on 72% of questions, up from 40%.
- Engineered a hybrid retriever (pgvector semantic search + Postgres full-text search, fused via Reciprocal Rank Fusion) as the measured baseline the trained model improves on.
- Designed an evaluation harness measuring recall@k, MRR, and nDCG@10, plus end-to-end answer faithfulness (90%) and citation accuracy (75%) via an LLM judge.
- Exported the trained model to ONNX and served it with onnxruntime (no PyTorch at serve time); containerized the stack with Docker and configured GitHub Actions CI.

## Architecture

Provenance is a Python-first, service-oriented system whose layers all share one retrieval/re-rank code path:

- Ingestion: Python — EDGAR fetch, parse (10-K HTML / Form 4 XML / PDF), chunk with exact source offsets
- Retrieval & Data: PostgreSQL 16 + pgvector (semantic) and full-text search (BM25-style), fused via RRF
- ML Core: a fine-tuned cross-encoder re-ranker (sentence-transformers), exported to ONNX
- Backend & API: a FastAPI service for hybrid retrieval, re-ranking (onnxruntime), and the answer layer
- Frontend: a thin, server-served HTML/JS interface (query, cited answers, source viewer, eval dashboard)
- LLM: Claude (Haiku) behind a swappable interface — used for data generation and the answer layer only
- Deployment: Docker and docker-compose, GitHub Actions CI, Fly.io deploy config

## Tech Stack

### Machine Learning
- Python, PyTorch
- sentence-transformers (cross-encoder fine-tuning)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (fine-tuned re-ranker)
- `BAAI/bge-small-en-v1.5` (embeddings)
- ONNX / onnxruntime

### Backend
- Python 3.12
- FastAPI
- psycopg / psycopg-pool
- Anthropic SDK (Claude)

### Retrieval & Data
- PostgreSQL 16
- pgvector
- Postgres full-text search (BM25-style)

### Frontend
- HTML / vanilla JavaScript / CSS (no build step)

### Cloud & Infrastructure
- Docker and docker-compose
- GitHub Actions (CI)
- Fly.io (deploy config)

### Integration
- SEC EDGAR (data source)
- Anthropic Claude API (data generation and answer layer)

## Running Locally

### Clone Repository

```
git clone https://github.com/garrettvernon115/provenance.git
cd provenance
```

### Configure Environment

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` (used for training-data generation and the answer layer). Retrieval and re-ranking work without it.

### Start PostgreSQL + pgvector

```
docker compose -f infra/docker-compose.yml up -d db
```

### Build the Corpus and Model

A fresh database is empty and the trained model is not committed, so reproduce them with each component's CLI (each component has its own virtual environment and README):

```
python ingestion/fetch_edgar.py            # pull recent 10-K / Form 4 filings
python ingestion/ingest.py                 # parse + chunk into Postgres
python backend/embed_chunks.py             # embed chunks (pgvector)

python reranker/generate_questions.py      # LLM-generated training questions
python reranker/build_training_data.py     # BM25 hard negatives -> triples
python reranker/train_reranker.py          # fine-tune the cross-encoder
python reranker/export_onnx.py             # -> models/reranker.onnx

python eval/build_gold.py                  # held-out gold set
python eval/run_eval.py                    # three-way comparison
```

### Run the Application

```
docker compose -f infra/docker-compose.yml up --build api
```

### Access Application

Open the application locally at http://localhost:8000
