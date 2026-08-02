# Provenance API + UI. Serves hybrid retrieval, the trained ONNX re-ranker, and
# the grounded answer layer. CPU-only; no torch at serving time (re-ranking runs
# through onnxruntime), so the image stays modest.
#
# Build context is the repo root:  docker build -f infra/Dockerfile -t provenance .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Deps first for layer caching. Install the CPU-only torch wheel up front so
# sentence-transformers doesn't pull the default CUDA build (~4 GB of unused
# GPU libraries) — torch is needed only for the bge query-encoder at serve time.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r backend/requirements.txt

# App code, served UI, and the trained model artifacts.
COPY backend/ backend/
COPY frontend/ frontend/
COPY models/ models/
# The eval dashboard reads data/eval/results.json at runtime. Bake it in so the
# Evaluation tab works on hosts without a file mount (e.g. Railway/Render). It's
# gitignored + regenerable; CI stubs it so this COPY stays valid. /api/eval still
# 404s gracefully if it's somehow absent.
COPY data/eval/results.json data/eval/results.json

# The embedding model (bge-small) is fetched on first query and cached here.
ENV HF_HOME=/app/.hf_cache

EXPOSE 8000
WORKDIR /app/backend
# Single worker: the ONNX session and embedding model load once per process.
# Shell form so managed hosts that inject $PORT (Railway, Render) are honoured;
# defaults to 8000 for local/compose where PORT is unset.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
