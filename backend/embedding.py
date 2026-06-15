"""Sentence-embedding model wrapper (first-stage semantic retrieval).

All model-specific behavior — the identity of the model, the 384-dim output, and
bge's asymmetric query instruction — is contained here, so swapping the embedding
model is a one-file change (plus a re-embed and a matching ``vector(N)`` migration).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger("backend.embedding")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384

# bge-*-en-v1.5 is asymmetric: queries are prefixed with a retrieval instruction,
# passages are embedded as-is. (https://huggingface.co/BAAI/bge-small-en-v1.5)
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model: Optional["SentenceTransformer"] = None
_model_lock = threading.Lock()


def get_model() -> "SentenceTransformer":
    """Load the model once (downloads ~130 MB on first call), then cache it."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                log.info("loading embedding model %s", MODEL_NAME)
                _model = SentenceTransformer(MODEL_NAME)
    return _model


def _encode(texts: Sequence[str], batch_size: int) -> np.ndarray:
    vectors = get_model().encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,  # unit vectors -> cosine distance is well-defined
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def encode_passages(texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    """Embed passages/chunks. Returns an (n, DIM) float32 array of unit vectors."""
    if not texts:
        return np.empty((0, DIM), dtype=np.float32)
    return _encode(texts, batch_size)


def encode_query(text: str) -> np.ndarray:
    """Embed a single search query (with bge's query instruction). Returns (DIM,)."""
    return _encode([QUERY_INSTRUCTION + text], batch_size=1)[0]
