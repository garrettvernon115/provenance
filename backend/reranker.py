"""Serve the trained cross-encoder re-ranker (Phase 4).

Reorders first-stage candidates from ``hybrid_search`` by a learned relevance
score. Serving is a plain ONNX Runtime call (no torch) over the model exported by
``reranker/export_onnx.py``. The model/tokenizer paths default to ``models/`` and
are overridable via ``RERANKER_ONNX_PATH`` / ``RERANKER_TOKENIZER_DIR``.

This is the component the whole project exists to produce: it slots in after
retrieval and must beat the no-rerank baseline (measured in Phase 5).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from retrieval import Result

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ONNX = REPO_ROOT / "models" / "reranker.onnx"
DEFAULT_TOKENIZER = REPO_ROOT / "models" / "reranker-tokenizer"
MAX_LENGTH = 256

_lock = threading.Lock()
_session = None
_tokenizer = None
_input_names: list[str] = []


def model_available() -> bool:
    onnx_path = Path(os.environ.get("RERANKER_ONNX_PATH", DEFAULT_ONNX))
    tok_dir = Path(os.environ.get("RERANKER_TOKENIZER_DIR", DEFAULT_TOKENIZER))
    return onnx_path.is_file() and tok_dir.is_dir()


def _load():
    global _session, _tokenizer, _input_names
    if _session is not None:
        return
    with _lock:
        if _session is not None:
            return
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_path = os.environ.get("RERANKER_ONNX_PATH", str(DEFAULT_ONNX))
        tok_dir = os.environ.get("RERANKER_TOKENIZER_DIR", str(DEFAULT_TOKENIZER))
        if not Path(onnx_path).is_file():
            raise FileNotFoundError(
                f"re-ranker ONNX model not found at {onnx_path}; run "
                "reranker/train_reranker.py then reranker/export_onnx.py"
            )
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        _tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        _input_names = [i.name for i in session.get_inputs()]
        _session = session


def score_pairs(query: str, passages: list[str], batch_size: int = 32) -> np.ndarray:
    """Relevance logits for (query, passage) pairs, in input order."""
    _load()
    scores: list[np.ndarray] = []
    for start in range(0, len(passages), batch_size):
        chunk = passages[start:start + batch_size]
        enc = _tokenizer(
            [query] * len(chunk), chunk,
            padding=True, truncation=True, max_length=MAX_LENGTH,
            return_tensors="np",
        )
        feeds = {name: enc[name] for name in _input_names}
        logits = _session.run(None, feeds)[0]
        scores.append(np.asarray(logits, dtype=np.float32).reshape(-1))
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def rerank(query: str, results: list[Result], top_k: Optional[int] = None) -> list[Result]:
    """Reorder retrieval results by the trained re-ranker; sets ``rerank_score``."""
    if not results:
        return results
    scores = score_pairs(query, [r.text for r in results])
    for r, s in zip(results, scores):
        r.rerank_score = float(s)
    ranked = sorted(results, key=lambda r: r.rerank_score, reverse=True)
    return ranked[:top_k] if top_k else ranked
