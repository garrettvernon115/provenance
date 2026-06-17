"""Turn the (question, +passage, -passages) triples into labeled training pairs.

Pure data wrangling — no torch/sentence-transformers — so it's fast to unit
test. A cross-encoder is trained on (query, passage) pairs with a binary label:
the positive passage is 1, each mined hard negative is 0.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def load_examples(path: str | Path) -> list[dict]:
    """Read the JSONL written by build_training_data.py."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def to_pairs(examples: list[dict]) -> list[dict]:
    """Expand each example into labeled (query, passage) rows.

    Each row carries the source ``question`` (used to split without leakage).
    """
    rows: list[dict] = []
    for ex in examples:
        q = ex["question"]
        rows.append({"query": q, "passage": ex["positive"]["text"], "label": 1.0})
        for neg in ex["negatives"]:
            rows.append({"query": q, "passage": neg["text"], "label": 0.0})
    return rows


def split_by_question(
    pairs: list[dict], val_frac: float = 0.1, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Split pairs into train/val by *question*, so a question's positive and
    negatives never straddle the split (which would leak)."""
    questions = sorted({p["query"] for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(questions)
    n_val = max(1, int(len(questions) * val_frac)) if questions else 0
    val_questions = set(questions[:n_val])
    train = [p for p in pairs if p["query"] not in val_questions]
    val = [p for p in pairs if p["query"] in val_questions]
    return train, val
