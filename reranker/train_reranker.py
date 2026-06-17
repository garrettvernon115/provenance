"""Fine-tune the cross-encoder re-ranker on the bootstrapped triples (Phase 4).

Starts from a proven MS MARCO MiniLM cross-encoder and fine-tunes it on our
(question, +passage, -passage) data as binary (query, passage)->{1,0} pairs.
The un-fine-tuned base doubles as Phase 5's "off-the-shelf" baseline, so the
before/after comparison is apples-to-apples.

CPU-friendly at the current dataset size; the same script + requirements
reproduce on a free Colab/Kaggle GPU when the dataset grows.

Usage:
    python train_reranker.py                       # defaults: 2 epochs, the sample
    python train_reranker.py --epochs 3 --batch-size 32 --max-length 384
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pairs

log = logging.getLogger("reranker.train")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_FILE = REPO_ROOT / "data" / "training" / "triples.jsonl"
DEFAULT_OUT = REPO_ROOT / "models" / "reranker-trained"
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune the cross-encoder re-ranker.")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256,
                        help="token cap per (query, passage) pair (default 256)")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not args.train_file.is_file():
        log.error("training file not found: %s — run build_training_data.py first",
                  args.train_file)
        return 1

    # Heavy imports kept out of module load so pairs.py stays torch-free for tests.
    from datasets import Dataset
    from sentence_transformers.cross_encoder import (
        CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    examples = pairs.load_examples(args.train_file)
    all_pairs = pairs.to_pairs(examples)
    train_pairs, val_pairs = pairs.split_by_question(all_pairs, args.val_frac, args.seed)
    log.info("%d examples -> %d pairs (%d train, %d val) across %d questions",
             len(examples), len(all_pairs), len(train_pairs), len(val_pairs),
             len({p["query"] for p in all_pairs}))

    train_ds = Dataset.from_list(train_pairs)
    eval_ds = Dataset.from_list(val_pairs) if val_pairs else None

    log.info("loading base cross-encoder %s", args.base_model)
    model = CrossEncoder(args.base_model, num_labels=1, max_length=args.max_length)
    loss = BinaryCrossEntropyLoss(model)

    train_args = CrossEncoderTrainingArguments(
        output_dir=str(REPO_ROOT / "models" / "_train_ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 32),
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
        seed=args.seed,
    )
    trainer = CrossEncoderTrainer(
        model=model, args=train_args,
        train_dataset=train_ds, eval_dataset=eval_ds, loss=loss,
    )
    log.info("training on CPU — this takes a while at this dataset size")
    trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    log.info("saved fine-tuned re-ranker to %s", args.out.resolve())

    # Quick sanity: a relevant pair should now outscore an unrelated one.
    s_rel, s_irr = model.predict([
        ("How much revenue did the company report?",
         "Total net sales for fiscal 2025 were $1.2 billion, up 8% year over year."),
        ("How much revenue did the company report?",
         "The board declared a quarterly cash dividend payable in March."),
    ])
    log.info("sanity scores: relevant=%.3f unrelated=%.3f%s",
             s_rel, s_irr, "  (ok)" if s_rel > s_irr else "  (!! check training)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
