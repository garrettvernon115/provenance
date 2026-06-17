"""Export the fine-tuned cross-encoder to ONNX and verify parity (Phase 4).

Writes ``models/reranker.onnx`` plus its tokenizer, then checks that ONNX Runtime
reproduces the PyTorch logits. The backend serves the ONNX model so re-ranking is
a lightweight onnxruntime call with no torch dependency at serving time.

Usage:
    python export_onnx.py
    python export_onnx.py --model ../models/reranker-trained --max-length 256
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("reranker.export")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "reranker-trained"
DEFAULT_ONNX = REPO_ROOT / "models" / "reranker.onnx"
DEFAULT_TOKENIZER = REPO_ROOT / "models" / "reranker-tokenizer"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the re-ranker to ONNX.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--tokenizer-out", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    if not args.model.exists():
        log.error("trained model not found: %s — run train_reranker.py first", args.model)
        return 1

    import numpy as np
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder

    ce = CrossEncoder(str(args.model), max_length=args.max_length)
    hf_model = ce.model.eval()
    tokenizer = ce.tokenizer

    sample = [
        ("How much revenue did the company report?",
         "Total net sales for fiscal 2025 were $1.2 billion."),
        ("Who is on the board of directors?",
         "The cafeteria offers vegetarian options on weekdays."),
    ]
    enc = tokenizer(
        [q for q, _ in sample], [p for _, p in sample],
        padding="max_length", truncation=True, max_length=args.max_length,
        return_tensors="pt",
    )
    # Canonical, fixed order so the wrapper has an explicit signature (the dynamo
    # exporter rejects variadic *args) and serving can feed inputs by name.
    input_names = [n for n in ("input_ids", "attention_mask", "token_type_ids")
                   if n in enc]
    log.info("model inputs: %s", input_names)

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            return self.model(**kwargs).logits

    wrapper = Wrapper(hf_model).eval()
    dummy = tuple(enc[name] for name in input_names)

    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {name: {0: "batch", 1: "seq"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch"}
    with torch.no_grad():
        # Legacy TorchScript exporter (dynamo=False) accepts dynamic_axes and a
        # fixed-signature module directly.
        torch.onnx.export(
            wrapper, dummy, str(args.onnx),
            input_names=input_names, output_names=["logits"],
            dynamic_axes=dynamic_axes, opset_version=args.opset,
            do_constant_folding=True, dynamo=False,
        )
    log.info("wrote %s", args.onnx.resolve())

    tokenizer.save_pretrained(str(args.tokenizer_out))
    log.info("wrote tokenizer to %s", args.tokenizer_out.resolve())

    # Parity: ONNX Runtime vs PyTorch on the sample.
    import onnxruntime as ort

    with torch.no_grad():
        torch_logits = wrapper(*dummy).numpy().reshape(-1)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    feeds = {name: enc[name].numpy() for name in input_names}
    onnx_logits = session.run(None, feeds)[0].reshape(-1)

    max_diff = float(np.max(np.abs(torch_logits - onnx_logits)))
    log.info("parity check: torch=%s onnx=%s max|diff|=%.2e",
             np.round(torch_logits, 4), np.round(onnx_logits, 4), max_diff)
    if max_diff > 1e-3:
        log.error("ONNX logits diverge from PyTorch (max diff %.2e) — not exporting clean", max_diff)
        return 2
    log.info("ONNX export verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
