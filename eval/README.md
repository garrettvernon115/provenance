# eval

Phase 5: the proof. A held-out gold set and the headline three-way comparison —
no-rerank vs. off-the-shelf re-ranker vs. the **trained** re-ranker — on
recall@k / MRR / nDCG@10. This is what turns "the model reorders results" into a
defensible "the fine-tuned re-ranker improved nDCG@10 from X to Y."

No dedicated venv: the scripts reuse the existing ones (keeps a 4th multi-GB torch
install off the machine).

## 1. Build the gold set (held-out, leak-free)

```bash
reranker/.venv/Scripts/python eval/build_gold.py --num-chunks 50
```

Generates one question per chunk on chunks **not** used to train the re-ranker
(absent from `generated_questions`), so the comparison has no train/test leakage.
Writes `data/eval/gold.jsonl`: `{question, gold_chunk_id, gold_document, ...}`.
Uses the Phase 3 Anthropic client, so it needs `ANTHROPIC_API_KEY` and runs in the
reranker venv.

> Caveat: each question has one labeled relevant chunk (its source). Other chunks
> may also be relevant but are unlabeled, so absolute metrics are a lower bound —
> but the three systems are scored on the identical gold set and candidate pool,
> so the **comparison** is valid.

## 2. Run the comparison

```bash
backend/.venv/Scripts/python eval/run_eval.py
```

For each gold question it pulls one hybrid candidate pool (`--candidates`, default
50) and scores it three ways:
- **no-rerank** — the RRF order from `backend.retrieval.hybrid_search`.
- **off-the-shelf** — that pool reordered by `cross-encoder/ms-marco-MiniLM-L-6-v2`
  (the base our model was fine-tuned from).
- **trained** — that pool reordered by our fine-tuned model, via the **served ONNX
  path** (`backend.reranker`), so the number reflects what actually ships.

Prints a table and writes `data/eval/results.json`. `found` = fraction of questions
whose gold chunk was in the candidate pool at all (the first-stage ceiling the
re-ranker works within).

## Result (50-question held-out set, 2026-06-17)

Candidate pool = top-50 hybrid; `found` = 0.96 (gold chunk in the pool for 48/50,
the first-stage ceiling the re-ranker works within).

| system        | recall@1 | recall@5 | recall@10 |   MRR  | nDCG@10 |
|---------------|:--------:|:--------:|:---------:|:------:|:-------:|
| no-rerank     |  0.4000  |  0.6600  |  0.7600   | 0.5294 | 0.5786  |
| off-the-shelf |  0.6600  |  0.8600  |  0.9200   | 0.7604 | 0.7983  |
| **trained**   |**0.7200**|**0.8800**|  0.9000   |**0.7893**|**0.8149**|

**Headline:** the fine-tuned re-ranker lifts nDCG@10 from **0.579 → 0.815** over
no-reranking, and edges the off-the-shelf MS MARCO baseline (0.798); it puts the
answer at rank 1 on **72%** of questions vs 66% off-the-shelf and 40% with no
re-ranking.

Honest reading: the dominant win is *re-ranking at all* (+0.22 nDCG@10). The
fine-tuning gain *over the already-strong off-the-shelf base* is real but modest
(+0.017 nDCG@10, +3 questions at recall@1) and within noise at n=50 — recall@10
even dips by one question. Expect the margin to widen with more than 593 training
examples. Reproduce with `run_eval.py`; numbers land in `data/eval/results.json`.

## Layout

- `metrics.py` — `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`, `aggregate` (pure).
- `build_gold.py` — held-out gold-set generator.
- `run_eval.py` — the three-way harness.

## Tests

```bash
backend/.venv/Scripts/python -m pytest eval/tests
```

`test_metrics.py` is pure (no DB, model, or network).

## Answer-layer eval (Phase 6)

Once the answer layer exists, `eval_answers.py` runs the full pipeline (hybrid →
trained re-ranker → grounded answer) on a sample of gold questions and reports
end-to-end quality:

```bash
backend/.venv/Scripts/python eval/eval_answers.py --sample 20
```

- **faithfulness** — an LLM-judge verdict on whether every claim in the answer is
  supported by the passages it cited (no hallucination).
- **citation accuracy** — how often the answer cites the gold passage.
- **gold in top-k** — how often the gold passage reached the answer prompt (the
  retrieval ceiling that bounds citation accuracy).

**Result (20-question sample, k=6, 2026-06-17):** faithfulness **90%**, citation
accuracy **75%**, gold-in-top-6 **75%** — i.e. whenever the gold passage was
retrieved into the prompt, the answer cited it; the limiter is first-stage recall at
k=6, not the answer layer. Needs `ANTHROPIC_API_KEY` and the DB; writes
`data/eval/answer_eval.json`.
