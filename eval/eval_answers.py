"""End-to-end answer eval: faithfulness + citation accuracy (Phase 6).

Completes the "proof" layer that Phase 5 deferred until an answer layer existed.
For a sample of held-out gold questions it runs the full pipeline (hybrid →
trained re-ranker → grounded answer) and reports:

- **citation accuracy** — how often the answer cites the gold passage;
- **gold in top-k** — how often the gold passage even reached the answer prompt
  (the retrieval ceiling);
- **faithfulness** — an LLM-judge verdict on whether every claim in the answer is
  supported by the passages it cited (no hallucination).

Run with the backend venv (retrieval + ONNX re-ranker + answer layer); needs
ANTHROPIC_API_KEY and the DB up.
    backend/.venv/Scripts/python eval/eval_answers.py --sample 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

import answer as answer_mod  # backend/answer.py
import db                    # backend/db.py
import reranker              # backend/reranker.py
import retrieval             # backend/retrieval.py

log = logging.getLogger("eval.answers")

DEFAULT_GOLD = REPO_ROOT / "data" / "eval" / "gold.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "eval" / "answer_eval.json"
CANDIDATE_POOL = 50

JUDGE_SYSTEM = (
    "You are a strict grader for a retrieval system. You are given a QUESTION, a "
    "proposed ANSWER, and the PASSAGES the answer was supposed to rely on. Decide "
    "whether every factual claim in the answer is supported by the passages. "
    "Return faithful=true only if nothing in the answer goes beyond what the "
    "passages state; otherwise faithful=false."
)


def judge_faithfulness(client, schema, question, answer_text, passages) -> bool:
    ctx = answer_mod.assemble_context(passages)
    prompt = (f"QUESTION: {question}\n\nANSWER:\n{answer_text}\n\n"
              f"PASSAGES:\n{ctx}\n\nIs the answer fully supported by the passages?")
    try:
        # minimal schema (a single bool) so the structured JSON can't get
        # truncated by max_tokens
        resp = client.messages.parse(
            model=answer_mod.DEFAULT_MODEL, max_tokens=16, system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}], output_format=schema,
        )
        parsed = resp.parsed_output
        return bool(parsed.faithful) if parsed is not None else False
    except Exception as exc:  # noqa: BLE001 - one flaky judgment shouldn't kill the run
        log.warning("judge call failed (%s); counting as unfaithful", exc)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Faithfulness + citation accuracy eval.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--sample", type=int, default=20, help="questions to evaluate")
    parser.add_argument("--k", type=int, default=6, help="passages sent to the answerer")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not args.gold.is_file():
        log.error("gold set not found: %s — run eval/build_gold.py first", args.gold)
        return 1
    gold = [json.loads(l) for l in args.gold.read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = gold[: args.sample]

    db.load_env()
    try:
        answerer = answer_mod.get_answer_generator()
        import anthropic
        from pydantic import BaseModel

        class _Verdict(BaseModel):
            faithful: bool

        judge_client = anthropic.Anthropic()
    except Exception as exc:  # noqa: BLE001
        log.error("answer layer/judge unavailable: %s (set ANTHROPIC_API_KEY)", exc)
        return 2

    conn = db.connect(args.db_url)
    n = 0
    faithful = 0
    cited_gold = 0
    gold_in_topk = 0
    try:
        for item in gold:
            q, gid = item["question"], item["gold_chunk_id"]
            candidates = retrieval.hybrid_search(conn, q, limit=CANDIDATE_POOL,
                                                 candidate_limit=CANDIDATE_POOL)
            ranked = (reranker.rerank(q, candidates, top_k=args.k)
                      if reranker.model_available() else candidates[: args.k])
            result = answerer.generate(q, ranked)
            cited_ids = {ranked[i - 1].chunk_id for i in result.citations
                         if 1 <= i <= len(ranked)}
            topk_ids = {r.chunk_id for r in ranked}

            n += 1
            gold_in_topk += int(gid in topk_ids)
            cited_gold += int(gid in cited_ids)
            cited_passages = [ranked[i - 1] for i in result.citations
                              if 1 <= i <= len(ranked)] or ranked
            faithful += int(judge_faithfulness(judge_client, _Verdict, q,
                                               result.text, cited_passages))
            if n % 5 == 0:
                log.info("  judged %d/%d", n, len(gold))
    finally:
        conn.close()

    if n == 0:
        log.error("no questions evaluated")
        return 1
    summary = {
        "n": n,
        "faithfulness": faithful / n,
        "citation_accuracy": cited_gold / n,
        "gold_in_topk": gold_in_topk / n,
        "k": args.k,
    }
    print("\nanswer-layer eval (n=%d, k=%d)" % (n, args.k))
    print("-" * 40)
    print(f"faithfulness (LLM-judge) : {summary['faithfulness']:.2%}")
    print(f"citation accuracy        : {summary['citation_accuracy']:.2%}")
    print(f"gold passage in top-k    : {summary['gold_in_topk']:.2%}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote %s", args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
