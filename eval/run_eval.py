"""Phase 5 headline comparison: no-rerank vs off-the-shelf vs trained re-ranker.

For each held-out gold question, take the same first-stage hybrid candidate pool
and score it three ways, then report recall@k / MRR / nDCG@10 for each. The only
thing that differs between systems is the ordering of an identical candidate set,
so the deltas isolate the re-ranker's contribution.

Run with the backend venv (it has retrieval, the ONNX re-ranker, and
sentence-transformers for the off-the-shelf baseline):
    backend/.venv/Scripts/python eval/run_eval.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

import db          # backend/db.py (registers pgvector)
import retrieval   # backend/retrieval.py
import reranker    # backend/reranker.py (ONNX serving — the trained model)
import metrics     # eval/metrics.py

log = logging.getLogger("eval.run")

DEFAULT_GOLD = REPO_ROOT / "data" / "eval" / "gold.jsonl"
DEFAULT_RESULTS = REPO_ROOT / "data" / "eval" / "results.json"
OFF_THE_SHELF = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 50


def rank_of(gold_chunk_id: int, ordered) -> Optional[int]:
    """1-based position of the gold chunk in a ranked result list, or None."""
    for i, r in enumerate(ordered, start=1):
        if r.chunk_id == gold_chunk_id:
            return i
    return None


def _print_table(results: dict, ks) -> None:
    cols = [f"recall@{k}" for k in ks] + ["mrr", "ndcg@10", "found"]
    name_w = max(len(n) for n in results)
    header = "system".ljust(name_w) + "  " + "  ".join(c.rjust(9) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for name, agg in results.items():
        row = name.ljust(name_w) + "  " + "  ".join(f"{agg[c]:9.4f}" for c in cols)
        print(row)
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Three-way re-ranker comparison.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--candidates", type=int, default=CANDIDATES,
                        help="first-stage pool size to re-rank (default 50)")
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not args.gold.is_file():
        log.error("gold set not found: %s — run eval/build_gold.py first", args.gold)
        return 1
    gold = [json.loads(line) for line in args.gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    log.info("loaded %d gold questions", len(gold))

    if not reranker.model_available():
        log.error("trained re-ranker ONNX not found — train + export it first")
        return 1

    from sentence_transformers.cross_encoder import CrossEncoder
    log.info("loading off-the-shelf baseline %s", OFF_THE_SHELF)
    off_model = CrossEncoder(OFF_THE_SHELF)

    def off_the_shelf_order(query, candidates):
        scores = off_model.predict([(query, c.text) for c in candidates],
                                   show_progress_bar=False)
        return [c for c, _ in sorted(zip(candidates, scores),
                                     key=lambda t: t[1], reverse=True)]

    db.load_env()
    conn = db.connect(args.db_url)

    ranks: dict[str, list] = {"no-rerank": [], "off-the-shelf": [], "trained": []}
    try:
        for i, item in enumerate(gold, start=1):
            q, gid = item["question"], item["gold_chunk_id"]
            candidates = retrieval.hybrid_search(conn, q, limit=args.candidates,
                                                 candidate_limit=args.candidates)
            ranks["no-rerank"].append(rank_of(gid, candidates))
            ranks["off-the-shelf"].append(rank_of(gid, off_the_shelf_order(q, candidates)))
            ranks["trained"].append(rank_of(gid, reranker.rerank(q, list(candidates))))
            if i % 10 == 0:
                log.info("  scored %d/%d", i, len(gold))
    finally:
        conn.close()

    ks = (1, 5, 10)
    results = {name: metrics.aggregate(rs, ks) for name, rs in ranks.items()}
    _print_table(results, ks)

    base = results["no-rerank"]["ndcg@10"]
    trained = results["trained"]["ndcg@10"]
    off = results["off-the-shelf"]["ndcg@10"]
    log.info("nDCG@10  no-rerank=%.4f  off-the-shelf=%.4f  trained=%.4f", base, off, trained)
    log.info("trained vs no-rerank: %+.4f   trained vs off-the-shelf: %+.4f",
             trained - base, trained - off)

    # Per-question gold ranks, so the dashboard can show where each gold passage
    # landed under each system rather than only the aggregates. null = not in pool.
    per_question = [
        {
            "question": item["question"],
            "gold_chunk_id": item["gold_chunk_id"],
            "ranks": {name: ranks[name][i] for name in ranks},
        }
        for i, item in enumerate(gold)
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "gold_size": len(gold),
        "candidate_pool": args.candidates,
        "off_the_shelf_model": OFF_THE_SHELF,
        "results": results,
        "per_question": per_question,
    }, indent=2), encoding="utf-8")
    log.info("wrote %s", args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
