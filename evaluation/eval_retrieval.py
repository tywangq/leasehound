"""Retrieval-layer evaluation: MRR and hit-rate against the statute-level ground truth.

Each ablation row is one PipelineConfig. Ground truth: the statute section a test
question was generated from must appear in the retrieved chunks' metadata.

Usage:
    python -m evaluation.eval_retrieval --name baseline \
        --collection wa_reference_naive --no-dual --no-grader --no-rerank
    python -m evaluation.eval_retrieval --name full
"""

import argparse
import json
import math
from pathlib import Path

from tqdm import tqdm

from leasehound.retrieval import PipelineConfig, fetch_context

TESTS_PATH = Path(__file__).parent / "tests.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.jsonl"


def load_tests() -> list[dict]:
    with open(TESTS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate(config: PipelineConfig, name: str) -> dict:
    tests = load_tests()
    reciprocal_ranks, hits5, hits10, ndcgs = [], 0, 0, []

    for case in tqdm(tests, desc=name):
        chunks = fetch_context(case["question"], config)
        sections = [c.metadata.get("section") for c in chunks]
        try:
            rank = sections.index(case["section"]) + 1
        except ValueError:
            rank = None

        reciprocal_ranks.append(1 / rank if rank else 0.0)
        ndcgs.append(1 / math.log2(rank + 1) if rank else 0.0)
        hits5 += 1 if rank and rank <= 5 else 0
        hits10 += 1 if rank and rank <= 10 else 0

    n = len(tests)
    summary = {
        "name": name,
        "collection": config.collection,
        "dual_query": config.dual_query,
        "grader": config.grader,
        "rerank": config.rerank,
        "n": n,
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "ndcg": round(sum(ndcgs) / n, 4),
        "hit@5": round(hits5 / n, 4),
        "hit@10": round(hits10 / n, 4),
    }
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Label for this ablation row")
    parser.add_argument("--collection", default="wa_reference")
    parser.add_argument("--no-dual", dest="dual", action="store_false")
    parser.add_argument("--no-grader", dest="grader", action="store_false")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false")
    args = parser.parse_args()

    config = PipelineConfig(
        collection=args.collection,
        dual_query=args.dual,
        grader=args.grader,
        rerank=args.rerank,
    )
    summary = evaluate(config, args.name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
