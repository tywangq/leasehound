"""What one ask-mode question costs, per pipeline configuration.

Every cost figure this project publishes is scan mode. Ask mode had none — and ask
mode is where the six-stage pipeline lives, kept over a two-stage one on a margin
of one or two questions per metric. A pipeline was rejected here on measured
evidence (hybrid retrieval: −.099 MRR, two false reds) while the six stages were
kept without their price ever being named, and that is the one place this project
failed to hold itself to its own standard.

The extra stages are LLM calls: query rewriting, self-grading, a second rewrite
when the grade fails, and reranking. So the comparison is calls, tokens, dollars
and seconds per question, for the shipped full pipeline against the two-stage
configuration that matched it within noise on the original test set.

Method, and its one deviation from production: retrieval is measured exactly as
production runs it, by metering the real stage calls. The answer call is measured
UNSTREAMED, because a streamed response carries no usage totals — same model, same
messages, same chunks, so the tokens are the tokens; only the delivery differs.

    python -m scripts.measure_ask_cost                 # 6 questions x 2 configs
    python -m scripts.measure_ask_cost --questions 12
"""

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

from litellm import completion

from evaluation.provenance import stamp
from leasehound import retrieval
from leasehound.answer import make_messages
from leasehound.metrics import ScanMeter
from leasehound.retrieval import GENERATION_MODEL, PipelineConfig

TESTS_PATH = Path(__file__).parent.parent / "evaluation" / "tests_adversarial.jsonl"
RESULTS_PATH = Path(__file__).parent.parent / "evaluation" / "ask_cost_results.json"

CONFIGS = {
    # The shipped ask-mode pipeline: semantic chunks, augmentation, dual query,
    # CRAG self-grading, LLM rerank.
    "full pipeline": PipelineConfig(),
    # The simplification the ablation could not separate from it on the original
    # question set: naive chunks, CRAG only.
    "naive + CRAG (two-stage)": PipelineConfig(
        collection="wa_reference_naive", dual_query=False, rerank=False
    ),
}


def measure_one(question: str, config: PipelineConfig) -> dict:
    """Meter every retrieval-stage call, then the answer call, for one question."""
    meter = ScanMeter()
    real_fetch, real_completion = retrieval.fetch_unranked, retrieval.completion

    def counted_fetch(query, cfg, _meter=None):
        return real_fetch(query, cfg, meter)  # the embedding lands in the meter

    def counted_completion(*args, **kwargs):
        response = real_completion(*args, **kwargs)
        meter.add_completion(response)
        return response

    start = time.perf_counter()
    with (
        patch.object(retrieval, "fetch_unranked", counted_fetch),
        patch.object(retrieval, "completion", counted_completion),
    ):
        chunks = retrieval.fetch_context(question, config)
    retrieval_seconds = time.perf_counter() - start
    retrieval_only = meter.summary()

    answer_start = time.perf_counter()
    response = completion(model=GENERATION_MODEL,
                          messages=make_messages(question, [], chunks))
    meter.add_completion(response)
    answer_seconds = time.perf_counter() - answer_start

    total = meter.summary()
    return {
        "retrieval_llm_calls": retrieval_only["llm_calls"],
        "retrieval_embedding_calls": retrieval_only["embedding_calls"],
        "retrieval_cost_usd": round(retrieval_only["cost_usd"], 6),
        "retrieval_seconds": round(retrieval_seconds, 2),
        "answer_prompt_tokens": response.usage.prompt_tokens,
        "answer_seconds": round(answer_seconds, 2),
        "total_llm_calls": total["llm_calls"],
        "total_cost_usd": round(total["cost_usd"], 6),
        "total_seconds": round(retrieval_seconds + answer_seconds, 2),
    }


def mean(rows: list[dict], key: str) -> float:
    return round(sum(r[key] for r in rows) / len(rows), 6)


def median(rows: list[dict], key: str) -> float:
    """Latency needs a median as well as a mean. The first run of this script had
    one question take 54s against a 3-9s spread — a provider hiccup, not a property
    of the configuration, and on six samples it inverted the mean and made the
    cheaper pipeline look slower. Cost is well behaved and stays on the mean."""
    values = sorted(r[key] for r in rows)
    middle = len(values) // 2
    if len(values) % 2:
        return round(values[middle], 6)
    return round((values[middle - 1] + values[middle]) / 2, 6)


def summarize(rows: list[dict]) -> dict:
    return {
        "mean_total_llm_calls": mean(rows, "total_llm_calls"),
        "mean_retrieval_llm_calls": mean(rows, "retrieval_llm_calls"),
        "mean_embedding_calls": mean(rows, "retrieval_embedding_calls"),
        "mean_answer_prompt_tokens": round(mean(rows, "answer_prompt_tokens")),
        "mean_cost_usd": mean(rows, "total_cost_usd"),
        "mean_retrieval_cost_usd": mean(rows, "retrieval_cost_usd"),
        "median_seconds": median(rows, "total_seconds"),
        "median_retrieval_seconds": median(rows, "retrieval_seconds"),
        "mean_seconds": mean(rows, "total_seconds"),
        "slowest_seconds": max(r["total_seconds"] for r in rows),
        "per_question": rows,
    }


def compare(full: dict, two: dict) -> dict:
    return {
        "extra_llm_calls": round(full["mean_total_llm_calls"] - two["mean_total_llm_calls"], 2),
        "cost_ratio": round(full["mean_cost_usd"] / two["mean_cost_usd"], 2),
        "latency_ratio_median": round(full["median_seconds"] / two["median_seconds"], 2),
        "extra_cost_usd": round(full["mean_cost_usd"] - two["mean_cost_usd"], 6),
        "extra_seconds_median": round(full["median_seconds"] - two["median_seconds"], 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=6,
                        help="how many questions to average over (each one is paid)")
    parser.add_argument("--tests", default=str(TESTS_PATH))
    parser.add_argument("--rescore", action="store_true",
                        help="recompute the summaries from the saved per-question rows, "
                             "spending nothing")
    args = parser.parse_args()

    if args.rescore:
        report = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        for name, block in report["configs"].items():
            report["configs"][name] = summarize(block["per_question"])
        report["comparison"] = compare(*(report["configs"][k] for k in CONFIGS))
        RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report["comparison"].items()}, indent=2))
        for name, block in report["configs"].items():
            print(f"{name:26} {block['mean_total_llm_calls']:.2f} calls  "
                  f"${block['mean_cost_usd']:.5f}  median {block['median_seconds']:.2f}s  "
                  f"(slowest {block['slowest_seconds']}s)")
        print(f"\nRescored {report['questions']} saved questions for $0.")
        return

    everything = [json.loads(line) for line in
                  Path(args.tests).read_text(encoding="utf-8").splitlines() if line.strip()]
    # Evenly spaced rather than the first N, so one topic doesn't dominate a small sample.
    step = max(1, len(everything) // args.questions)
    questions = everything[::step][: args.questions]

    report = {"questions": len(questions), "test_set": Path(args.tests).name,
              "provenance": stamp(), "configs": {}}
    for name, config in CONFIGS.items():
        print(f"\n=== {name} ===")
        rows = []
        for entry in questions:
            row = measure_one(entry["question"], config)
            rows.append(row)
            print(f"  {row['total_llm_calls']} calls  ${row['total_cost_usd']:.5f}  "
                  f"{row['total_seconds']:5.2f}s  {entry['question'][:56]}")
        report["configs"][name] = summarize(rows)

    report["comparison"] = compare(*(report["configs"][k] for k in CONFIGS))
    print("\n=== what the extra stages cost ===")
    for key, value in report["comparison"].items():
        print(f"  {key}: {value}")
    total_spend = sum(r["total_cost_usd"] for c in report["configs"].values()
                      for r in c["per_question"])
    report["measurement_cost_usd"] = round(total_spend, 5)
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nThis measurement cost ${total_spend:.5f}. Written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
