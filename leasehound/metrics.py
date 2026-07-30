"""Per-scan cost & latency metrics.

Every completed scan appends one JSON line to logs/scan_metrics.jsonl: clause
count, API calls, token usage, estimated cost, wall-clock seconds, verdict
counts. Only the file NAME is logged — never lease text (see Privacy in the
README). Costs come from litellm's price map, so they are estimates, not
billing data.

Usage:
    python -m leasehound.metrics    # summarize the log
"""

import argparse
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from litellm import completion_cost, cost_per_token

LOG_PATH = Path(__file__).parent.parent / "logs" / "scan_metrics.jsonl"
# The log is gitignored (it names files); this aggregate is committed so the cost and
# latency figures the README quotes are reproducible from the repo.
SUMMARY_PATH = Path(__file__).parent.parent / "evaluation" / "scan_cost_summary.json"


def cached_prompt_tokens(response) -> int:
    """Input tokens the provider served from its prompt cache, or 0 if not reported.

    Defensive on every hop: the field is provider-specific, arrives as an object on
    some responses and a dict on others, and is absent entirely on a cold prompt. A
    metrics helper must never be the thing that breaks a scan.
    """
    details = getattr(response.usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    if isinstance(details, dict):
        return int(details.get("cached_tokens") or 0)
    return int(getattr(details, "cached_tokens", 0) or 0)


class ScanMeter:
    """Thread-safe usage accumulator for one scan.

    Clause judgments run on a thread pool, so every add takes the lock.
    The clock starts at construction — build the meter when the scan starts.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self.llm_calls = 0
        self.embedding_calls = 0
        self.prompt_tokens = 0
        self.cached_prompt_tokens = 0
        self.completion_tokens = 0
        self.embedding_tokens = 0
        self.cost_usd = 0.0

    def add_completion(self, response) -> None:
        try:
            cost = completion_cost(completion_response=response)
        except Exception:
            cost = 0.0  # model missing from litellm's price map: count the call, skip cost
        with self._lock:
            self.llm_calls += 1
            self.prompt_tokens += response.usage.prompt_tokens
            # Input tokens the provider served from its own prompt cache, billed at a
            # discount. Recorded because it is the difference between two scans of the
            # same document costing $0.0149 and $0.0075 with byte-identical token
            # counts: same work, different rate. Without this field the cost log looks
            # like the pipeline became cheaper.
            self.cached_prompt_tokens += cached_prompt_tokens(response)
            self.completion_tokens += response.usage.completion_tokens
            self.cost_usd += cost

    def add_embedding(self, response, model: str) -> None:
        try:
            cost = cost_per_token(model=model, prompt_tokens=response.usage.prompt_tokens)[0]
        except Exception:
            cost = 0.0
        with self._lock:
            self.embedding_calls += 1
            self.embedding_tokens += response.usage.prompt_tokens
            self.cost_usd += cost

    def summary(self) -> dict:
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "embedding_calls": self.embedding_calls,
                "prompt_tokens": self.prompt_tokens,
                "cached_prompt_tokens": self.cached_prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "embedding_tokens": self.embedding_tokens,
                "cost_usd": round(self.cost_usd, 6),
                "seconds": round(time.perf_counter() - self._t0, 2),
            }


def log_scan(meter: ScanMeter, source: str, clauses: int,
             verdicts: dict | None = None, missing: int | None = None,
             split_mode: str | None = None, cache_hit: bool = False,
             gate_flagged: bool = False, clauses_total: int | None = None) -> dict:
    """Append one scan's metrics to the log; returns the record for display.

    The record is also printed to stdout: on Cloud Run the container filesystem
    (and this JSONL file with it) evaporates on scale-to-zero, but stdout lands
    in Cloud Logging and survives.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": Path(source).name,
        "clauses": clauses,
        **meter.summary(),
    }
    if verdicts is not None:
        record["verdicts"] = verdicts
    if missing is not None:
        record["missing_protections"] = missing
    if split_mode is not None:
        record["split_mode"] = split_mode
    if cache_hit:
        record["cache_hit"] = True
    if gate_flagged:
        # The document did not read as a lease but was scanned anyway; the verdicts
        # in this record are not comparable with the rest.
        record["gate_flagged"] = True
    if clauses_total is not None and clauses_total != clauses:
        # Over the clause cap: `clauses` is what was judged, this is what the
        # document holds. Verdict counts here describe a prefix, so a cost-per-clause
        # or red-rate average that ignores this field is comparing unlike scans.
        record["clauses_total"] = clauses_total
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    return record


def cost_line(record: dict) -> str:
    return (f"≈ ${record['cost_usd']:.4f} · {record['seconds']}s · "
            f"{record['llm_calls']} LLM calls + {record['embedding_calls']} embeddings")


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile on a sorted list — no numpy for a summary this small."""
    return values[min(len(values) - 1, int(round(fraction * (len(values) - 1))))]


def summarize_log(records: list[dict], clause_range: tuple[int, int] | None = None) -> dict:
    """Aggregate the scan log into a publishable summary.

    Deliberately carries no `source` field. The log records a file name per scan so a
    developer can match a row to a document, and that is exactly why the log itself
    is gitignored — but the cost and latency figures quoted in the README are the
    most-referenced operational numbers in the project and were reproducible from
    nothing. This is the shareable projection: counts, percentiles, no documents.
    """
    if clause_range:
        low, high = clause_range
        records = [r for r in records if low <= r["clauses"] <= high]
    paid = [r for r in records if not r.get("cache_hit")]
    n = len(paid)
    if not n:
        return {"scans": 0}
    seconds = sorted(r["seconds"] for r in paid)
    total_cost = sum(r["cost_usd"] for r in paid)
    return {
        "scans": n,
        "cache_hits_excluded": len(records) - n,
        "clauses_min": min(r["clauses"] for r in paid),
        "clauses_max": max(r["clauses"] for r in paid),
        "mean_clauses": round(sum(r["clauses"] for r in paid) / n, 1),
        "mean_cost_usd": round(total_cost / n, 4),
        "total_cost_usd": round(total_cost, 4),
        "p50_seconds": percentile(seconds, 0.50),
        "p95_seconds": percentile(seconds, 0.95),
        "max_seconds": seconds[-1],
        "mean_llm_calls": round(sum(r["llm_calls"] for r in paid) / n, 1),
        "mean_embedding_calls": round(sum(r["embedding_calls"] for r in paid) / n, 1),
        "partial_scans": sum(1 for r in paid if r.get("clauses_total")),
        "gate_flagged": sum(1 for r in paid if r.get("gate_flagged")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help=f"also write the shareable summary to {SUMMARY_PATH.name}")
    args = parser.parse_args()

    if not LOG_PATH.exists():
        print(f"No scans logged yet ({LOG_PATH}).")
        return
    records = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    overall = summarize_log(records)
    print(f"{overall['scans']} scans · total ≈ ${overall['total_cost_usd']:.4f} · "
          f"mean ≈ ${overall['mean_cost_usd']:.4f}/scan")
    print(f"latency: p50 {overall['p50_seconds']}s · p95 {overall['p95_seconds']}s · "
          f"max {overall['max_seconds']}s · mean clauses {overall['mean_clauses']}")

    if args.write:
        from evaluation.provenance import stamp
        SUMMARY_PATH.write_text(json.dumps({
            "note": "Aggregated from logs/scan_metrics.jsonl, which is gitignored because it "
                    "records a file name per scan. Regenerate with "
                    "`python -m leasehound.metrics --write`.",
            "provenance": stamp(),
            "all_scans": overall,
            # The README quotes this band, so it is reported as its own row rather
            # than left for a reader to reconstruct.
            "scans_9_to_15_clauses": summarize_log(records, clause_range=(9, 15)),
        }, indent=2), encoding="utf-8")
        print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
