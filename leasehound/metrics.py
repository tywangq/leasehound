"""Per-scan cost & latency metrics.

Every completed scan appends one JSON line to logs/scan_metrics.jsonl: clause
count, API calls, token usage, estimated cost, wall-clock seconds, verdict
counts. Only the file NAME is logged — never lease text (see Privacy in the
README). Costs come from litellm's price map, so they are estimates, not
billing data.

Usage:
    python -m leasehound.metrics    # summarize the log
"""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from litellm import completion_cost, cost_per_token

LOG_PATH = Path(__file__).parent.parent / "logs" / "scan_metrics.jsonl"


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
                "completion_tokens": self.completion_tokens,
                "embedding_tokens": self.embedding_tokens,
                "cost_usd": round(self.cost_usd, 6),
                "seconds": round(time.perf_counter() - self._t0, 2),
            }


def log_scan(meter: ScanMeter, source: str, clauses: int,
             verdicts: dict | None = None, missing: int | None = None,
             split_mode: str | None = None, cache_hit: bool = False,
             gate_flagged: bool = False) -> dict:
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
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    return record


def cost_line(record: dict) -> str:
    return (f"≈ ${record['cost_usd']:.4f} · {record['seconds']}s · "
            f"{record['llm_calls']} LLM calls + {record['embedding_calls']} embeddings")


def main() -> None:
    if not LOG_PATH.exists():
        print(f"No scans logged yet ({LOG_PATH}).")
        return
    records = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    n = len(records)
    seconds = sorted(r["seconds"] for r in records)
    total_cost = sum(r["cost_usd"] for r in records)
    print(f"{n} scans · total ≈ ${total_cost:.4f} · mean ≈ ${total_cost / n:.4f}/scan")
    print(f"latency: median {seconds[n // 2]}s · max {seconds[-1]}s · "
          f"mean clauses {sum(r['clauses'] for r in records) / n:.1f}")


if __name__ == "__main__":
    main()
