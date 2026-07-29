"""Scan concurrency: verify the pool's shape, and state the admission control.

Two claims about this system were asserted rather than measured, and they are
different kinds of claim.

The first is structural and testable for free: "latency is dominated by the
slowest clause in the pool, not by clause count." That follows from
`scan_clauses` running an eight-way thread pool, and it predicts a *staircase* —
1 to 8 clauses cost one wave, 9 to 16 cost two — rather than a slope. Stubbing
the per-clause API call with a sleep is enough to check the staircase is real,
because the shape comes from the pool, not from the API.

The second is about visitors, and a sleep stub cannot honestly measure it: the
scan is I/O-bound, so concurrent scans in one process barely contend, and the
real limits live elsewhere — Gradio's admission control, the provider's rate
limit, and the instance's memory. So this script measures what it can measure
(the staircase, and resident memory under concurrent scans) and does the
arithmetic for the part that is configuration rather than behaviour.

What is NOT measured here, stated plainly: real API latency, provider rate
limiting at peak fan-out, and Cloud Run cold-start or CPU throttling.

    python -m scripts.measure_concurrency
    python -m scripts.measure_concurrency --clause-seconds 0.25   # quick shape check
"""

import argparse
import json
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from leasehound.scan import MAX_PARALLEL_SCANS, scan_clauses

# Gradio admission control, from app.py's demo.queue(...).
QUEUE_CONCURRENCY = 4
QUEUE_MAX_SIZE = 16
# Derived from the logged scans: p50 8.1s over 9-15 clauses is two pool waves,
# so one clause costs roughly half of that.
DEFAULT_CLAUSE_SECONDS = 4.0

RESULTS_PATH = Path(__file__).parent.parent / "evaluation" / "concurrency_results.json"


def stub_clause(seconds: float):
    def scan_clause(clause, index, config, meter=None):
        time.sleep(seconds)
        return {"index": index, "clause": clause, "verdict": "green",
                "citations": [], "urls": {}, "explanation": "stub"}
    return scan_clause


def time_one_scan(clause_count: int) -> float:
    clauses = [f"{i}. CLAUSE. Terms." for i in range(1, clause_count + 1)]
    start = time.perf_counter()
    list(scan_clauses(clauses, config=None))
    return round(time.perf_counter() - start, 3)


def peak_rss_mb() -> float:
    """Peak resident memory. ru_maxrss is bytes on macOS, kilobytes on Linux —
    a platform difference, so branch on the platform rather than guessing from
    the magnitude (guessing reported 240 GB on the first run)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(usage / divisor, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clause-seconds", type=float, default=DEFAULT_CLAUSE_SECONDS)
    args = parser.parse_args()
    seconds = args.clause_seconds

    with patch("leasehound.scan.scan_clause", stub_clause(seconds)):
        print(f"Pool = {MAX_PARALLEL_SCANS} workers, stubbed clause = {seconds}s\n")
        print("clauses  waves  wall clock  per-clause cost if it were serial")
        staircase = {}
        for count in (1, 4, 8, 9, 16, 17, 24):
            elapsed = time_one_scan(count)
            waves = -(-count // MAX_PARALLEL_SCANS)
            staircase[count] = {"waves": waves, "seconds": elapsed}
            print(f"{count:7}  {waves:5}  {elapsed:9.2f}s  {count * seconds:6.1f}s serial")

        print("\nConcurrent scans of 15 clauses (I/O-bound, so expect independence):")
        concurrent = {}
        for n in (1, 2, 4):
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n) as pool:
                times = list(pool.map(lambda _: time_one_scan(15), range(n)))
            concurrent[n] = {"slowest_scan_seconds": round(max(times), 3),
                             "total_seconds": round(time.perf_counter() - start, 3)}
            print(f"  {n} at once: slowest scan {max(times):.2f}s, "
                  f"wall clock {concurrent[n]['total_seconds']:.2f}s")

    scan_seconds = staircase[16]["seconds"]
    admission = {
        "queue_concurrency": QUEUE_CONCURRENCY,
        "queue_max_size": QUEUE_MAX_SIZE,
        "peak_api_calls_at_full_concurrency": QUEUE_CONCURRENCY * MAX_PARALLEL_SCANS,
        "visitors_served_without_waiting": QUEUE_CONCURRENCY,
        "worst_wait_at_queue_full_seconds": round(
            (QUEUE_MAX_SIZE / QUEUE_CONCURRENCY) * scan_seconds, 1),
        "visitor_number_that_gets_turned_away": QUEUE_CONCURRENCY + QUEUE_MAX_SIZE + 1,
    }
    print("\nAdmission control (configuration, not measured):")
    for key, value in admission.items():
        print(f"  {key}: {value}")

    RESULTS_PATH.write_text(json.dumps({
        "pool_workers": MAX_PARALLEL_SCANS,
        "stubbed_clause_seconds": seconds,
        "note": "API latency is stubbed; this measures the pool's shape, not the API.",
        "staircase": staircase,
        "concurrent_scans": concurrent,
        "admission_control": admission,
        "peak_rss_mb": peak_rss_mb(),
    }, indent=2), encoding="utf-8")
    print(f"\nPeak RSS {peak_rss_mb()} MB. Written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
