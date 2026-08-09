"""Per-request cost & latency metrics, for both modes.

Every completed scan appends one JSON line to logs/scan_metrics.jsonl: clause
count, API calls, token usage, estimated cost, wall-clock seconds, verdict
counts. Every answered question appends one line to logs/ask_metrics.jsonl.
Only the file NAME is logged — never lease text, and never the question
(see Privacy in the README). Costs come from litellm's price map, so they are
estimates, not billing data.

The two logs are separate files rather than one with a `mode` field, because
summarize_log aggregates on `clauses` and every published scan figure comes
out of it: a mixed log would put questions into the cost-per-scan mean.

Usage:
    python -m leasehound.metrics    # summarize both logs
"""

import argparse
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from litellm import completion_cost, cost_per_token

LOG_PATH = Path(__file__).parent.parent / "logs" / "scan_metrics.jsonl"
ASK_LOG_PATH = Path(__file__).parent.parent / "logs" / "ask_metrics.jsonl"
# The log is gitignored (it names files); this aggregate is committed so the cost and
# latency figures the README quotes are reproducible from the repo.
SUMMARY_PATH = Path(__file__).parent.parent / "evaluation" / "scan_cost_summary.json"


def cached_prompt_tokens_from_usage(usage) -> int:
    """Input tokens the provider served from its prompt cache, or 0 if not reported.

    Defensive on every hop: the field is provider-specific, arrives as an object on
    some responses and a dict on others, and is absent entirely on a cold prompt. A
    metrics helper must never be the thing that breaks a scan.
    """
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    if isinstance(details, dict):
        return int(details.get("cached_tokens") or 0)
    return int(getattr(details, "cached_tokens", 0) or 0)


def cached_prompt_tokens(response) -> int:
    """The same, for a whole response. Streamed calls only ever have the usage."""
    return cached_prompt_tokens_from_usage(getattr(response, "usage", None))


class UsageMeter:
    """Thread-safe usage accumulator for one request — a scan or a question.

    Clause judgments run on a thread pool, so every add takes the lock.
    The clock starts at construction — build the meter when the work starts.

    Named ScanMeter until ask mode was metered. The name was already wrong
    before that: the ask-cost script and the real-format eval both built one,
    and a class whose name excludes two of its four callers is the kind of
    thing that makes a whole mode look out of scope for measurement.
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

    def add_streamed_completion(self, usage, model: str) -> None:
        """Book a STREAMED completion from the usage totals on its final chunk.

        A streamed response has no usage until it ends, which is why ask mode was
        the one unmetered path in the project: its answer call streams, so there
        was no response object to hand add_completion, and the cost of the whole
        mode was left to a one-off script. Requires stream_options=
        {"include_usage": True} on the request — without it the provider sends no
        usage chunk at all and this is never called.

        completion_cost() wants a response object, so the price comes from
        cost_per_token instead: same price map, both halves summed.
        """
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        try:
            cost = sum(cost_per_token(model=model, prompt_tokens=prompt,
                                      completion_tokens=completion))
        except Exception:
            cost = 0.0
        with self._lock:
            self.llm_calls += 1
            self.prompt_tokens += prompt
            self.cached_prompt_tokens += cached_prompt_tokens_from_usage(usage)
            self.completion_tokens += completion
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


def log_scan(meter: UsageMeter, source: str, clauses: int,
             verdicts: dict | None = None, missing: int | None = None,
             split_mode: str | None = None, cache_hit: bool = False,
             gate_flagged: bool = False, clauses_total: int | None = None,
             refused: bool = False, jurisdiction: str | None = None,
             client: str | None = None) -> dict:
    """Append one scan's metrics to the log; returns the record for display.

    The record is also printed to stdout: on Cloud Run the container filesystem
    (and this JSONL file with it) evaporates on scale-to-zero, but stdout lands
    in Cloud Logging and survives.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": Path(source).name,
        "clauses": clauses,
        # Which surface asked for this scan: cli, ui, api, or eval.
        #
        # The published cost and latency figures are computed from this log, and until
        # now the log had no idea what it was a sample OF. "190 logged scans" was really
        # "every scan that happened to be in my file that day" — eval runs, manual UI
        # clicks while building a feature, the demo recorder driving the browser, and
        # deploy smoke tests, averaged together and quoted as a per-scan cost. Those
        # populations are not interchangeable: an eval run scans the same six labelled
        # leases repeatedly, so it pulls the mean toward those documents.
        #
        # Recorded from the next run onward, like every other field here — the existing
        # rows stay untagged rather than being rewritten with a guess.
        "client": client or "unknown",
        **meter.summary(),
    }
    if verdicts is not None:
        record["verdicts"] = verdicts
        # Which judge produced them. The verdict shares published from this log are
        # the only measurement of how often the middle verdict gets used, and today
        # they span an unrecorded mixture: three judge configurations were measured in
        # one afternoon and only one of them ships. Imported here rather than at module
        # scope because scan.py imports this module.
        from leasehound.scan import judge_fingerprint

        record["judge"] = judge_fingerprint()
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
    if refused:
        # The gate read the document as unrelated to renting, so nothing was judged:
        # one classification call, ~$0.0002, zero clauses. Marked because it is not a
        # scan and must not be averaged with scans — the same reason `cache_hit` is
        # marked. Averaged in, a visitor uploading a recipe silently lowers the
        # project's published cost-per-scan.
        record["refused"] = True
    if jurisdiction is not None:
        # Passed only when it is NOT the state whose law was applied, so the field's
        # presence is itself the signal (the caller knows which state that was; this
        # function does not, and a `!= "wa"` here would have hard-coded the default).
        # Recorded because the published cost and latency figures are computed from
        # this log, and a scan of an out-of-state lease is a scan whose verdicts
        # nobody should be quoting — the same reason `gate_flagged` is here.
        record["jurisdiction"] = jurisdiction
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


def log_ask(meter: UsageMeter, retrieved: int, routed: bool,
            with_report: bool = False, jurisdiction: str | None = None) -> dict:
    """Append one answered question's metrics to logs/ask_metrics.jsonl.

    Deliberately records nothing about the question itself — not the text, not a
    length, not the retrieved sections. The scan log names a file because a
    developer needs to match a row to a document; a question is the user's own
    words and there is no equivalent need, so the honest minimum is what it cost.
    `retrieved` is a chunk count, which is a property of the pipeline config.

    Ask mode had no runtime metering at all: its cost was known only from a
    one-off script, so the six-stage pipeline was the one design decision in the
    project carrying no ongoing price tag.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # False means the router sent the message down the chitchat path: one call,
        # no retrieval. Mixing those into a mean makes ask mode look cheaper than
        # answering a legal question actually is.
        "routed_to_retrieval": routed,
        "retrieved_chunks": retrieved,
        **meter.summary(),
    }
    if with_report:
        # A scan report rides in the history, which is most of the prompt.
        record["with_report_context"] = True
    if jurisdiction is not None:
        # Passed only when the asker named a state this corpus does not cover, so the
        # field's presence is the signal — the same shape as the scan log's. A
        # two-letter code is not a question and not an identifier; it is the one thing
        # about the message worth counting, because a demo answering Oregon renters
        # with Washington law is a fact about the demo, not about them.
        record["jurisdiction"] = jurisdiction
    ASK_LOG_PATH.parent.mkdir(exist_ok=True)
    with ASK_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    return record


def summarize_ask_log(records: list[dict]) -> dict:
    """Aggregate the ask log. Retrieval-routed questions only — see log_ask."""
    routed = [r for r in records if r.get("routed_to_retrieval")]
    if not routed:
        return {"questions": 0}
    seconds = sorted(r["seconds"] for r in routed)
    n = len(routed)
    total_cost = sum(r["cost_usd"] for r in routed)
    return {
        "questions": n,
        "chitchat_excluded": len(records) - n,
        "mean_cost_usd": round(total_cost / n, 5),
        "total_cost_usd": round(total_cost, 5),
        "p50_seconds": percentile(seconds, 0.50),
        "p95_seconds": percentile(seconds, 0.95),
        "max_seconds": seconds[-1],
        "mean_llm_calls": round(sum(r["llm_calls"] for r in routed) / n, 2),
        "mean_embedding_calls": round(sum(r["embedding_calls"] for r in routed) / n, 2),
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in routed) / n),
        "cached_prompt_token_share": round(
            sum(r["cached_prompt_tokens"] for r in routed)
            / max(1, sum(r["prompt_tokens"] for r in routed)), 3),
    }


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
    # Refusals first, and before the clause filter: a refused document judged nothing,
    # so it is not a scan at any clause count. The 9–15 band happened to drop them
    # (they log 0 clauses) while `all_scans` counted them, which is the shape of bug
    # that makes a mean quietly wrong rather than obviously wrong.
    #
    # Zero judged clauses counts as a refusal even without the flag, for two reasons:
    # rows written before the flag existed are still in the log, and "0 clauses
    # judged" is not a scan whatever produced it. A document that splits into no
    # clauses raises NoTextExtracted and never reaches here, so this cannot swallow a
    # real scan.
    def refused(record: dict) -> bool:
        return bool(record.get("refused")) or record["clauses"] == 0

    refusals = [r for r in records if refused(r)]
    records = [r for r in records if not refused(r)]
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
        "refusals_excluded": len(refusals),
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
        "jurisdiction_mismatches": sum(1 for r in paid if r.get("jurisdiction")),
        # What this is a sample OF. Without it the headline mean is quoted over an
        # undefined population — eval runs rescanning the same six labelled leases,
        # manual UI clicks during development, the demo recorder driving a browser, and
        # deploy smoke tests, averaged together. Those are not interchangeable samples,
        # and an eval-heavy month would drag the mean toward the labelled documents
        # without anything looking wrong.
        #
        # `unknown` is every row written before the tag existed, which today is all of
        # them. It shrinks as the log turns over rather than being backfilled with a
        # guess about what a row from July was.
        "by_client": {
            client: sum(1 for r in paid if (r.get("client") or "unknown") == client)
            for client in sorted({(r.get("client") or "unknown") for r in paid})
        },
        # How often each verdict is actually used, over every clause these scans
        # judged. Yellow is here because nothing else measured it: the labelled sets
        # score red against a manifest and count yellow only as a hedge on a planted
        # violation, so a scanner that cautioned on every clause would post zero false
        # reds and look immaculate. This says what the shipped judge really does with
        # the middle verdict, on real traffic, for free.
        **verdict_shares(paid),
    }


def verdict_shares(records: list[dict]) -> dict:
    """Red / yellow / green as shares of the clauses judged, or {} if unrecorded.

    Rows written before `verdicts` existed are skipped rather than counted as
    all-green, and the denominator says how many scans the shares actually rest on.
    """
    with_verdicts = [r for r in records if r.get("verdicts")]
    judged = sum(sum(r["verdicts"].values()) for r in with_verdicts)
    if not judged:
        return {}
    return {
        "verdict_scans": len(with_verdicts),
        "verdict_clauses": judged,
        # Which judges these shares rest on. "unrecorded" is every row written before
        # the field existed, which is most of them — and more than a footnote, because
        # three judge configurations were measured against these sets in one afternoon
        # and a share averaged over them belongs to none of them. Listed rather than
        # filtered: dropping rows to make one number clean would throw away real
        # measurements to flatter it.
        "verdict_judges": sorted({r.get("judge", "unrecorded") for r in with_verdicts}),
        "verdict_share": {
            verdict: round(sum(r["verdicts"][verdict] for r in with_verdicts) / judged, 4)
            for verdict in ("red", "yellow", "green")
        },
    }


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help=f"also write the shareable summary to {SUMMARY_PATH.name}")
    args = parser.parse_args()

    records = read_log(LOG_PATH)
    asks = read_log(ASK_LOG_PATH)
    if not records and not asks:
        print(f"Nothing logged yet ({LOG_PATH.parent}).")
        return

    overall = summarize_log(records) if records else {"scans": 0}
    if overall["scans"]:
        print(f"{overall['scans']} scans · total ≈ ${overall['total_cost_usd']:.4f} · "
              f"mean ≈ ${overall['mean_cost_usd']:.4f}/scan")
        print(f"latency: p50 {overall['p50_seconds']}s · p95 {overall['p95_seconds']}s · "
              f"max {overall['max_seconds']}s · mean clauses {overall['mean_clauses']}")

    ask_overall = summarize_ask_log(asks)
    if ask_overall["questions"]:
        print(f"{ask_overall['questions']} questions · mean ≈ "
              f"${ask_overall['mean_cost_usd']:.5f}/question · "
              f"{ask_overall['mean_llm_calls']} LLM calls")
        print(f"latency: p50 {ask_overall['p50_seconds']}s · "
              f"p95 {ask_overall['p95_seconds']}s · max {ask_overall['max_seconds']}s · "
              f"{ask_overall['cached_prompt_token_share']:.0%} of input tokens cached")

    if args.write:
        from evaluation.provenance import stamp
        SUMMARY_PATH.write_text(json.dumps({
            "note": "Aggregated from logs/scan_metrics.jsonl and logs/ask_metrics.jsonl, "
                    "which are gitignored because the scan log records a file name per "
                    "scan. Regenerate with `python -m leasehound.metrics --write`.",
            "provenance": stamp(),
            "all_scans": overall,
            # The README quotes this band, so it is reported as its own row rather
            # than left for a reader to reconstruct.
            "scans_9_to_15_clauses": summarize_log(records, clause_range=(9, 15)),
            # Ask mode's shipped cost, from production traffic rather than from the
            # A/B script — which measures two configurations against each other and
            # so cannot be the source of truth for what the shipped one costs.
            "all_questions": ask_overall,
        }, indent=2), encoding="utf-8")
        print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
