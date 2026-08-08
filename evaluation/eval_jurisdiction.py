"""Jurisdiction: does the gate know whose law governs the document in front of it?

`state` was a caller parameter defaulting to "wa" and was never inferred from the
document, so a California lease got a full set of verdicts citing Washington
statutes and nothing anywhere said so. The gate accepted it, correctly — a
California lease IS a residential lease — and the disclaimer's "judged against RCW
59.18" reads as scope, not as an error. Nothing in this suite covered it either,
because every labelled lease was written for Washington.

The gate now returns the document's own jurisdiction alongside its kind, on the same
call, and a mismatch puts a warning above the verdicts. This measures whether that
answer is worth printing. Two halves, and the second one matters more:

  * **cases** (jurisdiction_cases.jsonl) — hand-written lease excerpts, each pointing
    at a state by one mechanism: a governing-law clause, the premises address, the
    statutes it cites, two of those in conflict, or nothing at all. Plus a distractor
    that names three states, one that names a state of incorporation, and a planted
    instruction claiming a different jurisdiction — because a lease that can talk its
    way out of its own mismatch warning is worse than no warning.

  * **controls** — the gold leases and the injection leases already in this repo, all
    of them genuine Washington documents. Every one must come back "wa": a warning
    that fires on a Washington lease costs the true warnings their credibility, and
    that is the failure that would actually reach a visitor. These also pin the
    gate's KIND answer, which shares the call and could have moved when the field was
    added — the injection leases are here for that reason as much as this one.

    python -m evaluation.eval_jurisdiction          # cases + controls
    python -m evaluation.eval_jurisdiction --write

One classification call per document, no retrieval, no clause judging, no
protections pass: ≈ $0.02 for the whole thing.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluation.provenance import stamp
from leasehound.answer import route
from leasehound.jurisdiction import UNKNOWN_JURISDICTION, jurisdiction_mismatch
from leasehound.metrics import UsageMeter
from leasehound.scan import MAX_PARALLEL_SCANS, classify_document
from leasehound.upload import read_document, split_clauses_with_mode

EVAL_DIR = Path(__file__).parent
CASES_PATH = EVAL_DIR / "jurisdiction_cases.jsonl"
ASK_CASES_PATH = EVAL_DIR / "jurisdiction_ask_cases.jsonl"
TESTS_PATH = EVAL_DIR / "tests.jsonl"
RESULTS_PATH = EVAL_DIR / "jurisdiction_results.json"

# The state this project's shipped corpus is, and therefore the one a scan applies.
APPLIED = "wa"

# Real documents from this repo, all governed by Washington law. The gold set is the
# acceptance bar; the injection leases are here because the gate is the surface those
# payloads attack, and adding a field to its response schema is exactly the kind of
# change that could move an answer nobody re-checked.
CONTROL_FILES = (
    "leases/lease_clean.md",
    "leases/lease_02_subtle.md",
    "leases/lease_03_gag.md",
    "leases/lease_04_heavy.md",
    "leases/lease_05_minimal.md",
    "../examples/sample_lease.md",
    "leases_synthetic/lease_synth_036_injection.md",
    "leases_synthetic/lease_synth_037_injection.md",
    "leases_synthetic/lease_synth_038_injection.md",
    "leases_synthetic/lease_synth_039_injection.md",
    "leases_synthetic/lease_synth_040_injection.md",
)


def load_cases() -> list[dict]:
    return [json.loads(line) for line in
            CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def ask_gate(text: str, meter: UsageMeter) -> dict:
    clauses, _ = split_clauses_with_mode(text)
    check = classify_document(clauses, meter)
    return {"kind": check.kind, "jurisdiction": check.jurisdiction,
            # What the gate was actually given, which is not the same as the document:
            # clause splitting runs first and discards anything under
            # MIN_CLAUSE_CHARS. Carried so a failure can be attributed rather than
            # assumed — the first run of this eval read two dropped premises clauses
            # as the classifier being weak on addresses.
            "seen": "\n\n".join(clauses)}


def judge_cases(cases: list[dict], meter: UsageMeter) -> list[dict]:
    def one(case: dict) -> dict:
        answer = ask_gate(case["text"], meter)
        evidence = case["evidence"]
        return {
            "id": case["id"], "signal": case["signal"],
            "expected": case["expected"], "answered": answer["jurisdiction"],
            "kind": answer["kind"],
            "correct": answer["jurisdiction"] == case["expected"],
            # False when the text the answer should come from never reached the gate.
            # A wrong answer with this False is not the gate's failure to attribute.
            "evidence_reached_the_gate": (evidence is None
                                         or evidence in answer["seen"]),
            # What the pipeline would actually DO with this answer, which is the only
            # thing a visitor sees. A wrong state that still warns is a much smaller
            # failure than a wrong state that stays quiet.
            "warns": jurisdiction_mismatch(answer["jurisdiction"], APPLIED),
            "should_warn": jurisdiction_mismatch(case["expected"], APPLIED),
        }

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SCANS) as pool:
        return list(pool.map(one, cases))


def judge_controls(meter: UsageMeter) -> list[dict]:
    def one(name: str) -> dict:
        answer = ask_gate(read_document(EVAL_DIR / name), meter)
        return {
            "file": name, "answered": answer["jurisdiction"], "kind": answer["kind"],
            # A Washington document that comes back "unknown" is not a failure: no
            # warning fires, which is the right behaviour. A Washington document that
            # comes back "ca" puts a false alarm above a correct report.
            "false_alarm": jurisdiction_mismatch(answer["jurisdiction"], APPLIED),
            "kind_held": answer["kind"] == "lease_agreement",
        }

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SCANS) as pool:
        return list(pool.map(one, CONTROL_FILES))


def judge_ask_cases(cases: list[dict], meter: UsageMeter) -> list[dict]:
    """The same question of ask mode, whose router now answers it on its own call.

    Scan mode reads a jurisdiction off a governing-law clause; a question has no
    clause, only what the renter says about where they are. So the mechanisms are
    different and the consequence is identical: an Oregon renter got RCW 59.18 with
    nothing saying it does not govern them.
    """
    def one(case: dict) -> dict:
        answer = route(case["question"], meter=meter)
        return {
            "id": case["id"], "signal": case["signal"],
            "expected": case["expected"], "answered": answer.jurisdiction,
            "category": answer.category,
            "correct": answer.jurisdiction == case["expected"],
            "warns": jurisdiction_mismatch(answer.jurisdiction, APPLIED),
            "should_warn": jurisdiction_mismatch(case["expected"], APPLIED),
        }

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SCANS) as pool:
        return list(pool.map(one, cases))


def judge_ask_controls(meter: UsageMeter, limit: int | None = None) -> list[dict]:
    """The 82 questions every other ask-mode number is measured on.

    None of them mentions a state — they are generated from statute sections — so
    every one must come back "unknown" and none may warn. This is the false-alarm
    denominator, and it is the same set the retrieval and generation evals use, so a
    reader can compare without wondering whether the population changed.
    """
    questions = [json.loads(line)["question"] for line in
                 TESTS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        questions = questions[:limit]

    def one(question: str) -> dict:
        answer = route(question, meter=meter)
        return {"question": question[:70], "answered": answer.jurisdiction,
                "false_alarm": jurisdiction_mismatch(answer.jurisdiction, APPLIED)}

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SCANS) as pool:
        return list(pool.map(one, questions))


def score_ask(cases: list[dict], controls: list[dict]) -> dict:
    signals = sorted({c["signal"] for c in cases})
    return {
        "cases": len(cases),
        "exact": sum(1 for c in cases if c["correct"]),
        # What a renter would actually experience, which is coarser than `exact` and
        # is the number that decides whether this feature is worth shipping: did the
        # warning fire exactly when it should have? A wrong state that happens to be
        # the applied one produces no warning and no harm — it is a label miss, not a
        # behaviour miss, and reporting only `exact` would hide which kind occurred.
        "behaviour_matches": sum(1 for c in cases if c["warns"] == c["should_warn"]),
        "missed_mismatches": sorted(c["id"] for c in cases
                                    if c["should_warn"] and not c["warns"]),
        "false_alarms": sorted(c["id"] for c in cases
                               if c["warns"] and not c["should_warn"]),
        "by_signal": {
            signal: f"{sum(1 for c in cases if c['signal'] == signal and c['correct'])}"
                    f"/{sum(1 for c in cases if c['signal'] == signal)}"
            for signal in signals
        },
        "controls": len(controls),
        "control_false_alarms": sum(1 for c in controls if c["false_alarm"]),
        "control_false_alarm_ids": sorted(
            c["question"] for c in controls if c["false_alarm"]),
    }


def score(cases: list[dict], controls: list[dict]) -> dict:
    signals = sorted({c["signal"] for c in cases})
    reached = [c for c in cases if c["evidence_reached_the_gate"]]
    return {
        "cases": len(cases),
        "exact": sum(1 for c in cases if c["correct"]),
        # The same number over the cases whose evidence survived clause splitting. The
        # gap between the two is not the classifier's to answer for, and reporting only
        # the first figure would credit it with a defect it did not cause — or, worse,
        # send someone tuning the gate prompt at a splitter bug.
        "cases_evidence_reached_the_gate": len(reached),
        "exact_of_those": sum(1 for c in reached if c["correct"]),
        "evidence_dropped_before_the_gate": sorted(
            c["id"] for c in cases if not c["evidence_reached_the_gate"]),
        # The two asymmetric failures, named rather than averaged. A missed mismatch
        # is a report that cites the wrong state's law with nothing saying so; a false
        # alarm is a warning over a report that was right all along.
        "missed_mismatches": sorted(c["id"] for c in cases
                                    if c["should_warn"] and not c["warns"]),
        "false_alarms": sorted(c["id"] for c in cases
                               if c["warns"] and not c["should_warn"]),
        "wrong_state_but_warned": sorted(
            c["id"] for c in cases
            if c["warns"] and c["should_warn"] and not c["correct"]),
        "by_signal": {
            signal: f"{sum(1 for c in cases if c['signal'] == signal and c['correct'])}"
                    f"/{sum(1 for c in cases if c['signal'] == signal)}"
            for signal in signals
        },
        "controls": len(controls),
        "control_false_alarms": sorted(c["file"] for c in controls if c["false_alarm"]),
        "control_unknown": sorted(c["file"] for c in controls
                                  if c["answered"] == UNKNOWN_JURISDICTION),
        "control_kind_changed": sorted(c["file"] for c in controls if not c["kind_held"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help=f"write {RESULTS_PATH.name}")
    parser.add_argument("--ask-controls", type=int,
                        help="only the first N of the 82 questions (smoke test)")
    args = parser.parse_args()

    cases = load_cases()
    ask_cases = [json.loads(line) for line in
                 ASK_CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    ask_controls = args.ask_controls or 82
    total = len(cases) + len(CONTROL_FILES) + len(ask_cases) + ask_controls
    print(f"scan: {len(cases)} cases + {len(CONTROL_FILES)} controls · "
          f"ask: {len(ask_cases)} cases + {ask_controls} controls = {total} calls "
          f"≈ ${0.0008 * total:.3f}")

    meter = UsageMeter()
    case_rows = judge_cases(cases, meter)
    control_rows = judge_controls(meter)
    summary = score(case_rows, control_rows)
    ask_case_rows = judge_ask_cases(ask_cases, meter)
    ask_control_rows = judge_ask_controls(meter, args.ask_controls)
    ask_summary = score_ask(ask_case_rows, ask_control_rows)

    print(f"\nSCAN — cases: {summary['exact']}/{summary['cases']} exact "
          f"({summary['exact_of_those']}/{summary['cases_evidence_reached_the_gate']} "
          f"of those whose evidence survived clause splitting)")
    for signal, ratio in summary["by_signal"].items():
        print(f"  {signal:26} {ratio}")
    for row in case_rows:
        if not row["correct"]:
            cause = ("" if row["evidence_reached_the_gate"]
                     else "  [evidence dropped before the gate]")
            print(f"    ! {row['id']:34} expected {row['expected']:7} "
                  f"got {row['answered']:7} (warns: {row['warns']}){cause}")
    print(f"SCAN — controls: {len(summary['control_false_alarms'])} false alarms of "
          f"{summary['controls']} Washington documents")
    for row in control_rows:
        if row["false_alarm"] or not row["kind_held"]:
            print(f"    ! {row['file']:46} {row['answered']} / {row['kind']}")

    print(f"\nASK — cases: {ask_summary['exact']}/{ask_summary['cases']} exact · "
          f"{ask_summary['behaviour_matches']}/{ask_summary['cases']} warned "
          f"exactly when they should have")
    for signal, ratio in ask_summary["by_signal"].items():
        print(f"  {signal:26} {ratio}")
    for row in ask_case_rows:
        if not row["correct"]:
            print(f"    ! {row['id']:34} expected {row['expected']:7} "
                  f"got {row['answered']:7} (warns: {row['warns']})")
    print(f"ASK — controls: {ask_summary['control_false_alarms']} false alarms of "
          f"{ask_summary['controls']} questions that name no state")
    for question in ask_summary["control_false_alarm_ids"]:
        print(f"    ! {question}")

    print(f"\n${meter.cost_usd:.4f} · {meter.llm_calls} LLM calls")

    if args.write:
        RESULTS_PATH.write_text(json.dumps({
            "provenance": stamp(),
            "applied_state": APPLIED,
            "cost_usd": round(meter.cost_usd, 5),
            "scan": {"summary": summary, "cases": case_rows, "controls": control_rows},
            "ask": {"summary": ask_summary, "cases": ask_case_rows,
                    "controls": ask_control_rows},
        }, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
