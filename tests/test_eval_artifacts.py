"""The artifacts and the prose about them must agree (no API calls).

Two failures, both of which happened.

A cheap run must not destroy an expensive one. eval_real_formats writes the same
results file with or without --scan. Splitting is free and so gets run casually; the
LLM passes cost money. A free run overwrote the paid protections verdicts once
already, leaving the write-up citing numbers that were no longer in the artifact —
so the carry-forward is pinned here.

And the inventory must not be hand-maintained. evaluation/README.md says which
artifacts carry a provenance stamp; it claimed all of them did, and the correction
that replaced that claim miscounted by one. Counting files is not something to do by
eye twice, so the tests below count them and hold the prose to it.
"""

import json
import re
from itertools import dropwhile, takewhile
from pathlib import Path

from evaluation.eval_real_formats import carry_paid_results

EVAL_DIR = Path(__file__).parent.parent / "evaluation"
README = EVAL_DIR / "README.md"
# Question sets, labelled clause sets, and the checklist register: things the evals
# READ, not results they write. The register is here because it holds decisions rather
# than measurements — tests/test_checklist_register.py is what keeps it honest.
INPUT_SETS = {"tests.jsonl", "tests_adversarial.jsonl", "permissive_pairs.jsonl",
              "jurisdiction_cases.jsonl", "jurisdiction_ask_cases.jsonl",
              "checklist_register.json"}


def result_files() -> list[Path]:
    return sorted(p for p in EVAL_DIR.iterdir()
                  if p.suffix in {".json", ".jsonl"} and p.name not in INPUT_SETS)


def is_stamped(path: Path) -> bool:
    """A .jsonl is one row per run, so its stamp would have to be on every row."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return bool(rows) and all("provenance" in row for row in rows)
    return "provenance" in json.loads(text)


def paid_artifact() -> dict:
    return {
        "provenance": {"commit": "618c00d"},
        "results": [{
            "file": "hud_90105a_model_lease.pdf", "chars": 39996, "clauses": 49,
            "gate_accepted_as_lease": True,
            "protections_missing": ["Fire safety information", "Mold information"],
            "protections_present": 2, "llm_calls": 2, "cost_usd": 0.00304,
        }],
    }


def free_run() -> list[dict]:
    return [{"file": "hud_90105a_model_lease.pdf", "chars": 39996, "clauses": 49,
             "protection_windows": 2}]


def test_a_free_run_keeps_the_paid_findings():
    carried = carry_paid_results(free_run(), paid_artifact())[0]
    assert carried["protections_missing"] == ["Fire safety information", "Mold information"]
    assert carried["protections_present"] == 2
    assert carried["cost_usd"] == 0.00304
    # And says where they came from, so nobody reads them as this run's measurement.
    assert carried["paid_fields_carried_from"] == "618c00d"


def test_the_free_run_still_updates_what_it_actually_measured():
    carried = carry_paid_results(free_run(), paid_artifact())[0]
    assert carried["protection_windows"] == 2, "free measurements are this run's own"


def test_a_paid_run_overwrites_rather_than_carrying():
    fresh = [{"file": "hud_90105a_model_lease.pdf", "protections_missing": [],
              "protections_present": 5, "cost_usd": 0.009}]
    carried = carry_paid_results(fresh, paid_artifact())[0]
    assert carried["protections_missing"] == [], "a new paid measurement wins"
    assert "paid_fields_carried_from" not in carried


def test_a_document_with_no_history_is_left_alone():
    carried = carry_paid_results(
        [{"file": "newly_added.pdf", "chars": 100}], paid_artifact())[0]
    assert set(carried) == {"file", "chars"}


def test_no_previous_artifact_is_not_an_error():
    assert carry_paid_results(free_run(), None) == free_run()


def test_every_result_artifact_is_named_in_the_readme():
    text = README.read_text(encoding="utf-8")
    unmentioned = [p.name for p in result_files() if p.name not in text]
    assert not unmentioned, (
        f"{unmentioned} sit in evaluation/ with nothing in evaluation/README.md naming "
        f"them. An unexplained results file is worse than no results file.")


def test_the_stamped_counts_match_the_directory():
    # Whitespace-normalised, so rewrapping the paragraph does not fail the test.
    text = " ".join(README.read_text(encoding="utf-8").split())
    json_files = [p for p in result_files() if p.suffix == ".json"]
    stamped = [p for p in json_files if is_stamped(p)]
    claim = (f"{len(json_files)} `.json` results, of which {len(stamped)} carry "
             f"a `provenance` stamp")
    assert claim in text, (
        f"evaluation/README.md should say: {claim!r}. Counted from disk, not from "
        f"the prose — update the sentence, not this test.")


def test_the_unstamped_artifacts_are_listed_by_name():
    """A count alone is useless: the reader needs to know WHICH numbers lack a stamp."""
    lines = README.read_text(encoding="utf-8").splitlines()
    marker = next(i for i, line in enumerate(lines) if line.rstrip().endswith("do not:"))
    # The blank line markdown needs between a paragraph and its list is not content.
    after = dropwhile(lambda line: not line.strip(), lines[marker + 1:])
    listed = {line.split("`")[1] for line in
              takewhile(lambda line: line.startswith("- `"), after)}
    unstamped = {p.name for p in result_files() if p.suffix == ".json" and not is_stamped(p)}
    assert unstamped == listed, (
        f"the bulleted list in evaluation/README.md and the directory disagree: "
        f"only on disk {sorted(unstamped - listed)}, only in the prose "
        f"{sorted(listed - unstamped)}")


def test_the_jsonl_logs_are_not_claimed_as_stamped():
    """Guards the other direction: prose says none of them carry a stamp."""
    logs = [p for p in result_files() if p.suffix == ".jsonl"]
    assert logs, "the append-style logs the README describes have gone missing"
    assert not [p.name for p in logs if is_stamped(p)], (
        "a .jsonl now stamps every row, so evaluation/README.md's claim that none "
        "of them do is stale")


def test_the_readme_reports_the_number_of_tests_that_exist():
    """The README said 212 for long enough that it had drifted to 260 collected.

    A count in prose is a fact about the repository stated in a second place, and this
    is the cheapest way to keep the two honest. Test FUNCTIONS, not collected items:
    parametrised cases expand at collection time, and a number that changes when someone
    adds a parameter is a number nobody will maintain.
    """
    root = Path(__file__).parent.parent
    written = sum(
        len(re.findall(r"^def test_", path.read_text(), re.M))
        for path in sorted((root / "tests").glob("test_*.py"))
    )
    readme = (root / "README.md").read_text()
    claimed = re.search(r"# (\d+) test functions across (\d+) files", readme)
    assert claimed, "the README no longer states a test count in the form this checks"
    assert int(claimed.group(1)) == written, (
        f"README says {claimed.group(1)} test functions, the suite has {written}")
    files = len(list((root / "tests").glob("test_*.py")))
    assert int(claimed.group(2)) == files, (
        f"README says {claimed.group(2)} test files, there are {files}")


def test_the_readme_quotes_the_cost_band_it_says_it_quotes():
    """The artifact holds two populations, and the README's argument depends on which.

    `all_scans` is every logged scan — 5 to 49 clauses — and `scans_9_to_15_clauses` is
    the homogeneous band. The cost section quotes the band on purpose: its next
    paragraph produces the 49-clause HUD lease as a contrast, which only means anything
    if the headline excluded it. Reading the wrong key looks like the README is stale by
    14 scans and a tenth of a cent, and "fixing" it deletes the argument.
    """
    root = Path(__file__).parent.parent
    summary = json.loads((root / "evaluation" / "scan_cost_summary.json").read_text())
    band = summary["scans_9_to_15_clauses"]
    readme = (root / "README.md").read_text()

    quoted = re.search(
        r"Across the ([\d,]+) scans in the log \((\d+)–(\d+) clauses each"
        r"[^*]*\*\*mean ≈ \$([\d.]+)/scan, p50 ([\d.]+) s, p95 ([\d.]+) s, max ([\d.]+) s\*\*",
        readme)
    assert quoted, "the cost paragraph no longer states its numbers in the form this checks"
    n, low, high, mean, p50, p95, worst = quoted.groups()
    assert int(n.replace(",", "")) == band["scans"], "the README quotes a different population"
    assert float(mean) == round(band["mean_cost_usd"], 4)
    assert (float(p50), float(p95), float(worst)) == (
        band["p50_seconds"], band["p95_seconds"], band["max_seconds"])
    # The stated clause range must contain the band's own, or the sentence describes a
    # set it did not measure.
    assert int(low) <= band["clauses_min"] and int(high) >= band["clauses_max"]

    headline = re.search(r"\| (\d+) logged scans \((\d+)–(\d+) clauses\) \| \*\*≈ \$([\d.]+)/scan",
                         readme)
    assert headline, "the evidence table no longer states the cost row in the form this checks"
    assert int(headline.group(1)) == band["scans"]
    assert float(headline.group(4)) == round(band["mean_cost_usd"], 3)
