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


# The two fields every `provenance.stamp()` carries, including `stamp(models=False)`
# for runs that call no model. Named here because a stamp arrives in two shapes and
# this guard used to see only one: the .json artifacts nest it under "provenance",
# while eval_generation spreads it into the row (`**stamp()`) so the fields land at
# top level. Looking only for the nested key made the guard blind to the shape that is
# actually in use — it read a fully stamped row as unstamped, and would have gone on
# saying so after a paid re-run stamped every row.
STAMP_KEYS = {"commit", "run_at"}


def is_stamped_dict(record: dict) -> bool:
    """Either shape counts: nested under `provenance`, or spread flat into the row."""
    return "provenance" in record or STAMP_KEYS <= record.keys()


def is_stamped(path: Path) -> bool:
    """A .jsonl is one row per run, so its stamp would have to be on every row."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return bool(rows) and all(is_stamped_dict(row) for row in rows)
    return is_stamped_dict(json.loads(text))


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


def test_no_speedup_claim_exceeds_the_pool_that_produces_it():
    """The README claimed 9x from a pool of 8 workers, which is not reachable.

    "a 15-clause lease scans in ~10 seconds instead of ~90" is a 9x claim, and
    MAX_PARALLEL_SCANS is 8 — a constant in a public file, three clicks from the README.
    A reader who divides is a reader who has found a number with no source. The ceiling
    is clauses / ceil(clauses / workers): 7.5x at 15 clauses, 6.5x at the log's mean.
    """
    root = Path(__file__).parent.parent
    from leasehound.scan import MAX_PARALLEL_SCANS

    for doc in ("README.md", "evaluation/README.md"):
        text = (root / doc).read_text()
        for claim in re.findall(r"(\d+(?:\.\d+)?)\s*(?:x|×)\b", text):
            factor = float(claim)
            # Only speedup-shaped claims are bounded by the pool; cost multiples are not.
            if factor <= MAX_PARALLEL_SCANS:
                continue
            context = text[max(0, text.index(claim + "x") - 120):text.index(claim + "x")] \
                if claim + "x" in text else ""
            assert not re.search(r"concurren|parallel|wall time|pool|faster", context, re.I), (
                f"{doc} claims {factor}x on a {MAX_PARALLEL_SCANS}-worker pool: {context[-90:]}")


def test_the_stamp_detector_recognises_the_stamp_the_evals_actually_write():
    """The guard above cannot fail for the reason it exists, which is worse than a
    wrong claim — a wrong claim gets corrected.

    `provenance.stamp()` returns a flat dict and `eval_generation` spreads it into the
    row (`**stamp()`), so the fields land at top level: `commit`, `run_at`,
    `judge_prompt`. `is_stamped` looked for a nested `"provenance"` key that nothing
    writes. The most recent row of generation_results.jsonl carries a full stamp today
    and is read as unstamped, and — the part that matters — when the paid re-run
    finally stamps every row, this guard would stay silent and the prose beside it
    would go quietly stale. That is the third time this section would have been wrong,
    and the reason the counting was automated in the first place.
    """
    from evaluation.provenance import stamp

    assert is_stamped_dict(stamp()), "a row carrying stamp() must read as stamped"
    assert is_stamped_dict({"provenance": stamp()}), "the nested shape must still count"
    assert not is_stamped_dict({"n": 82, "consistent": 1.0}), "a bare result is not stamped"


def test_the_newest_generation_row_is_seen_for_what_it_is():
    """Concrete instance of the above, on the file that actually has the problem."""
    rows = [json.loads(line) for line
            in (EVAL_DIR / "generation_results.jsonl").read_text().splitlines() if line.strip()]
    assert is_stamped_dict(rows[-1]), (
        "the closed-book row carries commit + judge_prompt + run_at and must be "
        "recognised as stamped")
    assert not is_stamped_dict(rows[0]), (
        "the row the README's 82/82 comes from carries no stamp, and the prose says so")


def test_the_stamp_names_the_index_a_number_was_measured_against():
    """The input that hid the longest.

    Models, prompts and the corpus snapshot were stamped from the start. The index
    they are queried against was not, and two builds of it from the same corpus are
    not the same index: the first CI run with an API key rebuilt the store and scored
    section_hit@1 at 0.667 where the committed artifact says 0.833 — on the
    retrieval-only eval, which makes no completions. Every other field of the stamp
    was identical, so nothing in the artifact could have explained the gap.
    """
    from evaluation.provenance import stamp

    # The field is always present; whether it is populated depends on a store existing,
    # and `None` is the honest answer where one does not. This test asserted a store
    # instead, passed on a laptop and failed on the first push — CI's free `test` job
    # has no vector_db/.
    assert "index" in stamp()


def test_a_degraded_index_says_so_rather_than_looking_clean():
    """`degraded` counts chunks over MAX_CHUNK_CHARS — sections whose validation never
    passed and which ingestion accepted anyway.

    Driven with documents rather than a store, so it runs everywhere. The thing worth
    asserting is the arithmetic, and the arithmetic does not need Chroma.
    """
    from evaluation.provenance import fingerprint_documents
    from leasehound.ingest import MAX_CHUNK_CHARS

    clean = fingerprint_documents(["a" * 100, "b" * 200])
    assert clean == {"collection": "wa_reference", "chunks": 2,
                     "digest": clean["digest"], "degraded": 0}

    # One section the chunker never managed to split, of the kind ingestion accepts
    # after three failed validations — RCW 59.18.150 came back as 9,959 characters.
    damaged = fingerprint_documents(["a" * 100, "x" * (MAX_CHUNK_CHARS + 1)])
    assert damaged["degraded"] == 1, "an unsplit section must not read as a clean store"


def test_two_builds_of_the_same_corpus_are_told_apart():
    """The whole purpose: same commit, same models, same corpus, different index."""
    from evaluation.provenance import fingerprint_documents

    shipped = fingerprint_documents(["rent is due", "entry needs notice"])
    rebuilt = fingerprint_documents(["rent is due", "entry needs noticeentry"])
    assert shipped["digest"] != rebuilt["digest"]
    # And order must not matter — chunk order is not a property of the index.
    assert fingerprint_documents(["b", "a"])["digest"] == fingerprint_documents(["a", "b"])["digest"]
