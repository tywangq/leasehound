"""Which commits are allowed to spend money (no API calls).

`eval.yml` runs paid evals on pushes that touch `leasehound/**`, minus an exclude
list. That list is the only thing standing between a stylesheet edit and a $0.23
gold-set re-run, and a list maintained by eye goes stale the first time somebody adds
a module — silently, in the direction of the bill.

So the directory is the source of truth and this holds the workflow to it: every file
under `leasehound/` must be classified exactly once, as either presentation (excluded,
cannot change a verdict) or pipeline (worth paying to re-measure). A new module fails
here until somebody says which it is.

The classification is checkable rather than a matter of taste: nothing excluded here
is reachable from `judge_fingerprint()`, which digests the judge prompt and its
response schema — the two things every published verdict number depends on.

The second half of the file pins the other axis. The path filter decides *whether* a
run happens; the per-step conditions decide what it *costs* — a ~$0.0002 canary per
push, a ~$0.23 tier weekly, ~$1.00 on dispatch. Neither axis is visible without
reading YAML, and a condition that reverts to the expensive default fails nothing.
It just bills.
"""

import re
from pathlib import Path

REPO = Path(__file__).parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "eval.yml"
PACKAGE = REPO / "leasehound"

# Presentation and plumbing: how a result is shown, served, or logged. None of it can
# change what a verdict IS. `metrics.py` is here because it records what a scan cost
# after the scan is over; `api.py` because a typed HTTP envelope around `scan_steps`
# cannot alter what `scan_steps` returns.
PRESENTATION = {
    "app.py", "api.py", "report_html.py", "metrics.py", "style.css",
    "assets/copy.svg", "assets/download.svg", "assets/favicon.png", "assets/trash.svg",
}

# Everything a published number depends on: the judge and its orchestration, the
# retrieval pipeline, clause splitting, corpus ingestion, the jurisdiction call.
PIPELINE = {
    "__init__.py", "answer.py", "bm25.py", "ingest.py", "ingest_naive.py",
    "jurisdiction.py", "retrieval.py", "scan.py", "upload.py",
}


def package_files() -> set[str]:
    return {str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


def excluded_in_workflow() -> set[str]:
    text = WORKFLOW.read_text(encoding="utf-8")
    return {m.removesuffix("/**") for m in
            re.findall(r'-\s*"!leasehound/([^"]+)"', text)}


def test_every_module_is_classified():
    """A new file under leasehound/ has to be called one thing or the other."""
    unclassified = package_files() - PRESENTATION - PIPELINE
    assert not unclassified, (
        f"new files under leasehound/ that nobody has classified: {sorted(unclassified)}. "
        "Add to PRESENTATION (and to eval.yml's exclude list) if it cannot change a "
        "verdict, or to PIPELINE if a push touching it should re-measure.")


def test_the_two_classes_do_not_overlap():
    assert not (PRESENTATION & PIPELINE)


def test_the_workflow_excludes_exactly_the_presentation_layer():
    """The list in eval.yml and the classification here must be the same list.

    Both directions matter. Something presentation that the workflow forgot to exclude
    costs money on every CSS commit; something pipeline that the workflow excludes
    stops being re-measured, which is the failure that ships a regression.
    """
    excluded = excluded_in_workflow()
    # `assets/**` is excluded as a directory, so compare at that granularity.
    presentation_dirs = {f.split("/")[0] + "/**" if "/" in f else f for f in PRESENTATION}
    normalised = {e + "/**" if e == "assets" else e for e in excluded}
    assert normalised == {p.removesuffix("/**") + "/**" if p.endswith("/**") else p
                          for p in presentation_dirs}, (
        f"eval.yml excludes {sorted(excluded)}; this file classifies "
        f"{sorted(presentation_dirs)} as presentation")


def test_the_fingerprint_is_built_from_the_judge_and_nothing_else():
    """What makes the exclusions safe, asserted on the thing that actually decides it.

    The first version of this test asserted that no excluded module is imported by
    `scan.py`, and `metrics.py` failed it — scan.py imports `UsageMeter` and
    `log_scan`. But being imported is not the same as being able to change a verdict:
    the meter accumulates counters while a scan runs and `log_scan` writes a row after
    it ends. The proxy was wrong, not the classification.

    So this asserts the real claim instead — that `judge_fingerprint()` digests
    exactly two things, the judge prompt and its response schema. Recomputed here from
    those two ingredients alone: if the digest ever starts depending on anything else,
    this fails, and the exclusion list has to be re-argued.

    Scoping honesty: this proves what the FINGERPRINT covers. That `metrics.py` cannot
    change a verdict is a claim about the code's shape, argued in PRESENTATION's
    comment, not proved here.
    """
    import hashlib
    import json as json_

    from leasehound.scan import ClauseVerdict, judge_fingerprint, make_judge_prompt

    material = make_judge_prompt("", []) + json_.dumps(
        ClauseVerdict.model_json_schema(), sort_keys=True)
    assert judge_fingerprint() == hashlib.sha256(
        material.encode("utf-8")).hexdigest()[:12]


# --- which tier runs when ---------------------------------------------------------
#
# The path filter above decides whether a run happens at all. These decide what it
# costs when it does. Both are load-bearing and neither is visible from the repo
# without reading YAML, so they get held here: a condition that quietly reverts to
# running the paid tier on every push does not fail anything, it just bills.

# ~$0.0002: one embedding per labelled clause, no completions.
CANARY = "Scan-mode retrieval eval"
# ~$0.23 together. Weekly and on dispatch.
PAID_WEEKLY = {"Scan eval (gold set)", "Retrieval eval", "Injection resistance (5 payloads)"}
# ~$1.00 together. Dispatch only.
PAID_DISPATCH = {"Generation eval (manual runs only)",
                 "Scan eval — synthetic silver set (manual runs only)"}


def step_conditions() -> dict[str, str]:
    """Every named step in eval.yml, mapped to its `if:` expression."""
    text = WORKFLOW.read_text(encoding="utf-8")
    steps = {}
    for block in re.split(r"\n      - name: ", text)[1:]:
        name = block.split("\n", 1)[0].strip()
        match = re.search(r"^        if: (.+)$", block, re.M)
        steps[name] = match.group(1).strip() if match else ""
    return steps


def test_only_the_canary_runs_on_a_push():
    """The whole point of the tiering. $0.0002 per commit instead of $0.23."""
    conditions = step_conditions()
    assert "github.event_name" not in conditions[CANARY], (
        "the canary must run on every qualifying push — it is the per-commit signal")
    for name in PAID_WEEKLY:
        assert "github.event_name != 'push'" in conditions[name], (
            f"{name!r} costs money and must not run per push")


def test_the_dearest_two_are_dispatch_only():
    conditions = step_conditions()
    for name in PAID_DISPATCH:
        assert "github.event_name == 'workflow_dispatch'" in conditions[name], (
            f"{name!r} is the ~$0.40-$0.60 tier and stays manual")


# Steps that call nothing paid: setup, and the one that prints which index the
# scores are against. "Build vector store" is here because it is not a tier — it runs
# only on a cache miss, and what it costs is the ~$0.10 of ingestion, not an eval.
FREE_STEPS = {
    "Gate on API key", "Install", "Restore vector store",
    "Build vector store (cache miss)", "Say which index these scores are against",
}


def test_every_paid_step_is_classified():
    """A new step added without a tier defaults to running per push, which is the
    expensive default. This is the line that notices."""
    named = set(step_conditions()) - {CANARY}
    unclassified = named - PAID_WEEKLY - PAID_DISPATCH - FREE_STEPS
    assert not unclassified, (
        f"eval.yml steps with no declared tier: {sorted(unclassified)}")


def test_the_weekly_tier_has_something_to_run_on():
    """The paid tier is excluded from push, so without a schedule it would only ever
    run when somebody remembered to click dispatch."""
    assert re.search(r"^  schedule:", WORKFLOW.read_text(encoding="utf-8"), re.M), (
        "the paid tier no longer runs on push and needs a cadence")
