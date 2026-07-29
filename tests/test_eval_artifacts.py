"""A cheap run must not destroy an expensive one (no API calls).

eval_real_formats writes the same results file with or without --scan. Splitting is
free and so gets run casually; the LLM passes cost money. A free run overwrote the
paid protections verdicts once already, leaving the write-up citing numbers that
were no longer in the artifact — so the carry-forward is pinned here.
"""

from evaluation.eval_real_formats import carry_paid_results


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
