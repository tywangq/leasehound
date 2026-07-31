"""Windowing the required-protections pass (no API calls).

The protections pass answers "what does this lease FAIL to include", and absence
is a claim about the whole document. The single 24k-character prompt it used to
send was therefore not a partial answer but a wrong one: at 39,996 characters the
49-clause HUD model lease overflowed it, so the published result described 28 of
its clauses and reported the rest missing without ever reading them.

Two properties matter here. Any lease that already fitted must produce the exact
same prompt it did before (otherwise every published gold number is up for
re-measurement), and the merge across windows has to let a later window's
"present" overrule an earlier window's "missing".
"""

import threading

from leasehound import scan
from leasehound.scan import (
    PROTECTIONS_WINDOW_CHARS,
    ProtectionStatus,
    make_protections_prompt,
    merge_protection_checks,
    protection_windows,
)


def status(index: int, state: str, evidence: str = "") -> ProtectionStatus:
    return ProtectionStatus(index=index, status=state, evidence=evidence)


def test_a_lease_that_fits_is_one_window_holding_every_clause():
    clauses = [f"{i}. CLAUSE. Ordinary terms." for i in range(1, 40)]
    windows = protection_windows(clauses)
    assert len(windows) == 1
    # Byte-identical to what the pre-windowing code joined and sent, which is why
    # 0 of the 48 labelled/example documents can move: all of them fit.
    assert windows[0] == "\n\n".join(clauses)


def test_a_long_lease_is_split_without_losing_or_cutting_a_clause():
    clauses = [f"{i}. CLAUSE. " + "x" * 900 for i in range(1, 121)]
    windows = protection_windows(clauses)
    assert len(windows) > 1
    assert all(len(w) <= PROTECTIONS_WINDOW_CHARS for w in windows)
    # Every clause appears in exactly one window, whole.
    rejoined = [c for w in windows for c in w.split("\n\n")]
    assert rejoined == clauses


def test_the_prompt_no_longer_truncates_the_text_it_is_handed():
    # The window function is the single place length is bounded. A second, silent
    # slice inside the prompt would re-open the exact hole this closes.
    text = "y" * (PROTECTIONS_WINDOW_CHARS + 5000)
    assert text in make_protections_prompt(text)


def test_present_in_any_window_beats_missing_in_another():
    # The real shape of the bug: window 1 sees the deposit clause and reports the
    # withholding terms absent; window 3 is where the lease states them.
    merged = merge_protection_checks([
        [status(1, "missing"), status(2, "not_applicable")],
        [status(1, "not_applicable"), status(2, "not_applicable")],
        [status(1, "present", "Clause 88"), status(2, "missing")],
    ])
    by_index = {c.index: c for c in merged}
    assert by_index[1].status == "present"
    assert by_index[1].evidence == "Clause 88", "the evidence must come from the window that found it"
    assert by_index[2].status == "missing"


def test_missing_needs_every_window_to_have_looked():
    # not_applicable means the precondition never held. It can only survive if no
    # window saw one hold — otherwise the item genuinely applies and is absent.
    assert merge_protection_checks([
        [status(1, "not_applicable")], [status(1, "not_applicable")],
    ])[0].status == "not_applicable"
    assert merge_protection_checks([
        [status(1, "not_applicable")], [status(1, "missing")],
    ])[0].status == "missing"


def test_merge_returns_items_in_checklist_order():
    merged = merge_protection_checks([[status(3, "present"), status(1, "missing")],
                                      [status(2, "missing")]])
    assert [c.index for c in merged] == [1, 2, 3]


def test_an_empty_document_still_yields_one_window():
    # scan_lease raises before this on a no-text document, but a bare [] must not
    # silently skip the pass and report zero missing protections.
    assert protection_windows([]) == [""]


def test_windows_run_concurrently_but_merge_in_document_order(monkeypatch):
    """Both halves matter, and they pull against each other.

    The windows used to run serially while clause judging used an 8-way pool, so
    the project's two long-document passes obeyed opposite latency rules. Making
    them concurrent is only safe if order survives: merge_protection_checks breaks
    ties by first-seen, so if results came back in completion order, "which window
    found the evidence" would depend on which API call returned first.
    """
    clauses = [f"{i}. CLAUSE. " + "x" * 900 for i in range(1, 121)]
    windows = protection_windows(clauses)
    assert len(windows) > 2, "the fixture has to overflow for this to test anything"

    started = threading.Barrier(len(windows), timeout=5)
    seen: list[str] = []

    def blocking_window(text, meter=None):
        # Deadlocks unless every window is in flight at once — a serial loop can
        # never fill the barrier, so this fails closed rather than by timing.
        started.wait()
        seen.append(text)
        index = windows.index(text) + 1
        return [status(1, "present", f"window {index}")]

    monkeypatch.setattr(scan, "check_protections_window", blocking_window)
    results = scan.check_protections(clauses)

    assert sorted(seen) == sorted(windows), "every window must still be read"
    # First-seen wins, so the merged evidence has to name window 1 no matter which
    # call the provider answered first.
    assert results[0]["evidence"] == "window 1"
