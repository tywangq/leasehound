"""The statute drift check: does it notice real changes without crying wolf?

Two failure modes matter and they pull in opposite directions. A check that
misses an amendment lets the scanner cite repealed law. A check that reports
drift every week gets muted, which is the same thing more slowly. The tests
below cover the comparison itself; the shared-rendering contract with
fetch_corpus.py is what keeps false positives out, so it is pinned too.
"""

from scripts.check_corpus_drift import MAX_DIFF_LINES, diff_corpus, render_live, report

SECTION = "rcw-59-18-230.md"


def test_identical_corpora_do_not_drift():
    side = {SECTION: "# RCW 59.18.230\n\nWaiver of chapter provisions.\n"}
    assert not diff_corpus(side, dict(side)).drifted


def test_a_changed_section_is_reported_with_both_versions_in_the_diff():
    committed = {SECTION: "# RCW 59.18.230\n\nLate fees after five days.\n"}
    live = {SECTION: "# RCW 59.18.230\n\nLate fees after seven days.\n"}
    result = diff_corpus(live, committed)
    assert result.drifted
    assert result.changed == [SECTION]
    assert not result.added and not result.removed
    assert "-Late fees after five days." in result.diffs[SECTION]
    assert "+Late fees after seven days." in result.diffs[SECTION]


def test_new_and_repealed_sections_are_separated_from_edits():
    # A session can add a section and repeal another; conflating either with an
    # edit would send the wrong checklist to whoever picks up the issue.
    result = diff_corpus({"rcw-59-18-999.md": "new"}, {"rcw-59-18-010.md": "old"})
    assert result.added == ["rcw-59-18-999.md"]
    assert result.removed == ["rcw-59-18-010.md"]
    assert result.changed == []
    assert result.drifted


def test_long_diffs_are_truncated_so_the_issue_body_stays_postable():
    committed = {SECTION: "\n".join(f"old line {i}" for i in range(200))}
    live = {SECTION: "\n".join(f"new line {i}" for i in range(200))}
    diff = diff_corpus(live, committed).diffs[SECTION].splitlines()
    assert len(diff) == MAX_DIFF_LINES + 1
    assert "more diff lines" in diff[-1]


def test_the_report_carries_the_diff_and_what_a_human_still_owes():
    committed = {SECTION: "five days\n"}
    live = {SECTION: "seven days\n"}
    body = report(diff_corpus(live, committed))
    assert SECTION in body
    assert "+seven days" in body
    # The corpus is legal ground truth: re-embedding is not the whole job.
    assert "CORPUS_SNAPSHOT" in body
    assert "eval_scan" in body
    assert "required-protections checklist" in body


def test_rendering_the_live_page_matches_the_committed_filename_convention():
    # This is the contract that keeps the check quiet: it renders through
    # fetch_corpus's own normalization, so identical text can never diff.
    page = (
        "<a name='59.18.999'>RCW 59.18.999</a>"
        "<p>Test section caption.</p><p>Body text of the section.</p>"
    )
    live = render_live(page)
    assert list(live) == ["rcw-59-18-999.md"]
    content = live["rcw-59-18-999.md"]
    assert content.startswith("# RCW 59.18.999 — Test section caption")
    assert "Body text of the section." in content
    assert "cite=59.18.999" in content
