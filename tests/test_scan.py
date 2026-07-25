"""Report rendering and the parallel-scan cancellation contract (no API calls)."""

import time
from unittest.mock import patch

import leasehound.scan as scan
from leasehound.scan import base_section, count_verdicts, render_report


def finding(index: int, verdict: str, clause: str = "", **kw) -> dict:
    return {
        "index": index,
        "clause": clause or f"{index}. CLAUSE. Ordinary terms apply to this provision.",
        "verdict": verdict,
        "citations": kw.get("citations", []),
        "urls": kw.get("urls", {}),
        "explanation": kw.get("explanation", "Explanation."),
    }


def test_base_section_strips_subsections():
    assert base_section("RCW 59.18.150(6)") == "RCW 59.18.150"
    assert base_section("RCW 59.18.230(2)(c)") == "RCW 59.18.230"
    assert base_section("RCW 59.18.060") == "RCW 59.18.060"
    assert base_section("not a citation") == "not a citation"


def test_count_verdicts_tallies_all_three():
    findings = [finding(1, "red"), finding(2, "green"), finding(3, "red"), finding(4, "yellow")]
    assert count_verdicts(findings) == {"red": 2, "yellow": 1, "green": 1}


def test_report_sections_in_severity_order():
    findings = [finding(1, "green"), finding(2, "red"), finding(3, "yellow")]
    report = render_report(findings, "lease.md", "wa")
    assert report.index("🚩 Red") < report.index("⚠️ Yellow") < report.index("✅ Clear")
    assert "Clauses 1 —" in report


def test_long_preview_truncates_at_word_boundary():
    # Regression: a mid-word cut once turned "$75" into "$7".
    clause = "3. LATE CHARGES. " + "word " * 20 + "pay a late charge of $75, plus more terms here."
    report = render_report([finding(1, "red", clause=clause)], "lease.md", "wa")
    preview_line = next(line for line in report.splitlines() if line.startswith("### Clause 1:"))
    assert preview_line.endswith(" …")
    last_word = preview_line.removesuffix(" …").split()[-1]
    assert last_word in clause.split()


def test_short_preview_gets_no_ellipsis():
    report = render_report([finding(1, "red", clause="1. SHORT. Brief clause text.")], "l.md", "wa")
    assert "Brief clause text.\n" in report + "\n"
    assert "…" not in report


def test_missing_protections_section_lists_only_missing():
    protections = [
        {"name": "Mold information", "requirement": "Provide mold info.",
         "citation": "RCW 59.18.060", "status": "missing", "evidence": ""},
        {"name": "Deposit terms", "requirement": "State withholding terms.",
         "citation": "RCW 59.18.260", "status": "present", "evidence": "Clause 4"},
    ]
    report = render_report([finding(1, "green")], "lease.md", "wa", protections)
    assert "🔍 1 missing protections" in report
    assert "Mold information" in report
    assert "Deposit terms" not in report.split("Missing protections")[1]


def test_closing_scan_generator_cancels_queued_clauses():
    started = []

    def slow_scan_clause(clause, index, config):
        started.append(index)
        time.sleep(0.05 if index == 1 else 0.5)  # first verdict lands while 7 still queue
        return {"index": index, "verdict": "green"}

    with patch.object(scan, "scan_clause", slow_scan_clause):
        gen = scan.scan_clauses([f"clause {i}" for i in range(1, 16)], config=None)
        next(gen)  # one verdict consumed, then the user calls it off
        t0 = time.time()
        gen.close()
        close_seconds = time.time() - t0

    time.sleep(0.7)  # let in-flight workers drain
    assert close_seconds < 0.5, "close() must not block on the pool"
    # 8 filled the pool at submit; at most one more sneaks in when the fast
    # clause frees a worker. The rest of the queue must never start.
    assert len(started) <= 9
