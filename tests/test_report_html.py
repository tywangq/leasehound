"""The web panel's renderer: escaping, and agreement with the markdown one.

Two renderers over one report can drift, and the drift is invisible — the panel and
the downloaded .md would simply disagree about what the scan found. The tests here
are the ones that catch that, plus the escaping tests, because the panel is a
gr.HTML and report_html.py is therefore the whole XSS boundary.
"""

from html.parser import HTMLParser

from leasehound.report_html import render_report_html
from leasehound.scan import (
    SECTION_TITLES,
    SUMMARY_LABELS,
    render_report,
    summary_counts,
)

HOSTILE = "<script>alert(1)</script><img src=x onerror=alert(2)> \"q\" & 'a'"


def finding(index, verdict, clause="1. RENT. Tenant shall pay rent.",
            explanation="Plain language.", citations=("RCW 59.18.170",), urls=None):
    return {"index": index, "verdict": verdict, "clause": clause,
            "explanation": explanation, "citations": list(citations),
            "urls": urls if urls is not None else {"RCW 59.18.170": "https://app.leg.wa.gov/x"}}


class Parsed(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tags, self.handlers, self.hrefs = set(), [], []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        for key, value in attrs:
            if key.startswith("on"):
                self.handlers.append((tag, key))
            if key == "href":
                self.hrefs.append(value)


def test_hostile_clause_text_cannot_open_a_tag():
    html = render_report_html([finding(1, "red", clause=HOSTILE, explanation=HOSTILE)],
                              HOSTILE, "wa", [])
    parsed = Parsed(html)
    assert "script" not in parsed.tags
    assert "img" not in parsed.tags
    assert parsed.handlers == []
    # Present, but as text rather than as markup.
    assert "&lt;script&gt;" in html


def test_only_http_urls_become_links():
    hostile = render_report_html(
        [finding(1, "red", urls={"RCW 59.18.170": "javascript:alert(1)"})], "l.md", "wa", [])
    assert Parsed(hostile).hrefs == []
    assert "RCW 59.18.170" in hostile

    safe = render_report_html(
        [finding(1, "red", urls={"RCW 59.18.170": "https://app.leg.wa.gov/x"})], "l.md", "wa", [])
    assert Parsed(safe).hrefs == ["https://app.leg.wa.gov/x"]


def test_the_markdown_rides_along_for_the_copy_button():
    html = render_report_html([finding(1, "green")], "l.md", "wa", [], markdown="# report\n- a")
    # Escaped enough to keep the textarea closed, and no more: the copy handler reads
    # `.value`, which decodes entities back to the markdown that was passed in.
    assert '<textarea class="lh-md"># report\n- a</textarea>' in html
    assert "lh-md" not in render_report_html([finding(1, "green")], "l.md", "wa", [])


def test_both_renderers_agree_on_the_summary_row():
    findings = [finding(1, "red"), finding(2, "yellow"), finding(3, "green")]
    protections = [{"name": "Mold", "status": "missing", "requirement": "r", "citation": "c"}]
    markdown = render_report(findings, "l.md", "wa", protections)
    html = render_report_html(findings, "l.md", "wa", protections)
    for _key, label in summary_counts(findings, protections):
        assert label in markdown
        assert label in html


def test_both_renderers_show_the_same_sections_in_the_same_order():
    findings = [finding(1, "red"), finding(2, "yellow"), finding(3, "green")]
    protections = [{"name": "Mold", "status": "missing", "requirement": "r", "citation": "c"}]
    html = render_report_html(findings, "l.md", "wa", protections)
    markdown = render_report(findings, "l.md", "wa", protections)
    # Both read the titles from the same map, so this asserts order, not wording.
    needles = [SECTION_TITLES[k] for k in ("red", "yellow", "missing")]
    for surface in (html, markdown):
        positions = [surface.index(n) for n in needles]
        assert positions == sorted(positions), surface[:200]
    # Green is a footnote in both, never a section of its own.
    assert "lh-section lh-green" not in html
    assert SUMMARY_LABELS["green"] in html


def test_a_refused_document_gets_no_counts():
    html = render_report_html([], "resume.pdf", "wa", [], gate_flagged=True,
                              refused=True, clauses_total=12)
    assert "Not judged" in html
    assert "lh-chip" not in html
    assert "0 red flags" not in html
    assert "12 clauses" in html


def test_notices_come_before_the_findings():
    findings = [finding(1, "red")]
    html = render_report_html(findings, "ca_lease.pdf", "wa", [], gate_flagged=True,
                             clauses_total=9, jurisdiction="ca")
    for kind in ("legal", "jurisdiction", "gate"):
        assert f"lh-note-{kind}" in html
    assert html.index("lh-note-jurisdiction") < html.index("lh-finding")
