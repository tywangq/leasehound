"""HTML rendering of a scan report, for the web panel only.

Why a second renderer at all: `render_report` in scan.py is a contract with three
other consumers — the API returns it as `report_markdown`, the CLI writes it to a
.md file, and eval_injection.py greps it — so it cannot become HTML. The web panel
is the one surface that wants structure a Markdown string cannot carry: verdict
chips, a bordered finding with a coloured rail, a serif clause quote.

Both renderers read the same `findings` list and share every decision that is not
markup — section order, summary labels, the disclaimer, where a long clause gets
cut — from scan.py. If a label or an order needs to change, it changes there once.

Escaping: every interpolated value is escaped here, without exception. The clause
text comes from a document a stranger uploaded and the explanation comes from a
model that has been shown that document, so both are untrusted input on a public
demo. The panel is a gr.HTML, which does not sanitize, so this module is the whole
boundary — hence the single `esc` helper and the test that feeds it a payload.
"""

from __future__ import annotations

import re
from html import escape

from leasehound.jurisdiction import UNKNOWN_JURISDICTION, jurisdiction_mismatch
from leasehound.scan import (
    CLEAR_NOTE,
    GATE_REFUSED,
    GATE_WARNING,
    MISSING_INTRO,
    SECTION_TITLES,
    VERDICT_ORDER,
    caveat,
    clause_label,
    clause_preview,
    jurisdiction_warning,
    partial_scan_notice,
    provenance_parts,
    statute_url,
    summary_counts,
)

# Only http(s) links reach an href. The urls come from the corpus rather than from a
# model, but an href is the one place in this document where a `javascript:` string
# would become executable, and a whitelist is cheaper than trusting the pipeline.
SAFE_SCHEMES = ("https://", "http://")


def esc(value: object) -> str:
    return escape(str(value), quote=True)


def _chip(key: str, label: str) -> str:
    return f'<span class="lh-chip lh-{key}"><i></i>{esc(label)}</span>'


# The notices are authored as markdown, because their first home was a .md report:
# each opens with "> " and a status emoji and marks its lead clause with **bold**.
# Escaping them verbatim printed those markers on the page, so they are converted —
# escape first, then reintroduce only <strong>, so nothing in the source text can
# open a tag. The emoji goes: the coloured rail on .lh-note says the same thing, and
# an emoji inside a styled callout is the redundancy this panel was rewritten to drop.
# The markdown keeps it, having no CSS to carry the signal.
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)


def _from_markdown(text: str) -> str:
    body = text.lstrip()
    if body.startswith(">"):
        body = body[1:].lstrip()
    if body and not body[0].isascii():
        body = body.split(" ", 1)[1].lstrip() if " " in body else ""
    return BOLD.sub(r"<strong>\1</strong>", esc(body))


def _note(kind: str, text: str) -> str:
    return f'<p class="lh-note lh-note-{kind}">{_from_markdown(text)}</p>'


def _citation(section: str, url: str) -> str:
    if url.startswith(SAFE_SCHEMES):
        return (f'<a class="lh-cite" href="{esc(url)}" target="_blank" '
                f'rel="noopener noreferrer">{esc(section)}</a>')
    return f'<span class="lh-cite">{esc(section)}</span>'


def _finding(f: dict, verdict: str) -> str:
    cites = "".join(_citation(s, f["urls"].get(s, "")) for s in f["citations"])
    cite_row = f'<p class="lh-cites">{cites}</p>' if cites else ""
    # clause_label, from scan.py, so this and the .md report agree about when a clause
    # gets a label at all — and never print a number that contradicts the one in the
    # quote directly below it. See its docstring for the off-by-one that prompted this.
    label = clause_label(f)
    head = f'<p class="lh-finding-head">{esc(label)}</p>' if label else ""
    return (
        f'<div class="lh-finding lh-{verdict}">'
        f"{head}"
        f'<blockquote class="lh-clause">{esc(clause_preview(f["clause"]))}</blockquote>'
        f'<p class="lh-explain">{esc(f["explanation"])}</p>'
        f"{cite_row}"
        f"</div>"
    )


def _head(source: str, status: str, counts_row: str) -> str:
    """Title with the file it is a report OF, then the chips. Nothing else.

    The title used to be "Scan report" alone with the file name pushed into a meta row
    that also carried "Judged against WA law" and today's date, while the legal callout
    below said "Judged against RCW 59.18 as of 2026-07-25" — two "judged against"
    claims with two different dates three lines apart. Merging them into one provenance
    line fixed the contradiction but left the line here, above the findings, where it
    was still the second thing anyone read. It says who judged the lease and against
    which snapshot: nobody needs that before the findings, and everybody wants it when
    quoting them. So it moved to the foot, beside the caveat (see render_report_html).

    `status` is the one thing that does belong in that slot: on a refused document it
    says "Not judged", which changes what every number below it would have meant — and
    there are no numbers below it.
    """
    return (
        '<div class="lh-head">'
        + '<p class="lh-title">Scan report'
        + f'<span class="lh-file">{esc(source)}</span></p>'
        + (f'<p class="lh-prov">{status}</p>' if status else "")
        + (f'<p class="lh-chips">{counts_row}</p>' if counts_row else "")
        + "</div>"
    )


def _provenance(state: str) -> str:
    """The parts come from scan.py, so this and the .md file cannot describe the same
    scan differently — the dots are the only thing this function decides."""
    return '<span class="lh-dot">·</span>'.join(esc(part) for part in provenance_parts(state))


def render_report_html(
    findings: list[dict], source: str, state: str, protections: list[dict] | None = None,
    gate_flagged: bool = False, clauses_total: int | None = None,
    refused: bool = False, jurisdiction: str = UNKNOWN_JURISDICTION,
    markdown: str = "",
) -> str:
    """The same report as `render_report`, as HTML for the panel.

    `markdown` rides along in a hidden field so the copy button keeps copying
    markdown: its handler is client-side, and a gr.State is only readable on the
    server. Without it, copy would put markup on the clipboard.
    """
    parts: list[str] = []

    if refused:
        # Not a report with zero findings — no findings exist. Rendering the usual
        # summary row here would show "0 red flags", which reads as a clean bill of
        # health for a document that was never judged.
        parts.append(_head(source, '<strong class="lh-notjudged">Not judged</strong>', ""))
        parts.append(_note("gate", GATE_REFUSED))
        parts.append(
            f'<p class="lh-tail">Split into {esc(clauses_total)} clauses, '
            f"<strong>0 judged</strong>. Nothing was spent beyond the one call that "
            f"classified the document.</p>"
        )
        return _wrap(parts, markdown)

    chips = "".join(_chip(key, label) for key, label in summary_counts(findings, protections))
    parts.append(_head(source, "", chips))

    # Above the other two notices, because it is the only one of the three that can
    # make every verdict in the report wrong rather than merely incomplete.
    if jurisdiction_mismatch(jurisdiction, state):
        parts.append(_note("jurisdiction", jurisdiction_warning(jurisdiction, state)))
    if gate_flagged:
        parts.append(_note("gate", GATE_WARNING))
    if clauses_total is not None and clauses_total > len(findings):
        parts.append(_note("partial", partial_scan_notice(len(findings), clauses_total)))

    for verdict in VERDICT_ORDER:
        matching = [f for f in findings if f["verdict"] == verdict]
        if not matching or verdict == "green":
            continue
        parts.append(f'<h2 class="lh-section lh-{verdict}">{SECTION_TITLES[verdict]}</h2>')
        parts.extend(_finding(f, verdict) for f in matching)

    missing = [p for p in (protections or []) if p["status"] == "missing"]
    if missing:
        parts.append(f'<h2 class="lh-section lh-missing">{SECTION_TITLES["missing"]}</h2>')
        parts.append(f'<p class="lh-explain">{esc(MISSING_INTRO)}</p>')
        parts.append('<ul class="lh-missing-list">')
        # _citation, the same helper the findings use, so a statute reference behaves
        # the same wherever it appears. These used to be a bare <span>: the checklist
        # retrieves nothing and so had no URL to carry, which is a fact about this
        # module's plumbing and not about the citation. statute_url derives it.
        parts.extend(
            f'<li><strong>{esc(p["name"])}</strong> — {esc(p["requirement"])} '
            f'{_citation(p["citation"], statute_url(p["citation"]))}</li>'
            for p in missing
        )
        parts.append("</ul>")

    clear = [str(f["index"]) for f in findings if f["verdict"] == "green"]
    if clear:
        parts.append(
            f'<p class="lh-tail">Clauses {esc(", ".join(clear))} — {esc(CLEAR_NOTE)}</p>'
        )

    # Last, and small. It was a bordered callout directly under the summary row, which
    # gave the one sentence on the page that says nothing about this lease the same
    # weight as the findings. Demoting it is not removing it: it is still on every
    # report, and the corpus date it used to carry now sits in the provenance line
    # where a reader looking for it would look.
    # Provenance immediately above it, which is also what makes "since" refer to
    # something: "corpus 2026-07-25" is now the line before "the law may have changed
    # since", rather than four sections above it.
    parts.append(f'<p class="lh-prov lh-prov-foot">{_provenance(state)}</p>')
    parts.append(
        f'<p class="lh-legal">{esc(caveat())}</p>'
    )
    return _wrap(parts, markdown)


def _wrap(parts: list[str], markdown: str) -> str:
    body = "".join(parts)
    # A textarea rather than a data attribute: its value survives newlines without
    # any encoding of its own, and only `<` and `&` need escaping to keep it closed.
    hidden = f'<textarea class="lh-md">{escape(markdown)}</textarea>' if markdown else ""
    return f'<div class="lh-report">{body}{hidden}</div>'
