"""Detect whether the live RCW 59.18 text has drifted from the committed corpus.

A retrieval system is only as correct as its snapshot of the law. LeaseHound
cites statute text from `corpus/wa/statutes/`, fetched once and committed — so
every verdict silently inherits the date of that fetch. Washington amends
RCW 59.18 most sessions (.230 was amended in 2025), and nothing in the repo
would notice.

This script re-fetches the chapter, renders it through the *same* normalization
`fetch_corpus.py` uses, and compares byte-for-byte against what is committed.
It makes zero API calls, so it costs nothing to run on a schedule — the price
of the fetch is one HTTP request, and the expensive work (re-embedding, the
gold-set eval) only happens when a human decides the change is real.

Deliberately an alarm, not an autopilot: the corpus is legal ground truth, and
a pipeline that quietly re-embedded amended statute text and reopened the
scanner for business without anyone reading the amendment would be the wrong
design. On drift, the workflow files an issue with the diff and the checklist.

Exit codes are the interface:
    0  corpus matches the live chapter
    1  drift — the live text differs from what is committed
    2  fetch or parse failure (the scraper broke, which is NOT drift)

That last distinction is the point of the section-count guard below. If
leg.wa.gov reorganizes its markup, `split_sections` returns nothing and a naive
check would report all 98 sections as deleted — a scraper bug dressed up as a
legislative earthquake. Those need different humans and different fixes.

Usage:
    python -m scripts.check_corpus_drift
    python -m scripts.check_corpus_drift --cached chapter.html --out report.md
"""

import argparse
import difflib
import sys
from dataclasses import dataclass, field
from pathlib import Path

from scripts.fetch_corpus import STATUTES_DIR, download, render_section, split_sections

# Below this share of the committed sections, assume the parser broke rather
# than that the legislature deleted the chapter.
MIN_PARSE_RATIO = 0.9
# Per-section diff lines kept in the report, so a big session can't blow past
# GitHub's issue-body limit.
MAX_DIFF_LINES = 24


@dataclass
class CorpusDiff:
    changed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    diffs: dict[str, str] = field(default_factory=dict)

    @property
    def drifted(self) -> bool:
        return bool(self.changed or self.added or self.removed)


def diff_corpus(live: dict[str, str], committed: dict[str, str]) -> CorpusDiff:
    """Compare rendered section content by filename. Pure — no I/O."""
    result = CorpusDiff(
        added=sorted(set(live) - set(committed)),
        removed=sorted(set(committed) - set(live)),
    )
    for name in sorted(set(live) & set(committed)):
        if live[name] == committed[name]:
            continue
        result.changed.append(name)
        lines = list(
            difflib.unified_diff(
                committed[name].splitlines(),
                live[name].splitlines(),
                fromfile=f"committed/{name}",
                tofile=f"live/{name}",
                lineterm="",
                n=1,
            )
        )
        if len(lines) > MAX_DIFF_LINES:
            lines = lines[:MAX_DIFF_LINES] + [f"… {len(lines) - MAX_DIFF_LINES} more diff lines"]
        result.diffs[name] = "\n".join(lines)
    return result


def read_committed(statutes_dir: Path) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(statutes_dir.glob("rcw-*.md"))}


def render_live(page: str) -> dict[str, str]:
    sections = split_sections(page)
    return {path.name: content for path, content in (render_section(n, f) for n, f in sections)}


def report(diff: CorpusDiff) -> str:
    """A markdown body for the drift issue — diff first, then what a human owes."""
    lines = ["## RCW 59.18 has changed since the committed snapshot", ""]
    for label, names in (
        ("Changed", diff.changed),
        ("New sections", diff.added),
        ("Sections no longer in the chapter", diff.removed),
    ):
        if names:
            lines.append(f"**{label} ({len(names)}):** {', '.join(names)}")
            lines.append("")
    for name in diff.changed:
        lines += [f"<details><summary><code>{name}</code></summary>", "",
                  "```diff", diff.diffs[name], "```", "", "</details>", ""]
    lines += [
        "## Before this corpus ships",
        "",
        "The scanner's verdicts are only as current as this snapshot, so the",
        "amendment has to be read by a person before it becomes ground truth.",
        "",
        "- [ ] Read the amendment — a changed prohibition may need a new or reworded",
        "      required-protections checklist item in `leasehound/scan.py`",
        "- [ ] `python scripts/fetch_corpus.py` to update `corpus/wa/statutes/`",
        "- [ ] Bump `CORPUS_SNAPSHOT` in `leasehound/scan.py` (it is stamped on every report)",
        "- [ ] `python -m leasehound.ingest` to re-embed",
        "- [ ] `python -m evaluation.eval_scan` — the gold set is the regression gate",
        "- [ ] Update the gold manifest if the amendment changed what counts as a violation",
        "",
        "_Filed by `scripts/check_corpus_drift.py`. No API calls were made._",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", help="Compare against a previously downloaded chapter HTML")
    parser.add_argument("--out", help="Write the markdown drift report here (only if drifted)")
    args = parser.parse_args()

    committed = read_committed(STATUTES_DIR)
    if not committed:
        print(f"No committed corpus at {STATUTES_DIR} — run scripts/fetch_corpus.py first")
        return 2

    try:
        page = Path(args.cached).read_text(encoding="utf-8") if args.cached else download()
        live = render_live(page)
    except Exception as err:  # network, decoding, markup — all "check broke", not "law changed"
        print(f"Could not fetch or parse the chapter: {err!r}")
        return 2

    if len(live) < MIN_PARSE_RATIO * len(committed):
        print(
            f"Parsed only {len(live)} of {len(committed)} expected sections — treating this as a "
            "scraper failure, not drift. The chapter page markup has probably changed; "
            "fix split_sections in scripts/fetch_corpus.py."
        )
        return 2

    diff = diff_corpus(live, committed)
    if not diff.drifted:
        print(f"No drift: {len(live)} sections match the committed corpus.")
        return 0

    print(
        f"DRIFT: {len(diff.changed)} changed, {len(diff.added)} added, "
        f"{len(diff.removed)} removed (of {len(committed)} committed sections)"
    )
    for name in diff.changed:
        print(f"  changed  {name}")
    for name in diff.added:
        print(f"  added    {name}")
    for name in diff.removed:
        print(f"  removed  {name}")
    body = report(diff)
    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"\nReport written to {args.out}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
