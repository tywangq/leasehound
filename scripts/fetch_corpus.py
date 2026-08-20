"""Fetch and normalize the LeaseHound reference corpus (Layer 1).

Sources (public domain, WA government):
  - RCW Chapter 59.18 "Residential Landlord-Tenant Act", full chapter view:
    https://app.leg.wa.gov/RCW/default.aspx?cite=59.18&full=true

Output: one markdown file per statute section under corpus/wa/statutes/,
matching the ingestion pipeline's `<corpus>/<type>/**/*.md` layout.

Usage:
    python scripts/fetch_corpus.py [--cached path/to/chapter.html]
"""

import argparse
import html
import re
import sys
import urllib.request
from pathlib import Path

CHAPTER_URL = "https://app.leg.wa.gov/RCW/default.aspx?cite=59.18&full=true"
STATUTES_DIR = Path(__file__).parent.parent / "corpus" / "wa" / "statutes"

ANCHOR_RE = re.compile(r"<a\s+name='(59\.18\.\d+)'\s*>")
TAG_RE = re.compile(r"<[^>]+>")


def download() -> str:
    req = urllib.request.Request(CHAPTER_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8")


def strip_tags(fragment: str) -> str:
    # Block-level tags become newlines so statute subsections stay separated.
    fragment = re.sub(r"</(div|p|h3|tr|table)>", "\n", fragment, flags=re.I)
    fragment = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.I)
    text = TAG_RE.sub("", fragment)
    text = html.unescape(text)
    # Collapse whitespace but keep paragraph breaks.
    lines = [re.sub(r"[ \t\xa0]+", " ", line).strip() for line in text.splitlines()]
    out, blank = [], False
    for line in lines:
        if line:
            out.append(line)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def split_sections(page: str) -> list[tuple[str, str]]:
    """Return (section_number, section_html) pairs from the full-chapter page."""
    anchors = list(ANCHOR_RE.finditer(page))
    sections = []
    for match, nxt in zip(anchors, anchors[1:] + [None]):
        end = nxt.start() if nxt else len(page)
        sections.append((match.group(1), page[match.start():end]))
    return sections


def section_title(body_text: str, number: str) -> str:
    # The first non-empty line after the section number is the caption.
    for line in body_text.splitlines():
        line = line.strip()
        if line and number not in line and line.lower() != "html" and line.lower() != "pdf":
            return line.rstrip(".")
    return "Untitled"


def render_section(number: str, fragment: str) -> tuple[Path, str]:
    """Normalize one section into its corpus filename and file content.

    Kept separate from writing so `check_corpus_drift` can render the live page
    and compare byte-for-byte against the committed files. A drift check that
    re-implemented this normalization would disagree with the fetcher over
    whitespace and cry wolf every week, which is worse than no check at all.
    """
    text = strip_tags(fragment)
    # Drop the leading anchor artifacts (HTML/PDF buttons, RCW number repeated).
    text = re.sub(rf"^(HTML|PDF)*\s*(RCW\s*)?{re.escape(number)}\s*", "", text).strip()
    title = section_title(text, number)
    path = STATUTES_DIR / f"rcw-{number.replace('.', '-')}.md"
    source_url = f"https://app.leg.wa.gov/RCW/default.aspx?cite={number}"
    content = (
        f"# RCW {number} — {title}\n\n"
        f"Source: {source_url}\n"
        f"Jurisdiction: Washington State (Residential Landlord-Tenant Act)\n\n"
        f"{text}\n"
    )
    return path, content


def write_section(number: str, fragment: str) -> Path:
    path, content = render_section(number, fragment)
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", help="Parse a previously downloaded chapter HTML file")
    args = parser.parse_args()

    page = Path(args.cached).read_text(encoding="utf-8") if args.cached else download()
    STATUTES_DIR.mkdir(parents=True, exist_ok=True)

    sections = split_sections(page)
    if not sections:
        sys.exit("No sections found — page markup may have changed")

    for number, fragment in sections:
        write_section(number, fragment)

    print(f"Wrote {len(sections)} statute sections to {STATUTES_DIR}")


if __name__ == "__main__":
    main()
