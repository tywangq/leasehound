"""Layer 2: parse an uploaded lease and split it into clauses.

Clause splitting is deliberately deterministic (regex on numbered headings with a
paragraph fallback) — no LLM call, so it's free, instant, and reproducible. The
LLM judgment happens later, per clause, in scan.py where it earns its cost.
"""

import re
from pathlib import Path

MIN_CLAUSE_CHARS = 80

# Real leases number clauses in more ways than one. The original pattern matched
# only "1. RENT" / "1) RENT", which silently sent every other convention to the
# paragraph fallback — see tests/test_upload_formats.py for the survey that found
# this. Each alternative demands explicit punctuation so that an ordinary wrapped
# line ("...within\n5 days of notice") cannot masquerade as a clause heading.
CLAUSE_SPLIT_RE = re.compile(
    r"\n(?=\s*(?:"
    r"\d{1,3}\.\d{1,2}(?:\.\d{1,2})*\.?\s+[A-Z]"           # 1.1 Rent / 4.10.2 Rent
    r"|\d{1,3}[.)]\s+[A-Z]"                                 # 1. RENT / 12) Rent / 101. Rent
    r"|(?:ARTICLE|Article|SECTION|Section)\s+(?:\d{1,3}|[IVXL]{1,7})\b"  # ARTICLE III / Section 4
    r"))"
)

# A clause longer than this is not a clause — it is an unsplit document. scan.py
# only shows the judge the first 1200 characters, so without this the scanner
# would quietly grade 7% of a lease and report it as a finished scan.
MAX_CLAUSE_CHARS = 1200


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


SENTENCE_END_RE = re.compile(r"(?<=[.;:])\s+")


def cap_clause_length(clauses: list[str]) -> list[str]:
    """Break any clause longer than the judge's window, on sentence boundaries.

    The point is to convert an invisible loss into visible work. scan.py judges
    `clause[:MAX_CLAUSE_CHARS]`, so an unsplit document used to be graded on its
    opening 1200 characters and reported as a completed scan. Now it becomes
    more clauses instead — and if that pushes it past the 60-clause cap, the
    scan refuses with an explanation, which is the honest outcome.
    """
    out: list[str] = []
    for clause in clauses:
        if len(clause) <= MAX_CLAUSE_CHARS:
            out.append(clause)
            continue
        buffer = ""
        for sentence in SENTENCE_END_RE.split(clause):
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if buffer and len(candidate) > MAX_CLAUSE_CHARS:
                out.append(buffer)
                buffer = sentence
            else:
                buffer = candidate
        # One sentence can still overrun the window (rent tables, run-on legalese).
        while len(buffer) > MAX_CLAUSE_CHARS:
            out.append(buffer[:MAX_CLAUSE_CHARS].strip())
            buffer = buffer[MAX_CLAUSE_CHARS:]
        if buffer.strip():
            out.append(buffer.strip())
    return out


def split_clauses_with_mode(text: str) -> tuple[list[str], str]:
    """Split into clauses; also name which strategy applied.

    The mode ("numbered", "paragraphs", or "lines") travels into the scan metrics
    log — the fallbacks produce arbitrary text blocks rather than true clauses,
    and that degradation should be visible, not silent. "lines" is the weakest:
    it means the document had neither clause numbering nor blank lines to work
    from, which is what PDF text extraction often leaves behind.
    """
    parts = CLAUSE_SPLIT_RE.split(text)
    clauses = [p.strip() for p in parts if len(p.strip()) >= MIN_CLAUSE_CHARS]
    if len(clauses) >= 3:
        return cap_clause_length(clauses), "numbered"

    # Fallback for unnumbered documents. Blank lines are the better boundary, but
    # PDF extraction frequently emits single newlines only, and blank-line
    # splitting then returns the entire document as one "paragraph".
    mode = "paragraphs"
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) < 3:
        mode = "lines"
        paragraphs = [line.strip() for line in text.splitlines() if line.strip()]

    clauses, buffer = [], ""
    for paragraph in paragraphs:
        buffer = f"{buffer}\n\n{paragraph}".strip()
        if len(buffer) >= MIN_CLAUSE_CHARS * 3:
            clauses.append(buffer)
            buffer = ""
    if buffer:
        clauses.append(buffer)
    return cap_clause_length(clauses), mode


def split_clauses(text: str) -> list[str]:
    return split_clauses_with_mode(text)[0]


def load_clauses(path: str | Path) -> list[str]:
    return split_clauses(read_document(Path(path)))
