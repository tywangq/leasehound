"""Layer 2: parse an uploaded lease and split it into clauses.

Clause splitting is deliberately deterministic (regex on numbered headings with a
paragraph fallback) — no LLM call, so it's free, instant, and reproducible. The
LLM judgment happens later, per clause, in scan.py where it earns its cost.
"""

import re
from pathlib import Path

# The byte bound on an upload, and the outermost of the three size limits here.
# Lives in this module because both surfaces that accept a file need it and neither
# may import the other: app.py mounts api.py, so a constant owned by either one and
# read by the other is a circular import waiting to happen.
#
# Inside it, scan.MAX_DOCUMENT_CHARS bounds the extracted TEXT, which is the bound
# that governs spend. This one bounds the bytes, and exists so a file too big to be
# worth parsing is never parsed: on Cloud Run the filesystem is memory, so an upload
# is resident before extraction has a chance to reject it.
#
# 8 MB is ~16× the largest real lease PDF in evaluation/leases_real (491 KB) and
# leaves room for an image-heavy scan of a lease, which extracts to little or no text
# and should get the "no text layer" answer rather than a size error.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

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

# A clause longer than this is not a clause — it is an unsplit document, and one
# verdict for a whole lease is worthless. Matching scan.py's retrieval window
# (`fetch_unranked(clause[:1200])`) also makes that truncation unreachable: every
# clause becomes its own complete query instead of the document's first 1200
# characters standing in for all of it.
MAX_CLAUSE_CHARS = 1200


class EncryptedDocument(Exception):
    """A PDF locked with a user password: pypdf cannot read a page of it.

    Separate from NoTextExtracted, which is the scanned-photo case — there the file
    opens and has no text layer, here it does not open. The two need different advice
    ("try a text-based PDF" against "export an unprotected copy"), and telling someone
    to re-export a file that has no text layer sends them in a circle.

    Owner-password PDFs — the "no printing, no copying" kind that most application forms
    carry — are NOT this: they open with an empty password once a crypto backend is
    installed, which is what `cryptography` is in the dependencies for. Without it pypdf
    raises DependencyError on those too, and every such upload came back as the generic
    "the hound tripped over an error" (found in Cloud Run's logs, not reproduced here:
    the deployed image had no cryptography and neither did this venv).
    """


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        from pypdf.errors import DependencyError, PdfReadError

        try:
            reader = PdfReader(str(path))
            # An owner-password PDF ("no printing, no copying") opens with an empty
            # password and is the common case — application forms and anything produced
            # by a document portal. A user-password one returns NOT_DECRYPTED, and there
            # is nothing to do but say so.
            if reader.is_encrypted and not reader.decrypt(""):
                raise EncryptedDocument(path.name)
        except DependencyError as missing:
            # The crypto backend is a dependency now, so this is a broken install rather
            # than a locked file — but it used to surface as the generic error handler,
            # so it says something true either way.
            raise EncryptedDocument(path.name) from missing
        except PdfReadError as unreadable:
            raise EncryptedDocument(path.name) from unreadable
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


SENTENCE_END_RE = re.compile(r"(?<=[.;:])\s+")


def cap_clause_length(clauses: list[str]) -> list[str]:
    """Break any clause longer than the retrieval window, on sentence boundaries.

    The point is to convert an invisible loss into visible work. The judge does
    see a clause in full — what collapses is *granularity*. An unsplit document
    arrived as one clause, so it drew one verdict for the whole lease, retrieved
    against a query that was only its first 1200 characters: the statutes fetched
    described the opening (parties, premises, rent) while the prompt forbids
    flagging anything the extracts don't cover. A lease with seven violations
    came back with one finding, and the report looked complete.

    Now such a document becomes more clauses instead — each with its own
    retrieval and its own verdict — and if that pushes it past the 60-clause cap
    the scan judges the first 60 and names the rest as unjudged, so a report that
    covers a prefix cannot read like one that covers everything.
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
    """The clauses only, for callers with nothing to say about how they were found."""
    return split_clauses_with_mode(text)[0]
