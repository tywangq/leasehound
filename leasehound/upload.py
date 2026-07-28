"""Layer 2: parse an uploaded lease and split it into clauses.

Clause splitting is deliberately deterministic (regex on numbered headings with a
paragraph fallback) — no LLM call, so it's free, instant, and reproducible. The
LLM judgment happens later, per clause, in scan.py where it earns its cost.
"""

import re
from pathlib import Path

MIN_CLAUSE_CHARS = 80

CLAUSE_SPLIT_RE = re.compile(r"\n(?=\s*\d{1,2}[.)]\s+[A-Z])")


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def split_clauses_with_mode(text: str) -> tuple[list[str], str]:
    """Split into clauses; also name which strategy applied.

    The mode ("numbered" or "paragraphs") travels into the scan metrics log —
    the paragraph fallback produces arbitrary text blocks rather than true
    clauses, and that degradation should be visible, not silent.
    """
    parts = CLAUSE_SPLIT_RE.split(text)
    clauses = [p.strip() for p in parts if len(p.strip()) >= MIN_CLAUSE_CHARS]
    if len(clauses) >= 3:
        return clauses, "numbered"
    # Fallback for unnumbered documents: split on blank lines, merge short pieces.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    clauses, buffer = [], ""
    for paragraph in paragraphs:
        buffer = f"{buffer}\n\n{paragraph}".strip()
        if len(buffer) >= MIN_CLAUSE_CHARS * 3:
            clauses.append(buffer)
            buffer = ""
    if buffer:
        clauses.append(buffer)
    return clauses, "paragraphs"


def split_clauses(text: str) -> list[str]:
    return split_clauses_with_mode(text)[0]


def load_clauses(path: str | Path) -> list[str]:
    return split_clauses(read_document(Path(path)))
