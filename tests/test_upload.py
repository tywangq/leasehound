"""Clause splitting: deterministic, so it gets exact tests."""

from leasehound.upload import MIN_CLAUSE_CHARS, split_clauses

FILLER = "The parties agree to the terms set forth in this provision as written. "

NUMBERED_LEASE = f"""Residential Lease Agreement between Landlord and Tenant for the premises. {FILLER}

1. TERM. The initial term of this lease is twelve months. {FILLER}

2. RENT. Monthly rent is $1,850 due on the first of the month. {FILLER}

3. LATE CHARGES. A late charge applies after the grace period. {FILLER}

4. ENTRY. Landlord shall give two days' written notice before entry. {FILLER}
"""


def test_numbered_lease_splits_on_clause_headings():
    clauses = split_clauses(NUMBERED_LEASE)
    assert len(clauses) == 5  # preamble + 4 numbered clauses
    assert clauses[1].startswith("1. TERM")
    assert clauses[4].startswith("4. ENTRY")


def test_fragments_below_minimum_are_dropped():
    text = NUMBERED_LEASE + "\n5. STUB.\n"
    clauses = split_clauses(text)
    assert all(len(c) >= MIN_CLAUSE_CHARS for c in clauses)
    assert not any(c.startswith("5. STUB") for c in clauses)


def test_unnumbered_document_falls_back_to_paragraph_merging():
    paragraphs = "\n\n".join(FILLER for _ in range(12))
    clauses = split_clauses(paragraphs)
    assert len(clauses) >= 2
    assert all(len(c) >= MIN_CLAUSE_CHARS for c in clauses)


def test_empty_document_yields_no_clauses():
    assert split_clauses("") == []


def test_tail_paragraph_is_kept_even_when_short():
    # The fallback flushes its final buffer regardless of size, so a lease's
    # last paragraph is never silently dropped.
    assert split_clauses("Signed by both parties.") == ["Signed by both parties."]
