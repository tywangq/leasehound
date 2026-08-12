"""Clause splitting: deterministic, so it gets exact tests."""

import pytest

from leasehound.upload import MIN_CLAUSE_CHARS, split_clauses, split_clauses_with_mode

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


def test_split_mode_names_the_strategy_that_applied():
    # The paragraph fallback yields arbitrary blocks, not true clauses — the
    # mode goes into the metrics log so that degradation is visible.
    assert split_clauses_with_mode(NUMBERED_LEASE)[1] == "numbered"
    unnumbered = "\n\n".join(FILLER for _ in range(12))
    assert split_clauses_with_mode(unnumbered)[1] == "paragraphs"


def test_pdf_dependency_is_installed():
    # read_document imports pypdf lazily, so a missing install only explodes on
    # the first real PDF upload — which is exactly how it shipped broken once.
    import pypdf  # noqa: F401


def test_a_locked_pdf_says_so_instead_of_crashing(tmp_path, monkeypatch):
    """An AES-encrypted PDF used to reach the generic error handler.

    pypdf raises before this app sees the file — DependencyError without a crypto backend,
    NOT_DECRYPTED with one — and the visitor got "the hound tripped over an error", which
    names nothing they can act on. Found in Cloud Run's logs after an application form was
    uploaded to the demo; `cryptography` is a dependency now, so the owner-password kind
    (the "no printing" kind that most forms carry) reads normally, and only a file with a
    real user password gets refused.
    """
    from pypdf import PdfWriter

    from leasehound.upload import EncryptedDocument, read_document

    owner = tmp_path / "owner.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("", owner_password="owner", algorithm="AES-256")
    writer.write(str(owner))
    assert read_document(owner) == "", "an owner-password PDF must open"

    locked = tmp_path / "locked.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("secret", algorithm="AES-256")
    writer.write(str(locked))
    with pytest.raises(EncryptedDocument):
        read_document(locked)
