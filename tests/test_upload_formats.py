"""Clause splitting against the numbering conventions real leases actually use.

This file exists because of a measured failure. The splitter was written against
the labeled corpus, where every lease numbers clauses "1. RENT" — and a survey of
seven real-world conventions found that six of them fell through to the paragraph
fallback, which then returned the *entire document as one clause* whenever the
source had no blank lines (PDF text extraction routinely emits single newlines).

What that costs is granularity, not text: the judge reads a clause in full, but a
one-clause document draws one verdict for the whole lease, retrieved against a
query truncated to `clause[:MAX_CLAUSE_CHARS]` — so the statutes fetched covered
the opening while the prompt forbids flagging anything the extracts don't address.
A lease with seven violations came back with one finding, and the report looked
finished.

The generated dataset could never have caught this — the generator writes the one
convention the splitter already handled. So the invariants live here instead:
every convention must split, and no clause may exceed the retrieval window.
"""

import pytest

from leasehound.upload import (
    MAX_CLAUSE_CHARS,
    cap_clause_length,
    split_clauses_with_mode,
)

BODIES = [
    "Tenant shall pay $2,000 monthly on the first day of each month without demand.",
    "The tenancy begins June 1 and continues month to month until terminated per statute.",
    "Landlord holds a deposit of $2,000 in a trust account at the bank named below.",
    "Tenant pays electricity and internet; Landlord pays water, sewer, and garbage.",
]

CONVENTIONS = {
    "plain": ["1. RENT.", "2. TERM.", "3. DEPOSIT.", "4. UTILITIES."],
    "paren": ["1) Rent.", "2) Term.", "3) Deposit.", "4) Utilities."],
    "decimal": ["1.1 Rent.", "1.2 Term.", "1.3 Deposit.", "2.1 Utilities."],
    "deep_decimal": ["1.10.1 Rent.", "1.10.2 Term.", "2.1.1 Deposit.", "2.1.2 Utilities."],
    "article_roman": ["ARTICLE I - RENT.", "ARTICLE II - TERM.",
                      "ARTICLE III - DEPOSIT.", "ARTICLE IV - UTILITIES."],
    "section_colon": ["Section 1: Rent.", "Section 2: Term.",
                      "Section 3: Deposit.", "Section 4: Utilities."],
    "three_digit": ["101. RENT.", "102. TERM.", "103. DEPOSIT.", "104. UTILITIES."],
}


def lease(headings: list[str], separator: str = "\n") -> str:
    return separator.join(f"{h} {b}" for h, b in zip(headings, BODIES))


@pytest.mark.parametrize("convention", sorted(CONVENTIONS))
def test_every_real_world_numbering_convention_splits_into_its_clauses(convention):
    clauses, mode = split_clauses_with_mode(lease(CONVENTIONS[convention]))
    assert mode == "numbered", f"{convention} fell through to the {mode} fallback"
    assert len(clauses) == 4


@pytest.mark.parametrize("convention", sorted(CONVENTIONS))
def test_conventions_split_the_same_way_with_blank_lines_between_clauses(convention):
    # Some extractors keep paragraph breaks, some don't; neither may change the result.
    assert len(split_clauses_with_mode(lease(CONVENTIONS[convention], "\n\n"))[0]) == 4


def test_the_labeled_corpus_convention_is_pinned():
    # The gold and generated sets all use this shape. Widening the pattern must
    # never re-split them, or every published eval number silently changes.
    clauses, mode = split_clauses_with_mode(lease(CONVENTIONS["plain"]))
    assert mode == "numbered"
    assert clauses[0].startswith("1. RENT.")
    assert clauses[3].startswith("4. UTILITIES.")


def test_unnumbered_text_without_blank_lines_does_not_become_one_clause():
    # The original bug: no numbering and no blank lines returned the whole
    # document as a single clause, which drew a single verdict for the lease.
    prose = "\n".join(
        "This paragraph of the rental agreement states an obligation of the parties. " * 3
        for _ in range(20)
    )
    clauses, mode = split_clauses_with_mode(prose)
    assert mode == "lines"
    assert all(len(c) <= MAX_CLAUSE_CHARS for c in clauses)
    # The pathology was one clause holding the whole document. The property that
    # matters is that no single clause does, so each gets judged on its own.
    assert max(len(c) for c in clauses) < len(prose) / 5
    assert sum(len(c) for c in clauses) > 0.9 * len(prose.replace("\n", ""))


def test_blank_line_separated_prose_still_reports_paragraph_mode():
    prose = "\n\n".join("An obligation of the parties is described here at length. " * 3
                        for _ in range(10))
    assert split_clauses_with_mode(prose)[1] == "paragraphs"


@pytest.mark.parametrize("convention", sorted(CONVENTIONS))
def test_no_clause_ever_exceeds_the_retrieval_window(convention):
    # The invariant that makes query truncation unreachable rather than unlikely:
    # every clause is short enough to be its own complete retrieval query.
    headings = CONVENTIONS[convention]
    long_bodies = [b + " The parties further agree to the terms stated herein. " * 40
                   for b in BODIES]
    text = "\n".join(f"{h} {b}" for h, b in zip(headings, long_bodies))
    clauses, _ = split_clauses_with_mode(text)
    assert clauses
    assert all(len(c) <= MAX_CLAUSE_CHARS for c in clauses)


def test_oversized_clauses_break_on_sentence_boundaries():
    sentences = ["Sentence number %d is complete and ends with a period." % i for i in range(60)]
    pieces = cap_clause_length([" ".join(sentences)])
    assert len(pieces) > 1
    assert all(len(p) <= MAX_CLAUSE_CHARS for p in pieces)
    # Boundaries land between sentences, not mid-word.
    assert all(p.endswith(".") for p in pieces)


def test_a_single_runaway_sentence_is_still_bounded():
    # Rent tables and run-on legalese have no sentence boundary to break on;
    # a hard cut is the last resort, but the window must still hold.
    pieces = cap_clause_length(["word " * 1000])
    assert all(len(p) <= MAX_CLAUSE_CHARS for p in pieces)


@pytest.mark.parametrize("wrapped", [
    "Notice must be given to the Landlord at the address stated above\n5 days before entry.",
    "Tenant shall pay the amounts described in Section\n1.1 of this Agreement each month.",
    "The total due at signing is\n2,400 dollars, payable by certified check only.",
])
def test_a_wrapped_line_is_not_mistaken_for_a_clause_heading(wrapped):
    # Every alternative in the pattern demands explicit punctuation and an
    # uppercase word, precisely so cross-references and wrapped numbers survive.
    padding = " The parties agree to the terms stated in this agreement. " * 4
    clauses, mode = split_clauses_with_mode(wrapped + padding)
    assert mode != "numbered"
    assert len(clauses) == 1
