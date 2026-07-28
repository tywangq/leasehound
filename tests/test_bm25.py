"""BM25 scoring and tokenization: deterministic, so it gets exact tests (no API calls)."""

import leasehound.bm25 as bm25_module
from leasehound.bm25 import Bm25Index, get_index, tokenize

DOCS = [
    "RCW 59.18.230 Prohibition of waiver of rights. A rental agreement may not "
    "require the tenant to waive rights or indemnify the landlord.",
    "RCW 59.18.150 Landlord's right of entry. The landlord shall give at least two "
    "days' written notice of intent to enter the dwelling unit.",
    "RCW 59.18.060 Landlord duties. The landlord shall maintain the premises and "
    "provide written fire safety information to the tenant.",
]
METAS = [{"section": s} for s in ("RCW 59.18.230", "RCW 59.18.150", "RCW 59.18.060")]


def sections(hits):
    return [meta["section"] for _, meta in hits]


def test_statute_citations_survive_as_one_token():
    # A whitespace or \w+ tokenizer shreds a citation into three numbers, which
    # makes "59.18.230" unsearchable as a unit.
    assert "59.18.230" in tokenize("See RCW 59.18.230 for details.")
    # A sentence-ending period is not part of the token.
    assert tokenize("Rent is late.") == ["rent", "is", "late"]
    # Single characters carry no signal here and would only add noise.
    assert tokenize("a b rent") == ["rent"]


def test_search_ranks_the_section_sharing_query_terms_first():
    index = Bm25Index(DOCS, METAS)
    assert sections(index.search("waive rights indemnify", 3))[0] == "RCW 59.18.230"
    assert sections(index.search("notice of intent to enter", 3))[0] == "RCW 59.18.150"
    assert sections(index.search("fire safety information", 3))[0] == "RCW 59.18.060"


def test_a_citation_can_be_searched_directly():
    index = Bm25Index(DOCS, METAS)
    assert sections(index.search("RCW 59.18.150", 1)) == ["RCW 59.18.150"]


def test_chunks_sharing_no_term_are_omitted_rather_than_padded():
    # Returning arbitrary low-scoring chunks to fill k would hand the judge
    # statutes with nothing to do with the clause.
    index = Bm25Index(DOCS, METAS)
    assert index.search("bicycle parking garage", 3) == []


def test_rarer_terms_outrank_common_ones():
    index = Bm25Index(DOCS, METAS)
    # "landlord" appears in all three documents, "indemnify" in one — a query
    # carrying both must be decided by the rare term.
    assert sections(index.search("landlord indemnify", 1)) == ["RCW 59.18.230"]


def test_length_normalization_does_not_let_a_long_document_win_on_bulk():
    padded = DOCS[1] + " " + ("The parties agree to the foregoing provision. " * 60)
    index = Bm25Index([DOCS[0], padded], [METAS[0], METAS[1]])
    # The long document still contains the entry language; the short one is the
    # only place the query's distinctive terms live.
    assert sections(index.search("waiver of rights indemnify", 1)) == ["RCW 59.18.230"]


def test_index_is_built_once_per_collection():
    calls = []

    def load():
        calls.append(1)
        return DOCS, METAS

    bm25_module._indexes.clear()
    first = get_index("wa_reference", load)
    second = get_index("wa_reference", load)
    assert first is second
    assert len(calls) == 1
    bm25_module._indexes.clear()


def test_empty_corpus_does_not_divide_by_zero():
    index = Bm25Index([], [])
    assert index.search("anything", 5) == []
