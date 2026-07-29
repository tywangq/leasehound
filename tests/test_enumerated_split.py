"""Deterministic splitting of enumerated statute catalogs (no API calls).

RCW 59.18.230(2) lists ten prohibited lease provisions. ingest.py's LLM chunker
is *instructed* to give each enumerated item its own chunk and returned four
length-shaped chunks instead, which is why .230 was the only section that never
reached the judge in scan mode. This parses the statute's own subsection markers,
so it cannot quietly stop complying.

The experiment it enabled is written up in evaluation/enumerated_split_results.json:
strict retrieval .492 -> .984, and one false red on the gold set, so it is measured
and not shipped. These tests cover the parser, which is what a future CA corpus or
a different section would reuse.
"""

from pathlib import Path

from leasehound.ingest import split_enumerated_catalog

CORPUS = Path(__file__).parent.parent / "corpus" / "wa" / "statutes"

CATALOG = """(1) Some preamble that is not a list at all.
(2) No rental agreement may provide that the tenant:
(a) Agrees to waive rights; or
(b) Authorizes confession of judgment; or
(c) Agrees to pay the landlord's fees; or
(d) Agrees to arbitrate disputes.
(3) A provision prohibited by subsection (2) is unenforceable.
"""


def test_each_enumerated_item_becomes_one_unit():
    units = split_enumerated_catalog(CATALOG)
    assert [u.label for u in units] == ["(2)(a)", "(2)(b)", "(2)(c)", "(2)(d)"]


def test_a_unit_carries_the_stem_that_gives_it_meaning():
    # "Agrees to pay the landlord's fees" prohibits nothing by itself — the statute
    # writes one sentence distributed over a list, so the stem has to travel along
    # or the chunk reads as a fragment and retrieves for the wrong reason.
    unit = next(u for u in split_enumerated_catalog(CATALOG) if u.label == "(2)(c)")
    assert unit.text.startswith("No rental agreement may provide that the tenant:")
    assert "landlord's fees" in unit.text


def test_the_list_stops_at_the_next_top_level_subsection():
    # (3) is prose about the list, not a member of it.
    assert all("unenforceable" not in u.text for u in split_enumerated_catalog(CATALOG))


def test_a_short_either_or_pair_is_left_alone():
    # Splitting a two-item alternative produces fragments, not retrievable rules.
    assert split_enumerated_catalog(
        "(4) The landlord shall either:\n(a) Repair the defect; or\n(b) Refund the rent.\n"
    ) == []


def test_prose_with_no_catalog_yields_nothing():
    assert split_enumerated_catalog("(1) A plain subsection of ordinary statutory prose.") == []


def test_the_real_section_yields_its_ten_prohibitions():
    text = (CORPUS / "rcw-59-18-230.md").read_text(encoding="utf-8")
    units = split_enumerated_catalog(text)
    assert [u.label for u in units] == [f"(2)({c})" for c in "abcdefghij"]
    # The two that the labelled leases actually plant violations of.
    exculpation = next(u for u in units if u.label == "(2)(f)")
    assert "exculpation" in exculpation.item
    electronic = next(u for u in units if u.label == "(2)(j)")
    assert "electronic means only" in electronic.item


def test_every_unit_fits_the_retrieval_window_comfortably():
    text = (CORPUS / "rcw-59-18-230.md").read_text(encoding="utf-8")
    # The whole point is granularity: a unit the size of the original chunk would
    # smear the same way the four length-shaped chunks did.
    assert all(len(u.text) < 700 for u in split_enumerated_catalog(text))
