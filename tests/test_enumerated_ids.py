"""Ids in the enumerated collection must stay readable by chunk_order (no API calls).

A latent hazard rather than a live bug, and the interesting kind: two features that
are each fine alone and broken together. Ids carry document order — `chunk_order`
parses the integer suffix so `section_completion` can reassemble a section in
reading order, and anything unparseable sorts to 0. The first version of the
enumerated builder emitted `wa-230(2)(a)`, which reads as position 0, so turning on
both off-by-default features at once would have scrambled the order silently.
"""

from leasehound.retrieval import chunk_order
from scripts.build_enumerated_collection import splice_units


def entry(index: int, section: str) -> dict:
    return {"id": f"wa-{index}", "document": f"text {index}",
            "metadata": {"section": section}, "embedding": [0.0]}


def unit(label: str) -> dict:
    return {"id": "", "document": f"unit {label}", "metadata": {"section": "RCW 59.18.230"},
            "embedding": [1.0]}


def collection() -> list[dict]:
    return [entry(0, "RCW 59.18.060"), entry(1, "RCW 59.18.060"),
            entry(2, "RCW 59.18.230"), entry(3, "RCW 59.18.230"),
            entry(4, "RCW 59.18.260")]


def test_every_id_stays_parseable_by_chunk_order():
    spliced = splice_units(collection(), "RCW 59.18.230", [unit("(2)(a)"), unit("(2)(b)"),
                                                           unit("(2)(c)")])
    orders = [chunk_order(e["id"]) for e in spliced]
    assert orders == list(range(len(spliced))), "ids must be a dense, ordered integer sequence"
    assert 0 not in orders[1:], "no id may collapse to the sentinel position"


def test_units_land_where_the_replaced_section_was():
    spliced = splice_units(collection(), "RCW 59.18.230", [unit("(2)(a)"), unit("(2)(b)"),
                                                           unit("(2)(c)")])
    sections = [e["metadata"]["section"] for e in spliced]
    # Two chunks out, three units in, and the neighbours keep their reading positions.
    assert sections == ["RCW 59.18.060", "RCW 59.18.060", "RCW 59.18.230",
                       "RCW 59.18.230", "RCW 59.18.230", "RCW 59.18.260"]
    assert [e["document"] for e in spliced][2:5] == ["unit (2)(a)", "unit (2)(b)", "unit (2)(c)"]


def test_untouched_chunks_keep_their_documents_and_embeddings():
    spliced = splice_units(collection(), "RCW 59.18.230", [unit("(2)(a)")])
    assert spliced[0]["document"] == "text 0" and spliced[0]["embedding"] == [0.0]
    assert spliced[-1]["document"] == "text 4", "the tail must not be dropped by the splice"


def test_a_scattered_section_is_refused_rather_than_guessed():
    scattered = [entry(0, "RCW 59.18.230"), entry(1, "RCW 59.18.060"),
                 entry(2, "RCW 59.18.230")]
    try:
        splice_units(scattered, "RCW 59.18.230", [unit("(2)(a)")])
    except SystemExit as stop:
        assert "not contiguous" in str(stop)
    else:
        raise AssertionError("a non-contiguous section must not be silently spliced")


def test_a_missing_section_is_refused():
    try:
        splice_units(collection(), "RCW 59.18.999", [unit("(2)(a)")])
    except SystemExit as stop:
        assert "no chunks" in str(stop)
    else:
        raise AssertionError("splicing into a section that isn't there must fail loudly")
