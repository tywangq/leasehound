"""Section completion: implemented, measured, and enabled nowhere.

Kept for the same reason bm25.py is kept — the experiment is the deliverable.
The scan-mode retrieval eval showed it cannot address the failure it was aimed
at: every one of the 31 partial retrieval misses on the generated set is the
same section (RCW 59.18.230), and a section that never arrives cannot be
completed. It also costs ranking, because expanding the top sections pushes
other sections' chunks down. So these tests pin the behaviour of dead-but-kept
code, and the invariant that it stays off.
"""

from unittest.mock import patch

from leasehound.retrieval import (
    PipelineConfig,
    Result,
    chunk_order,
    complete_sections,
)
from leasehound.scan import scan_config


def test_it_is_off_in_the_shipped_scan_configuration():
    assert PipelineConfig().section_completion is False
    assert scan_config("wa").section_completion is False


def test_chunk_order_reads_the_id_suffix_that_ingest_writes():
    # Metadata carries no chunk index, so the id suffix is the only thing that
    # puts a section back into reading order.
    assert sorted(["wa-117", "wa-9", "wa-114"], key=chunk_order) == ["wa-9", "wa-114", "wa-117"]
    assert chunk_order("no-digits") == 0


def fake_collection(by_section: dict[str, list[tuple[str, str]]]):
    class Collection:
        def get(self, where):
            section = where["section"]
            rows = by_section[section]
            return {
                "ids": [i for i, _ in rows],
                "documents": [d for _, d in rows],
                "metadatas": [{"section": section} for _ in rows],
            }
    return Collection()


def test_a_hit_section_arrives_whole_and_in_document_order():
    # The point of the experiment: retrieval surfaced chunk 4 of the section,
    # which covers landlord's liens, while the prohibition sits in chunk 3.
    collection = fake_collection({"RCW 59.18.230": [
        ("wa-117", "distress for rent"), ("wa-116", "prohibited clauses"),
        ("wa-115", "restrictions"), ("wa-114", "waiver of rights"),
    ]})
    retrieved = [Result(page_content="distress for rent", metadata={"section": "RCW 59.18.230"})]
    config = PipelineConfig(section_completion=True)
    with patch("leasehound.retrieval._get_collection", lambda name: collection):
        out = complete_sections(retrieved, config)
    assert [c.page_content for c in out] == [
        "waiver of rights", "restrictions", "prohibited clauses", "distress for rent"]


def test_sections_past_the_cap_keep_only_the_chunk_retrieval_chose():
    # Completing every section would multiply the judge's prompt, so only the
    # first few expand; the rest must survive untouched rather than be dropped.
    by_section = {f"RCW 59.18.{n}": [(f"wa-{n}0", f"{n} whole")] for n in (10, 20, 30)}
    retrieved = [
        Result(page_content=f"{n} hit", metadata={"section": f"RCW 59.18.{n}"})
        for n in (10, 20, 30, 40)
    ]
    config = PipelineConfig(section_completion=True)
    with patch("leasehound.retrieval._get_collection", lambda name: fake_collection(by_section)):
        out = complete_sections(retrieved, config)
    assert "40 hit" in [c.page_content for c in out]
    assert len(out) == 4
