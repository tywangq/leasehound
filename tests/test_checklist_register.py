"""The protections checklist and the register of statutes must agree (no API calls).

The checklist is five hand-curated items with an admission criterion written in a code
comment, and until `evaluation/eval_checklist_coverage.py` existed nothing checked it
against the statute. That is the worst shape a gap can have: an item that is not on the
list is never looked for, so its absence from a lease is invisible to the scanner and
also to every eval, because they score reported-missing against a manifest and a
requirement nobody wrote down appears in neither.

The register is now the decision layer over the sweep, and these tests keep it and the
shipped checklist from drifting apart in either direction — an item on the list with
nobody's reasoning behind it, or a piece of reasoning about an item that no longer
exists. The sweep itself is not tested here; it costs money and its answers move.
"""

import json
from pathlib import Path

from leasehound.scan import PROTECTION_CHECKLIST

REGISTER = json.loads(
    (Path(__file__).parent.parent / "evaluation" / "checklist_register.json")
    .read_text(encoding="utf-8"))
ENTRIES = REGISTER["entries"]
ON_LIST = ("in_checklist", "in_checklist_but_fails_the_criterion")

VALID_STATUSES = {
    "in_checklist",
    "in_checklist_but_fails_the_criterion",
    "missing_from_checklist",
    "excluded",
}


def test_every_shipped_checklist_item_has_a_recorded_statute_and_reason():
    """A checklist item with no register entry is an item nobody has justified against
    the statute — which is the state the whole five-item list was in."""
    claimed = {e["checklist_item"] for e in ENTRIES if e["status"] in ON_LIST}
    shipped = {item["name"] for item in PROTECTION_CHECKLIST}
    assert claimed == shipped, (
        f"only in evaluation/checklist_register.json: {sorted(claimed - shipped)}; "
        f"only in scan.PROTECTION_CHECKLIST: {sorted(shipped - claimed)}")


def test_a_checklist_item_and_its_register_entry_cite_the_same_statute():
    by_name = {item["name"]: item["citation"] for item in PROTECTION_CHECKLIST}
    for entry in ENTRIES:
        if entry["status"] not in ON_LIST:
            continue
        assert entry["section"] == by_name[entry["checklist_item"]], (
            f"{entry['checklist_item']} is checked against "
            f"{by_name[entry['checklist_item']]} but justified from "
            f"{entry['section']}")


def test_every_entry_carries_a_decision_and_a_reason():
    for entry in ENTRIES:
        assert entry["status"] in VALID_STATUSES, entry["status"]
        # The reason is the artifact. A status with no argument behind it is the
        # undocumented judgment call this file replaced.
        assert len(entry["why"]) > 60, f"{entry['section']}: {entry['why']!r}"
        assert entry["duty"], entry["section"]
        # Anything not simply on the list is a proposed change to shipped behaviour and
        # has to say what the change is.
        if entry["status"] != "in_checklist":
            assert entry["status"] == "excluded" or entry.get("recommendation"), entry


def test_the_criterion_in_the_register_is_the_one_in_the_code():
    """Two copies of a rule stay in step by luck. Both halves of the criterion are
    load-bearing — the second is why a co-signed move-in checklist qualifies and a
    posted mold notice does not — so both are pinned."""
    criterion = REGISTER["criterion"]
    assert "satisfiable ONLY by text in the lease" in criterion
    assert "a document the lease must record delivering" in criterion
    assert "no false reds" in criterion
