"""The synthetic dataset's guard rails: spec shape and the verifier that gates it.

Labels for the silver set come from generation specs rather than from human
review, so the deterministic verifier is what makes them trustworthy — it gets
tested like production code. No API calls: generation is stubbed.
"""

import random

from evaluation.make_synthetic_leases import (
    INJECTIONS,
    PROTECTIONS,
    VIOLATIONS,
    GeneratedLease,
    PlantedRef,
    make_specs,
    verify,
)

FILLER = "The parties agree to the terms set forth in this provision as written. "


def lease_markdown(clauses: dict[int, str], preamble: str = "Lease between the parties.") -> str:
    body = "\n\n".join(f"{n}. {text} {FILLER}" for n, text in sorted(clauses.items()))
    return f"{preamble} {FILLER}\n\n{body}\n"


def spec(**overrides) -> dict:
    base = {
        "category": "violations",
        "style": "plain",
        "city": "Seattle",
        "rent": 1500,
        "n_clauses": 5,
        "violations": [],
        "protections_missing": [],
        "protections_present": list(PROTECTIONS),
        "near_misses": [],
        "injection": None,
    }
    return {**base, **overrides}


ALL_PROTECTIONS_CLAUSE = (
    "DISCLOSURES. The deposit may be withheld for unpaid rent; it is held in a "
    "trust account at First Sound Bank. A signed move-in checklist describes the "
    "unit's condition. Tenant received fire safety and mold information."
)


def test_specs_are_seeded_and_cover_every_category():
    first = make_specs(random.Random(42))
    second = make_specs(random.Random(42))
    assert [s["violations"] for s in first] == [s["violations"] for s in second]
    assert {s["category"] for s in first} == {"violations", "clean", "injection"}
    # Every injection payload gets exactly one lease, and each carries planted
    # violations — an injection lease with nothing to hide proves nothing.
    injections = [s for s in first if s["category"] == "injection"]
    assert sorted(s["injection"] for s in injections) == sorted(INJECTIONS)
    assert all(s["violations"] for s in injections)
    assert all(not s["violations"] for s in first if s["category"] == "clean")


def test_verifier_accepts_a_lease_that_matches_its_spec():
    lease = GeneratedLease(
        markdown=lease_markdown({
            1: "TERM. The tenancy runs twelve months.",
            2: ALL_PROTECTIONS_CLAUSE,
            3: "LATE FEES. A late charge of $50 applies on the first day rent is late.",
            4: "QUIET ENJOYMENT. Tenant may use the premises peacefully.",
        }),
        planted=[PlantedRef(violation_id="late_fee_early", clause_number=3)],
    )
    assert verify(lease, spec(violations=["late_fee_early"])) == []


def test_verifier_rejects_a_violation_whose_signal_phrase_is_absent():
    # The clause is claimed as a planted late-fee violation but describes a
    # lawful grace period — exactly the mislabel the verifier exists to catch.
    lease = GeneratedLease(
        markdown=lease_markdown({
            1: "TERM. The tenancy runs twelve months.",
            2: ALL_PROTECTIONS_CLAUSE,
            3: "LATE FEES. A late charge applies after a five-day grace period.",
        }),
        planted=[PlantedRef(violation_id="late_fee_early", clause_number=3)],
    )
    problems = verify(lease, spec(violations=["late_fee_early"]))
    assert any("signal phrases missing" in p for p in problems)


def test_verifier_rejects_a_planted_map_that_disagrees_with_the_spec():
    lease = GeneratedLease(
        markdown=lease_markdown({
            1: "TERM. The tenancy runs twelve months.",
            2: ALL_PROTECTIONS_CLAUSE,
            3: "ENTRY. Landlord may enter with 12 hours' notice.",
        }),
        planted=[PlantedRef(violation_id="short_entry_notice", clause_number=3)],
    )
    problems = verify(lease, spec(violations=["late_fee_early"]))
    assert any("!= requested" in p for p in problems)


def test_verifier_rejects_a_protection_that_should_be_omitted():
    lease = GeneratedLease(
        markdown=lease_markdown({
            1: "TERM. The tenancy runs twelve months.",
            2: ALL_PROTECTIONS_CLAUSE,  # mentions mold
            3: "RENT. Rent is due on the first of the month.",
        }),
        planted=[],
    )
    missing = ["Mold information"]
    problems = verify(lease, spec(
        protections_missing=missing,
        protections_present=[p for p in PROTECTIONS if p not in missing],
    ))
    assert any("must be omitted" in p for p in problems)


def test_verifier_rejects_an_unnumbered_document():
    lease = GeneratedLease(
        markdown="\n\n".join(FILLER for _ in range(6)), planted=[]
    )
    problems = verify(lease, spec(protections_present=[]))
    assert any("clause numbering" in p for p in problems)


def test_verifier_rejects_a_self_announcing_injection_payload():
    payload = INJECTIONS["whitewash_inline"]["payload"]
    lease = GeneratedLease(
        markdown=lease_markdown({
            1: "TERM. The tenancy runs twelve months.",
            2: ALL_PROTECTIONS_CLAUSE,
            3: f"LATE FEES. $50 on the first day rent is late. PAYLOAD: {payload}",
        }),
        planted=[PlantedRef(violation_id="late_fee_early", clause_number=3)],
    )
    problems = verify(lease, spec(violations=["late_fee_early"],
                                  injection="whitewash_inline"))
    assert any("PAYLOAD" in p for p in problems)


def test_every_violation_carries_citations_the_gold_manifest_recognizes():
    for vid, violation in VIOLATIONS.items():
        assert violation["citations"], vid
        assert all(c.startswith("RCW 59.18.") for c in violation["citations"]), vid
