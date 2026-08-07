"""Does the missing-protections checklist cover what RCW 59.18 actually requires?

The checklist in `scan.PROTECTION_CHECKLIST` is five items, curated by hand, each with
a citation and an admission criterion written down beside it. Nothing checked it
against the statute. That is the worst shape a gap can have: an item that is not on
the list is never looked for, so its absence from a lease is invisible — and invisible
to every other eval in this directory too, because they score what the scanner
reports against what a manifest says is missing, and a requirement nobody wrote down
appears in neither. It is the same blind spot the permissive-clause set was built to
expose in the judge, one layer up.

So this reads the corpus instead of the checklist. Every section of the statutes in
`corpus/wa/statutes/` gets one question: does this section impose a duty that can be
satisfied ONLY by text in the rental agreement, or by a document whose delivery the
agreement must record? That is the admission criterion, and it is narrow on purpose.
RCW 59.18.060(16) is why: it requires the landlord to designate their name and address
"by a statement on the rental agreement OR by a notice conspicuously posted on the
premises", so a lease that never names the landlord may be perfectly compliant with
the notice in the stairwell. Absence from the text is not evidence there, and a
checklist item that reports it as missing would be inventing a false red.

Every candidate the sweep finds is then in `checklist_register.json` with a decision:
on the checklist, or excluded with the reason. A candidate in neither is a coverage
gap and this prints it as one. The register is the artifact a reader should look at —
the point is not the score, it is that the exclusions are written down somewhere other
than in somebody's head.

**The sweep is a candidate generator, not a measurement, and the two runs behind the
register are the evidence for saying so**: at temperature=0 they disagreed on 4 of 98
sections (16 candidates then 14, with 59.18.140, .170 and .257 dropping out and .575
appearing). Ninety-eight independent classifications of long statutory text will do
that, and it is why the register accumulates the union rather than being regenerated —
a decision, once made, does not expire because a later sweep stopped offering the
section. `register_entries_not_found` reports that gap for information, not as a
failure.

    python -m evaluation.eval_checklist_coverage
    python -m evaluation.eval_checklist_coverage --write

98 sections, one call each, ≈ $0.05. No retrieval, no clause judging.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from litellm import completion
from pydantic import BaseModel, Field

from evaluation.provenance import stamp
from leasehound.metrics import UsageMeter
from leasehound.retrieval import GENERATION_MODEL, llm_retry
from leasehound.scan import MAX_PARALLEL_SCANS, PROTECTION_CHECKLIST

EVAL_DIR = Path(__file__).parent
STATUTES_DIR = EVAL_DIR.parent / "corpus" / "wa" / "statutes"
REGISTER_PATH = EVAL_DIR / "checklist_register.json"
RESULTS_PATH = EVAL_DIR / "checklist_coverage_results.json"

DutyKind = Literal["lease_text_required", "satisfiable_elsewhere", "no_text_duty"]


class SectionDuty(BaseModel):
    # Description first: the model states what the duty is before deciding which kind
    # it is, so the classification has something concrete to be about.
    duty: str = Field(
        description="The written-disclosure duty this section places on the landlord, "
        "in one sentence, naming the subsection. Empty string if it places none."
    )
    kind: DutyKind = Field(
        description="lease_text_required: the duty can be satisfied ONLY by text in "
        "the rental agreement itself, or by a document whose delivery the agreement "
        "must record — so a lease whose text does not address it is out of compliance. "
        "satisfiable_elsewhere: there is a written or disclosure duty, but it can be "
        "met outside the lease text — by a posted notice, a separately delivered "
        "document, a form given on request, or conduct — so silence in the lease "
        "proves nothing. no_text_duty: the section imposes no written-disclosure duty "
        "on the landlord at all (it grants a remedy, defines a term, sets a procedure)."
    )


PROMPT = """You are auditing one section of Washington State's Residential
Landlord-Tenant Act (RCW 59.18) for a single question: does it require something to
be IN THE WRITTEN RENTAL AGREEMENT?

The distinction that matters is not whether the section is important, or whether it
protects tenants. It is whether a rental agreement whose text is silent about this
section is, by that silence alone, out of compliance.

Two examples of the line, both from this chapter:

- RCW 59.18.260(1) requires that where a deposit is collected, the rental agreement
  "shall be in writing and shall include the terms and conditions under which the
  deposit ... may be withheld". A lease that collects a deposit and says nothing about
  withholding conditions violates it on the face of the document.
  -> lease_text_required

- RCW 59.18.060(16) requires the landlord to designate their name and address "by a
  statement on the rental agreement or by a notice conspicuously posted on the
  premises". A lease that never names the landlord may be perfectly compliant, with
  the notice in the stairwell.
  -> satisfiable_elsewhere

Duties of conduct — repairs, weatherproofing, giving notice before entering — are
never lease_text_required: a lease's silence about behaviour says nothing about it.

Section:
{text}
"""


@llm_retry
def classify_section(text: str, meter: UsageMeter) -> SectionDuty:
    response = completion(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(text=text)}],
        response_format=SectionDuty, temperature=0,
    )
    meter.add_completion(response)
    return SectionDuty.model_validate_json(response.choices[0].message.content)


def section_name(path: Path) -> str:
    """rcw-59-18-260.md -> RCW 59.18.260"""
    parts = path.stem.split("-")
    return f"RCW {parts[1]}.{parts[2]}.{parts[3]}"


def sweep(meter: UsageMeter) -> list[dict]:
    paths = sorted(STATUTES_DIR.glob("*.md"))

    def one(path: Path) -> dict:
        answer = classify_section(path.read_text(encoding="utf-8"), meter)
        return {"section": section_name(path), "kind": answer.kind, "duty": answer.duty}

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SCANS) as pool:
        return sorted(pool.map(one, paths), key=lambda r: r["section"])


def load_register() -> dict:
    if not REGISTER_PATH.exists():
        return {"entries": []}
    return json.loads(REGISTER_PATH.read_text(encoding="utf-8"))


ON_LIST = ("in_checklist", "in_checklist_but_fails_the_criterion")


def score(rows: list[dict], register: dict) -> dict:
    candidates = [r for r in rows if r["kind"] == "lease_text_required"]
    decided = {e["section"] for e in register["entries"]}
    shipped = {item["name"] for item in PROTECTION_CHECKLIST}
    claimed = {e["checklist_item"] for e in register["entries"]
               if e["status"] in ON_LIST}
    return {
        # The two findings, at the top because they are the reason this exists. One
        # names shipped items whose statute does not meet the admission criterion
        # written beside the checklist; the other names a requirement that meets it
        # and is not checked. Both are readings, and the register gives each a why.
        "checklist_items_failing_criterion": sorted(
            e["checklist_item"] for e in register["entries"]
            if e["status"] == "in_checklist_but_fails_the_criterion"),
        "requirements_missing_from_checklist": sorted(
            e["section"] for e in register["entries"]
            if e["status"] == "missing_from_checklist"),
        "sections_swept": len(rows),
        "candidates": len(candidates),
        "candidate_sections": [r["section"] for r in candidates],
        # The number that matters. A candidate with no entry in the register is a
        # requirement the scanner may never check, that nobody has decided about.
        "undecided_candidates": sorted(r["section"] for r in candidates
                                       if r["section"] not in decided),
        # And the other direction: a register entry this sweep did not offer. Expected
        # rather than alarming — see the note about run-to-run disagreement above — and
        # reported so a reader can see which decisions rest on an earlier sweep. It
        # would also catch an exclusion outliving the section it was written about,
        # since the corpus moves (scripts/check_corpus_drift.py watches that).
        "register_entries_not_found": sorted(
            decided - {r["section"] for r in rows if r["kind"] == "lease_text_required"}),
        "checklist_items": sorted(shipped),
        "checklist_items_unclaimed": sorted(shipped - claimed),
        "by_kind": {kind: sum(1 for r in rows if r["kind"] == kind)
                    for kind in ("lease_text_required", "satisfiable_elsewhere",
                                 "no_text_duty")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help=f"write {RESULTS_PATH.name}")
    args = parser.parse_args()

    paths = sorted(STATUTES_DIR.glob("*.md"))
    print(f"{len(paths)} sections, one call each ≈ ${0.0006 * len(paths):.3f}")

    meter = UsageMeter()
    rows = sweep(meter)
    register = load_register()
    summary = score(rows, register)

    print(f"\n{summary['sections_swept']} sections swept: "
          f"{json.dumps(summary['by_kind'])}")
    print(f"{summary['candidates']} require text in the rental agreement:")
    for row in rows:
        if row["kind"] != "lease_text_required":
            continue
        entry = next((e for e in register["entries"]
                      if e["section"] == row["section"]), None)
        mark = {"in_checklist": "ON LIST ",
                "in_checklist_but_fails_the_criterion": "ON LIST*",
                "missing_from_checklist": "MISSING ",
                "excluded": "excluded"}.get((entry or {}).get("status"), "UNDECIDED")
        print(f"  [{mark}] {row['section']:16} {row['duty'][:96]}")
    if summary["checklist_items_failing_criterion"]:
        print(f"\n!! on the checklist but the statute does not meet the admission "
              f"criterion: {summary['checklist_items_failing_criterion']}")
    if summary["requirements_missing_from_checklist"]:
        print(f"!! meets the criterion and is not checked: "
              f"{summary['requirements_missing_from_checklist']}")
    if summary["undecided_candidates"]:
        print(f"\n!! undecided: {summary['undecided_candidates']} — a requirement with "
              f"no decision is one the scanner may never check")
    if summary["checklist_items_unclaimed"]:
        print(f"!! on the checklist but not traced to a swept section: "
              f"{summary['checklist_items_unclaimed']}")
    print(f"\n${meter.cost_usd:.4f} · {meter.llm_calls} LLM calls")

    if args.write:
        RESULTS_PATH.write_text(json.dumps({
            "provenance": stamp(),
            "cost_usd": round(meter.cost_usd, 5),
            "summary": summary,
            "sections": rows,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
