"""Zero-shot baseline: the same labeled leases, the same model, no pipeline.

The README argues that pasting a lease into a chat window is the wrong tool
("the law shouldn't come from parametric memory"). This script measures that
claim instead of asserting it: each hand-labeled lease goes to the model
whole, zero-shot — no retrieved statute text, no curated protections
checklist, no clause splitting — and the output is scored against the same
manifest.json ground truth as eval_scan, with the same metrics.

Scoring is deliberately generous to the baseline: verdicts match by printed
clause number; a claimed missing protection matches checklist items by
keyword; claims matching no checklist item (federal disclosures, inventions)
are recorded but not penalized. Raw model output is saved for audit.

Usage:
    python -m evaluation.eval_baseline
    python -m evaluation.eval_baseline --model openai/gpt-4.1
"""

import argparse
import json
from pathlib import Path
from typing import Literal

from litellm import completion
from pydantic import BaseModel, Field

from leasehound.retrieval import GENERATION_MODEL, llm_retry
from leasehound.scan import base_section

LEASES_DIR = Path(__file__).parent / "leases"
MANIFEST_PATH = LEASES_DIR / "manifest.json"
RESULTS_PATH = Path(__file__).parent / "baseline_results.json"


class BaselineFlag(BaseModel):
    clause_number: int = Field(description="The clause number as printed in the lease")
    verdict: Literal["red", "yellow"] = Field(
        description="red = the clause is prohibited or unenforceable under Washington "
        "law; yellow = potentially problematic or fact-dependent"
    )
    citations: list[str] = Field(
        description="RCW section numbers grounding this verdict, e.g. ['RCW 59.18.230']"
    )
    explanation: str = Field(description="1-3 sentences in plain language")


class BaselineReport(BaseModel):
    flags: list[BaselineFlag]
    missing_protections: list[str] = Field(
        description="Disclosures or protections Washington law requires a residential "
        "lease to provide that this lease fails to include"
    )


def make_baseline_prompt(lease_text: str) -> str:
    return f"""
You are reviewing a residential lease under Washington State's Residential
Landlord-Tenant Act (RCW 59.18).

1. Identify every clause that is prohibited or unenforceable under Washington law
   (verdict "red") and every clause that is potentially problematic or fact-dependent
   (verdict "yellow"). Reference each by its printed clause number and cite the RCW
   section(s) the verdict rests on.
2. List any disclosures or protections Washington law requires that this lease fails
   to provide.

Plain language; this is legal information, not legal advice.

Lease:
{lease_text}
"""


@llm_retry
def judge_zero_shot(lease_text: str, model: str) -> BaselineReport:
    messages = [{"role": "user", "content": make_baseline_prompt(lease_text)}]
    response = completion(
        model=model, messages=messages, response_format=BaselineReport, temperature=0
    )
    return BaselineReport.model_validate_json(response.choices[0].message.content)


# Maps a free-form "this lease is missing X" claim onto the checklist items the
# manifest scores. Deposit items require deposit context so e.g. "right to
# withhold rent" doesn't read as a deposit claim. First match wins; unmatched
# claims are kept as extras and not penalized.
def _any(claim: str, *keywords: str) -> bool:
    return any(keyword in claim for keyword in keywords)


PROTECTION_MATCHERS = {
    "Deposit withholding terms":
        lambda c: "deposit" in c and _any(c, "withh", "deduct"),
    "Move-in condition checklist":
        lambda c: _any(c, "checklist", "walk-through", "walkthrough",
                       "condition report", "statement of condition"),
    "Deposit location disclosure":
        lambda c: "depository" in c
        or ("deposit" in c and _any(c, " held", "location", "bank", "trust account",
                                    "interest")),
    "Fire safety information": lambda c: "fire" in c,
    "Mold information": lambda c: "mold" in c,
}


def map_protections(claims: list[str]) -> tuple[set[str], list[str]]:
    mapped, extras = set(), []
    for claim in claims:
        lowered = claim.lower()
        for name, matches in PROTECTION_MATCHERS.items():
            if matches(lowered):
                mapped.add(name)
                break
        else:
            extras.append(claim)
    return mapped, extras


def evaluate_lease(entry: dict, model: str) -> dict:
    text = (LEASES_DIR / entry["file"]).resolve().read_text(encoding="utf-8")
    report = judge_zero_shot(text, model)
    expected_red = {int(num): cites for num, cites in entry["red"].items()}

    # Dedupe by clause number, red outranking yellow, before scoring.
    verdicts: dict[int, str] = {}
    citations: dict[int, set[str]] = {}
    for flag in report.flags:
        if verdicts.get(flag.clause_number) != "red":
            verdicts[flag.clause_number] = flag.verdict
        citations.setdefault(flag.clause_number, set()).update(
            base_section(c) for c in flag.citations
        )

    strict = [n for n in expected_red if verdicts.get(n) == "red"]
    lenient_only = [n for n in expected_red if verdicts.get(n) == "yellow"]
    missed = [n for n in expected_red if n not in verdicts]
    false_reds = [n for n, v in verdicts.items() if v == "red" and n not in expected_red]
    citation_hits = sum(1 for n in strict if citations[n] & set(expected_red[n]))

    mapped, extras = map_protections(report.missing_protections)
    expected_missing = set(entry["missing_protections"])
    return {
        "file": entry["file"],
        "planted": sorted(expected_red),
        "flagged_red": sorted(strict),
        "flagged_yellow": sorted(lenient_only),
        "missed": sorted(missed),
        "false_reds": sorted(false_reds),
        "citation_hits": citation_hits,
        "protections_expected": sorted(expected_missing),
        "protections_reported": sorted(mapped),
        "protections_unmatched": extras,
        "protections_exact": mapped == expected_missing,
        "raw_flags": [flag.model_dump() for flag in report.flags],
        "raw_missing": report.missing_protections,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=GENERATION_MODEL,
                        help="litellm model id (default: the pipeline's generation model)")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    results = []
    for entry in manifest["leases"]:
        print(f"zero-shot {args.model}: {entry['file']}")
        results.append(evaluate_lease(entry, args.model))

    planted = sum(len(r["planted"]) for r in results)
    strict = sum(len(r["flagged_red"]) for r in results)
    lenient = strict + sum(len(r["flagged_yellow"]) for r in results)
    false_reds = sum(len(r["false_reds"]) for r in results)
    citation_hits = sum(r["citation_hits"] for r in results)
    protections_exact = sum(1 for r in results if r["protections_exact"])

    summary = {
        "leases": len(results),
        "planted_violations": planted,
        "strict_recall": round(strict / planted, 4),
        "lenient_recall": round(lenient / planted, 4),
        "precision": round(strict / (strict + false_reds), 4) if strict + false_reds else 1.0,
        "citation_accuracy": round(citation_hits / strict, 4) if strict else 0.0,
        "false_reds": false_reds,
        "protections_exact_leases": f"{protections_exact}/{len(results)}",
    }

    # One results file, one entry per model — reruns overwrite their own row.
    existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    existing[args.model] = {"summary": summary, "leases": results}
    RESULTS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    for r in results:
        flag = "OK " if not r["missed"] and not r["false_reds"] and r["protections_exact"] else "!!"
        print(f"{flag} {r['file']}: red {r['flagged_red']} of {r['planted']}, "
              f"yellow {r['flagged_yellow']}, missed {r['missed']}, false {r['false_reds']}, "
              f"protections {'exact' if r['protections_exact'] else r['protections_reported']}")
    print(f"Details written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
