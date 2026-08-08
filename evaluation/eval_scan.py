"""Scan-layer evaluation: red-flag precision/recall on labeled synthetic leases.

Ground truth lives in a manifest.json: each lease lists which printed clause
numbers contain planted violations (with acceptable citations) and which
required protections are genuinely absent. Verdict correctness is the primary
metric; citation accuracy is reported alongside.

Two sets share this scorer:
- gold (leases/, default): 6 hand-labeled leases — the acceptance bar.
- silver (leases_synthetic/): generated at scale by make_synthetic_leases.py,
  including prompt-injection leases; scored per category.

Scoring:
- strict recall    planted violations flagged red
- lenient recall   planted violations flagged red OR yellow
- precision        red flags that are planted violations / all red flags
- citation hit     red-flagged planted clauses citing an acceptable section
- protections      exact set match of reported vs expected missing protections

A lease the pipeline refuses to scan (e.g. an injection that tricks the
is-this-a-lease gate) scores as every planted violation missed — an attack
that suppresses the scan must cost recall, not crash the eval.

Usage:
    python -m evaluation.eval_scan
    python -m evaluation.eval_scan --manifest evaluation/leases_synthetic/manifest.json \
        --results evaluation/synthetic_results.json
"""

import argparse
import json
import re
from pathlib import Path

from evaluation.provenance import stamp
from leasehound.scan import base_section, scan_lease

LEASES_DIR = Path(__file__).parent / "leases"
MANIFEST_PATH = LEASES_DIR / "manifest.json"
RESULTS_PATH = Path(__file__).parent / "scan_results.json"

CLAUSE_NUMBER_RE = re.compile(r"^\s*(\d{1,2})[.)]\s")


def printed_number(clause_text: str) -> int | None:
    match = CLAUSE_NUMBER_RE.match(clause_text)
    return int(match.group(1)) if match else None


def evaluate_lease(entry: dict, leases_dir: Path, keep_raw: bool = False,
                   collection: str | None = None) -> dict:
    """Scan one lease and score it. With keep_raw, the scan's own findings and
    protections ride along under "_scan" so a caller can reuse them (the
    injection eval renders the report from them) without paying to rescan."""
    path = (leases_dir / entry["file"]).resolve()
    expected_red = {int(num): cites for num, cites in entry["red"].items()}
    expected_missing = set(entry["missing_protections"])
    result = {"file": entry["file"]}
    if entry.get("category"):
        result["category"] = entry["category"]

    scan = scan_lease(path, collection=collection)
    findings, protections = scan.findings, scan.protections
    gate_flagged = scan.gate_flagged
    if scan.refused:
        # This branch used to be `except SystemExit`, and it had quietly stopped
        # firing: refusing was a raise when the gate answered yes-or-no, and became a
        # `refused` result when it grew three kinds. A refused lease still scored 0
        # recall — the manifest supplies `planted` either way — but the loud "REFUSED
        # by the pipeline" line never printed, so a gate regression would have looked
        # like a judge that suddenly missed every violation in a lease.
        #
        # The scoring is unchanged and deliberate: an attack that suppresses the scan
        # must cost recall rather than crash the eval, so every planted violation
        # counts as missed.
        return {
            **result,
            "rejected": "the gate refused this document as unrelated to renting",
            "planted": sorted(expected_red),
            "flagged_red": [], "flagged_yellow": [],
            "missed": sorted(expected_red),
            "false_reds": [], "citation_hits": 0,
            "unplanted_yellows": [], "clauses_judged": 0,
            "protections_expected": sorted(expected_missing),
            "protections_reported": [],
            "protections_exact": not expected_missing,
        }

    strict, lenient_only, missed, false_reds, citation_hits = [], [], [], [], 0
    # The evidence behind the two scores that get quoted. citation_accuracy used to
    # be a bare count in the artifact, so "18/18 cited correctly" could not be
    # checked without paying for a re-run; and false_reds held printed clause
    # NUMBERS with no text, which is how a false red once got diagnosed against the
    # wrong clause — printed numbers and indexes disagree by one in some leases.
    citations: dict[str, list[str]] = {}
    false_red_clauses: dict[str, str] = {}
    # Yellow on a clause with nothing planted in it used to fall through both arms of
    # this loop and vanish. That made a scanner which cautions on everything score
    # perfectly on the property this project defends most loudly: it would post zero
    # false reds while telling a renter that half their lease is questionable. Lenient
    # recall counts yellow on a PLANTED clause and so does not see it either.
    #
    # Recorded with the clause text, like false reds, so the next paid run makes the
    # cautions auditable instead of merely counted — one of these is a defensible
    # "fact-dependent" reading and another is noise, and only the text says which.
    cautions: dict[str, str] = {}
    for finding in findings:
        number = printed_number(finding["clause"])
        if number in expected_red:
            if finding["verdict"] == "red":
                strict.append(number)
                cited = {base_section(c) for c in finding["citations"]}
                citations[str(number)] = sorted(cited)
                if cited & set(expected_red[number]):
                    citation_hits += 1
            elif finding["verdict"] == "yellow":
                lenient_only.append(number)
            else:
                missed.append(number)
        elif finding["verdict"] == "red":
            false_reds.append(number)
            false_red_clauses[str(number)] = finding["clause"][:300]
        elif finding["verdict"] == "yellow":
            cautions[str(number)] = finding["clause"][:300]

    reported_missing = {p["name"] for p in protections if p["status"] == "missing"}
    result["gate_flagged"] = gate_flagged
    if scan.partial:
        # No labelled lease is over the clause cap today, so this never fires. It
        # exists because if one ever is, every planted violation past the cap would
        # score as a recall miss and quietly make the scanner look worse than it is.
        result["partial_scan"] = f"{scan.clauses_judged}/{scan.clauses_total} clauses judged"
    if keep_raw:
        result["_scan"] = {"findings": findings, "protections": protections,
                           "gate_flagged": gate_flagged}
    return {
        **result,
        "planted": sorted(expected_red),
        "flagged_red": sorted(strict),
        "flagged_yellow": sorted(lenient_only),
        "missed": sorted(missed),
        "false_reds": sorted(f for f in false_reds if f is not None),
        "citation_hits": citation_hits,
        # Recorded so the two scores above are auditable from the artifact rather
        # than only from a re-run. Not backfilled: re-running a paid eval to add
        # evidence for numbers that did not change is the kind of spend this
        # project declines, so these appear from the next legitimate run onward.
        "citations_by_clause": {k: citations[k] for k in sorted(citations)},
        "false_red_clauses": {k: false_red_clauses[k] for k in sorted(false_red_clauses)},
        # Cautions on clauses the manifest says nothing about. On a lease with no
        # planted violations at all, every one of these is a caution over a clause the
        # label set calls fine.
        "unplanted_yellows": sorted(int(k) for k in cautions),
        "unplanted_yellow_clauses": {k: cautions[k] for k in sorted(cautions)},
        "clauses_judged": len(findings),
        "protections_expected": sorted(expected_missing),
        "protections_reported": sorted(reported_missing),
        "protections_exact": reported_missing == expected_missing,
    }


def summarize(results: list[dict]) -> dict:
    planted = sum(len(r["planted"]) for r in results)
    strict = sum(len(r["flagged_red"]) for r in results)
    lenient = strict + sum(len(r["flagged_yellow"]) for r in results)
    false_reds = sum(len(r["false_reds"]) for r in results)
    citation_hits = sum(r["citation_hits"] for r in results)
    protections_exact = sum(1 for r in results if r["protections_exact"])

    # The caution rate, over the clauses where a caution is the pipeline's own idea
    # rather than a hedge on something planted. Reported as a rate because the
    # degenerate scanner this guards against is the one that yellows everything: on a
    # clean lease "0 false reds" and "every clause is questionable" are the same score.
    judged = sum(r.get("clauses_judged", 0) for r in results)
    unplanted_yellows = sum(len(r.get("unplanted_yellows", [])) for r in results)

    summary = {
        "leases": len(results),
        "planted_violations": planted,
        "strict_recall": round(strict / planted, 4) if planted else None,
        "lenient_recall": round(lenient / planted, 4) if planted else None,
        "precision": round(strict / (strict + false_reds), 4) if strict + false_reds else 1.0,
        "citation_accuracy": round(citation_hits / strict, 4) if strict else None,
        "false_reds": false_reds,
        "unplanted_yellows": unplanted_yellows,
        "unplanted_yellow_rate": (round(unplanted_yellows / judged, 4)
                                  if judged else None),
        "clauses_judged": judged,
        "protections_exact_leases": f"{protections_exact}/{len(results)}",
    }
    rejected = sum(1 for r in results if r.get("rejected"))
    if rejected:
        summary["rejected_leases"] = rejected
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_PATH),
                        help="ground-truth manifest (default: the gold set)")
    parser.add_argument("--results", default=str(RESULTS_PATH),
                        help="where to write the detailed results JSON")
    parser.add_argument("--collection",
                        help="retrieve from a non-default collection — this is the paid "
                             "precision gate for an experimental index")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = [evaluate_lease(entry, manifest_path.parent, collection=args.collection)
               for entry in manifest["leases"]]

    provenance = stamp()
    if args.collection:
        provenance["collection"] = args.collection
    output = {"summary": summarize(results), "provenance": provenance}
    categories = sorted({r["category"] for r in results if r.get("category")})
    if categories:
        output["by_category"] = {
            c: summarize([r for r in results if r.get("category") == c])
            for c in categories
        }
    output["leases"] = results

    Path(args.results).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "leases"}, indent=2))
    for r in results:
        if r.get("rejected"):
            print(f"!! {r['file']}: REFUSED by the pipeline — {r['rejected']}")
            continue
        flag = "OK " if not r["missed"] and not r["false_reds"] and r["protections_exact"] else "!!"
        print(f"{flag} {r['file']}: red {r['flagged_red']} of {r['planted']}, "
              f"yellow {r['flagged_yellow']}, missed {r['missed']}, false {r['false_reds']}, "
              f"protections {'exact' if r['protections_exact'] else r['protections_reported']}")
    print(f"Details written to {args.results}")


if __name__ == "__main__":
    main()
