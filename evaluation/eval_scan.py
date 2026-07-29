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

    try:
        scan = scan_lease(path, collection=collection)
        findings, protections = scan.findings, scan.protections
        gate_flagged = scan.gate_flagged
    except SystemExit as stop:
        return {
            **result,
            "rejected": str(stop),
            "planted": sorted(expected_red),
            "flagged_red": [], "flagged_yellow": [],
            "missed": sorted(expected_red),
            "false_reds": [], "citation_hits": 0,
            "protections_expected": sorted(expected_missing),
            "protections_reported": [],
            "protections_exact": not expected_missing,
        }

    strict, lenient_only, missed, false_reds, citation_hits = [], [], [], [], 0
    for finding in findings:
        number = printed_number(finding["clause"])
        if number in expected_red:
            if finding["verdict"] == "red":
                strict.append(number)
                cited = {base_section(c) for c in finding["citations"]}
                if cited & set(expected_red[number]):
                    citation_hits += 1
            elif finding["verdict"] == "yellow":
                lenient_only.append(number)
            else:
                missed.append(number)
        elif finding["verdict"] == "red":
            false_reds.append(number)

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

    summary = {
        "leases": len(results),
        "planted_violations": planted,
        "strict_recall": round(strict / planted, 4) if planted else None,
        "lenient_recall": round(lenient / planted, 4) if planted else None,
        "precision": round(strict / (strict + false_reds), 4) if strict + false_reds else 1.0,
        "citation_accuracy": round(citation_hits / strict, 4) if strict else None,
        "false_reds": false_reds,
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
