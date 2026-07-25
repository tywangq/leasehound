"""Scan-layer evaluation: red-flag precision/recall on hand-labeled synthetic leases.

Ground truth lives in leases/manifest.json: each lease lists which printed clause
numbers contain planted violations (with acceptable citations) and which required
protections are genuinely absent. Verdict correctness is the primary metric;
citation accuracy is reported alongside.

Scoring:
- strict recall    planted violations flagged red
- lenient recall   planted violations flagged red OR yellow
- precision        red flags that are planted violations / all red flags
- citation hit     red-flagged planted clauses citing an acceptable section
- protections      exact set match of reported vs expected missing protections

Usage:
    python -m evaluation.eval_scan
"""

import json
import re
from pathlib import Path

from leasehound.scan import base_section, scan_lease

LEASES_DIR = Path(__file__).parent / "leases"
MANIFEST_PATH = LEASES_DIR / "manifest.json"
RESULTS_PATH = Path(__file__).parent / "scan_results.json"

CLAUSE_NUMBER_RE = re.compile(r"^\s*(\d{1,2})[.)]\s")


def printed_number(clause_text: str) -> int | None:
    match = CLAUSE_NUMBER_RE.match(clause_text)
    return int(match.group(1)) if match else None


def evaluate_lease(entry: dict) -> dict:
    path = (LEASES_DIR / entry["file"]).resolve()
    findings, protections = scan_lease(path)
    expected_red = {int(num): cites for num, cites in entry["red"].items()}

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
    expected_missing = set(entry["missing_protections"])
    return {
        "file": entry["file"],
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


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    results = [evaluate_lease(entry) for entry in manifest["leases"]]

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

    RESULTS_PATH.write_text(
        json.dumps({"summary": summary, "leases": results}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    for r in results:
        flag = "OK " if not r["missed"] and not r["false_reds"] and r["protections_exact"] else "!!"
        print(f"{flag} {r['file']}: red {r['flagged_red']} of {r['planted']}, "
              f"yellow {r['flagged_yellow']}, missed {r['missed']}, false {r['false_reds']}, "
              f"protections {'exact' if r['protections_exact'] else r['protections_reported']}")
    print(f"Details written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
