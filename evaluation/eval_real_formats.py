"""Real published housing documents: does the pipeline survive real formatting?

Every labeled lease in this repo was written for it — by hand or by a generator
following a spec — so both sets share one authorial voice and one numbering
convention. That makes them useless for one specific question: what happens when
a document comes from somewhere else entirely, as a PDF, with page furniture and
signature blocks and thirty pages of enumerated sub-provisions.

There are no labels here, so this reports no precision or recall. It probes the
deterministic front of the pipeline — pypdf extraction, then clause splitting —
and optionally the two whole-document LLM passes. Findings from it belong in the
README as observations, never as scores.

The documents are government and public-university publications, listed with
provenance in sources.json and deliberately not committed: they are third-party
files, they are large, and their canonical home is the URL. Run --fetch to get
them.

    python -m evaluation.eval_real_formats                 # free: extract + split
    python -m evaluation.eval_real_formats --fetch         # download the sources
    python -m evaluation.eval_real_formats --scan          # + the paid LLM passes

Splitting costs nothing, so it is the default. --scan adds the is-this-a-lease
gate and the required-protections pass per document. Neither is capped by clause
count: the gate reads the opening, and the protections pass reads every clause in
24k windows because "this lease omits X" is a claim about the whole document.
"""

import argparse
import json
import urllib.request
from pathlib import Path

from evaluation.provenance import stamp
from leasehound.metrics import UsageMeter
from leasehound.scan import (
    MAX_CLAUSES,
    check_protections,
    looks_like_lease,
    protection_windows,
)
from leasehound.upload import MAX_CLAUSE_CHARS, read_document, split_clauses_with_mode

DOCS_DIR = Path(__file__).parent / "leases_real"
SOURCES_PATH = DOCS_DIR / "sources.json"
RESULTS_PATH = Path(__file__).parent / "real_format_results.json"

TINY_CLAUSE = 120


def fetch(sources: list[dict]) -> None:
    for entry in sources:
        target = DOCS_DIR / entry["file"]
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
        print(f"{entry['file']}  {target.stat().st_size} bytes  <- {entry['url']}")


def measure(entry: dict, scan: bool) -> dict:
    path = DOCS_DIR / entry["file"]
    result = {"file": entry["file"], "publisher": entry["publisher"], "kind": entry["kind"]}
    if not path.exists():
        return {**result, "error": "not downloaded — run with --fetch"}

    try:
        text = read_document(path)
    except Exception as err:
        # A real PDF can fail to parse in ways the labeled .md corpus never does.
        return {**result, "error": f"extraction failed: {err!r}"}

    clauses, mode = split_clauses_with_mode(text)
    result.update(chars=len(text), split_mode=mode, clauses=len(clauses))
    if not clauses:
        return {**result, "error": "no clauses recovered"}

    lengths = sorted(len(c) for c in clauses)
    result.update(
        median_clause_chars=lengths[len(lengths) // 2],
        max_clause_chars=lengths[-1],
        # Zero is the invariant: every clause must fit the retrieval window, so
        # each one is its own complete query rather than a truncated stand-in.
        clauses_over_retrieval_window=sum(1 for n in lengths if n > MAX_CLAUSE_CHARS),
        tiny_clauses=sum(1 for n in lengths if n < TINY_CLAUSE),
        over_clause_cap=len(clauses) > MAX_CLAUSES,
    )

    # How many prompts the protections pass needs. This was 1 by assumption and the
    # assumption was wrong: at 39,996 characters the HUD model lease overflowed the
    # single 24k prompt, so the protections result published here previously described
    # 28 of its 49 clauses and reported the rest missing without reading them.
    result["protection_windows"] = len(protection_windows(clauses))

    if not scan:
        return result

    meter = UsageMeter()
    accepted = looks_like_lease(clauses, meter)
    result["gate_accepted_as_lease"] = accepted
    if accepted:
        protections = check_protections(clauses, meter)
        result["protections_missing"] = sorted(
            p["name"] for p in protections if p["status"] == "missing"
        )
        result["protections_present"] = sum(1 for p in protections if p["status"] == "present")
    result["llm_calls"] = meter.llm_calls
    result["cost_usd"] = round(meter.cost_usd, 5)
    return result


# Keys only a --scan run produces. A free run must not erase them.
PAID_KEYS = ("gate_accepted_as_lease", "protections_missing", "protections_present",
             "llm_calls", "cost_usd")


def carry_paid_results(results: list[dict], existing: dict | None) -> list[dict]:
    """Preserve a previous paid run's findings through a later free run.

    This exists because it already went wrong. The paid probe (commit 618c00d)
    recorded the protections verdicts this eval's write-up discusses; a later free
    run — same script, same output path, just without --scan — overwrote the file
    and silently deleted them, leaving evaluation/README.md pointing at evidence that was no
    longer in the artifact. Splitting cost nothing and so gets run casually, which
    is exactly why it must not be able to destroy something that cost money.
    """
    if not existing:
        return results
    previous = {r.get("file"): r for r in existing.get("results", [])}
    for result in results:
        # Per document, and all-or-nothing. Carrying key by key would let this run's
        # protections verdict sit beside an older run's gate decision in one record
        # that describes neither run — worse than losing the old numbers outright.
        if any(key in result for key in PAID_KEYS):
            continue
        carried = {key: previous.get(result.get("file"), {})[key] for key in PAID_KEYS
                   if key in previous.get(result.get("file"), {})}
        if carried:
            result.update(carried)
            result["paid_fields_carried_from"] = existing.get("provenance", {}).get("commit")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="Download the source documents")
    parser.add_argument("--scan", action="store_true", help="Also run the paid LLM passes")
    args = parser.parse_args()

    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))["documents"]
    if args.fetch:
        fetch(sources)

    existing = (json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
                if RESULTS_PATH.exists() else None)
    results = carry_paid_results([measure(entry, args.scan) for entry in sources], existing)
    total = sum(r.get("cost_usd", 0) for r in results)
    output = {"documents": len(results), "cost_usd": round(total, 5),
              "scan_run": args.scan, "provenance": stamp(), "results": results}
    if not args.scan and existing:
        output["note"] = ("Free run: splitting was re-measured, and any paid fields shown were "
                          "carried forward from an earlier --scan run rather than re-paid for.")
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    for r in results:
        if r.get("error"):
            print(f"!! {r['file']}: {r['error']}")
            continue
        line = (f"   {r['file']:38} {r['chars']:6} chars  {r['split_mode']:10} "
                f"clauses={r['clauses']:3} median={r['median_clause_chars']:5} "
                f"over_window={r['clauses_over_retrieval_window']}")
        if r.get("over_clause_cap"):
            line += f"  partial: {MAX_CLAUSES}/{r['clauses']} judged"
        if r.get("protection_windows", 1) > 1:
            line += f"  protections in {r['protection_windows']} windows"
        if "gate_accepted_as_lease" in r:
            line += f"  gate={'lease' if r['gate_accepted_as_lease'] else 'REJECTED'}"
        print(line)
    print(f"\nTotal cost: ${total:.4f}. Details written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
