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
gate and the required-protections pass per document, and skips any document over
the clause cap rather than quietly spending on a book.
"""

import argparse
import json
import urllib.request
from pathlib import Path

from evaluation.provenance import stamp
from leasehound.metrics import ScanMeter
from leasehound.scan import MAX_CLAUSES, check_protections, looks_like_lease
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

    if not scan:
        return result
    if result["over_clause_cap"]:
        result["scan"] = f"skipped — {len(clauses)} clauses is over the {MAX_CLAUSES} cap"
        return result

    meter = ScanMeter()
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="Download the source documents")
    parser.add_argument("--scan", action="store_true", help="Also run the paid LLM passes")
    args = parser.parse_args()

    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))["documents"]
    if args.fetch:
        fetch(sources)

    results = [measure(entry, args.scan) for entry in sources]
    total = sum(r.get("cost_usd", 0) for r in results)
    output = {"documents": len(results), "cost_usd": round(total, 5),
              "provenance": stamp(), "results": results}
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    for r in results:
        if r.get("error"):
            print(f"!! {r['file']}: {r['error']}")
            continue
        line = (f"   {r['file']:38} {r['chars']:6} chars  {r['split_mode']:10} "
                f"clauses={r['clauses']:3} median={r['median_clause_chars']:5} "
                f"over_window={r['clauses_over_retrieval_window']}")
        if r.get("over_clause_cap"):
            line += f"  OVER {MAX_CLAUSES}-CLAUSE CAP"
        if "gate_accepted_as_lease" in r:
            line += f"  gate={'lease' if r['gate_accepted_as_lease'] else 'REJECTED'}"
        print(line)
    print(f"\nTotal cost: ${total:.4f}. Details written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
