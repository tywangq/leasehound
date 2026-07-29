"""Scan-mode retrieval: does the governing statute reach the judge at all?

The ablation suite measures retrieval for *ask* mode — colloquial questions
generated from a known section. Scan mode, which is the product's headline
feature, had no retrieval eval: its retrieval was only ever observed indirectly,
through whether a verdict came out red. That is a bad instrument, because a
missed violation has two very different causes — retrieval never surfaced the
governing law, or it did and the judge misread it — and telling them apart took
three wrong hypotheses the one time it mattered (see the README's hybrid
experiment). This makes the distinction mechanical.

The labels already existed. The gold manifest maps each planted violation's
printed clause number to the sections that would be an acceptable citation, and
`scan_clause` uses the clause text as its own query. So for every labelled
clause: embed it, retrieve, and check whether an acceptable section came back.

Cost is one embedding per labelled clause and no completions at all, which is
why this can be run before deciding whether any paid eval is worth it.

Read `section_hit` narrowly. It asks whether a chunk *from* an acceptable
section arrived, not whether the chunk containing the governing rule arrived — a
section split across four chunks satisfies it when the wrong one surfaces, which
is exactly the failure the hybrid experiment eventually traced. Section
completion closes that gap by construction: hand the judge every chunk of a hit
section and a section hit becomes a chunk hit.

    python -m evaluation.eval_scan_retrieval
    python -m evaluation.eval_scan_retrieval --manifest evaluation/leases_synthetic/manifest.json
    python -m evaluation.eval_scan_retrieval --section-completion
"""

import argparse
import json
from pathlib import Path

from evaluation.eval_scan import printed_number
from evaluation.provenance import stamp
from leasehound.retrieval import fetch_unranked
from leasehound.scan import base_section, scan_config
from leasehound.upload import MAX_CLAUSE_CHARS, read_document, split_clauses_with_mode

MANIFEST_PATH = Path(__file__).parent / "leases" / "manifest.json"
RESULTS_PATH = Path(__file__).parent / "scan_retrieval_results.json"


def ranks_by_section(chunks: list, acceptable: set[str]) -> dict[str, int | None]:
    """1-based rank of the first chunk of each acceptable section, or None.

    Both a lenient and a strict reading are needed, and the difference is not
    cosmetic. A manifest often accepts more than one citation — the exculpation
    clause in lease 018 accepts either RCW 59.18.060 (landlord duties) or
    RCW 59.18.230 (the prohibition itself). Scoring "did any acceptable section
    arrive" calls that a hit when .060 shows up and .230 never does, which is
    precisely the case the scanner missed. So report both.
    """
    found: dict[str, int | None] = dict.fromkeys(sorted(acceptable))
    for position, chunk in enumerate(chunks, start=1):
        section = base_section(chunk.metadata.get("section", ""))
        if section in found and found[section] is None:
            found[section] = position
    return found


def evaluate(manifest_path: Path, config) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = []
    for entry in manifest["leases"]:
        text = read_document(manifest_path.parent / entry["file"])
        clauses, _ = split_clauses_with_mode(text)
        by_number = {printed_number(c): c for c in clauses}
        for number, citations in entry["red"].items():
            clause = by_number.get(int(number))
            if clause is None:
                # The splitter did not recover this labelled clause at all; that is
                # a splitting failure, not a retrieval one, and must not be scored
                # as a retrieval hit or silently dropped.
                cases.append({"file": entry["file"], "clause": int(number),
                              "error": "clause not recovered by the splitter"})
                continue
            acceptable = {base_section(c) for c in citations}
            # Same truncation scan_clause applies, so this measures the real query.
            chunks = fetch_unranked(clause[:MAX_CLAUSE_CHARS], config)
            ranks = ranks_by_section(chunks, acceptable)
            hit_ranks = [r for r in ranks.values() if r is not None]
            cases.append({
                "file": entry["file"], "clause": int(number),
                "acceptable": sorted(acceptable),
                "retrieved_sections": [c.metadata.get("section") for c in chunks],
                "ranks": ranks,
                "rank": min(hit_ranks) if hit_ranks else None,
                "all_acceptable_arrived": len(hit_ranks) == len(ranks),
                "absent": sorted(k for k, v in ranks.items() if v is None),
            })

    scored = [c for c in cases if "rank" in c]
    n = len(scored)
    hits = [c["rank"] for c in scored if c["rank"] is not None]
    summary = {
        "labelled_clauses": len(cases),
        "scored": n,
        "not_recovered_by_splitter": len(cases) - n,
        "section_hit@1": round(sum(1 for r in hits if r <= 1) / n, 4) if n else None,
        "section_hit@3": round(sum(1 for r in hits if r <= 3) / n, 4) if n else None,
        "section_hit@5": round(sum(1 for r in hits if r <= 5) / n, 4) if n else None,
        "section_hit@k": round(len(hits) / n, 4) if n else None,
        "mrr": round(sum(1 / r for r in hits) / n, 4) if n else None,
        # Stricter: EVERY section the manifest accepts arrived, not just one of
        # them. This is the reading that exposes a governing prohibition going
        # missing behind a more general section that happens to also be accepted.
        "all_acceptable_arrived": round(
            sum(1 for c in scored if c["all_acceptable_arrived"]) / n, 4) if n else None,
        "k": config.retrieval_k,
        "collection": config.collection,
        "section_completion": config.section_completion,
        "bm25": config.bm25,
        "misses": [f"{c['file']} clause {c['clause']}" for c in scored if c["rank"] is None],
        "partial": [f"{c['file']} clause {c['clause']} missing {c['absent']}"
                    for c in scored if c["rank"] is not None and not c["all_acceptable_arrived"]],
    }
    return {"summary": summary, "provenance": stamp(), "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--results", default=str(RESULTS_PATH))
    parser.add_argument("--section-completion", action="store_true",
                        help="Expand each hit section to all of its chunks")
    parser.add_argument("--bm25", action="store_true", help="Add the lexical channel")
    parser.add_argument("--collection",
                        help="Retrieve from a different collection, e.g. wa_reference_230split "
                             "from scripts/build_enumerated_collection.py")
    args = parser.parse_args()

    config = scan_config("wa")
    config.section_completion = args.section_completion
    config.bm25 = args.bm25
    if args.collection:
        config.collection = args.collection
    output = evaluate(Path(args.manifest), config)

    print(json.dumps(output["summary"], indent=2))
    for case in output["cases"]:
        if case.get("error"):
            print(f"!! {case['file']} clause {case['clause']}: {case['error']}")
        elif case["rank"] is None:
            print(f"MISS {case['file']} clause {case['clause']}: wanted {case['acceptable']}, "
                  f"got {case['retrieved_sections'][:5]}")
    Path(args.results).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Details written to {args.results}")


if __name__ == "__main__":
    main()
