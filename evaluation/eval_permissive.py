"""Permissive vs prohibited: can the judge tell "you may" from "you must only"?

This exists to unblock one specific decision. `split_enumerated_catalog` indexes
the ten prohibitions of RCW 59.18.230 separately and takes strict scan retrieval
from **.492 to .984** — measured, reproducible, and still not shipped, because the
paid gold set caught it costing **one false red**: a clause permitting *check,
money order, or electronic transfer* was read as violating (2)(j), which prohibits
paying by "electronic means only". Making the governing rule reachable put a crisp
narrow prohibition in front of the judge and the judge over-applied it.

The write-up said the honest fix needs a labelled set rather than a prompt rule
written against the one observed failure — that would be buying a passing number.
This is that set: 30 clauses across all ten prohibitions, in three kinds.

  * `prohibited` — must come back red. Here so a fix that suppresses false reds by
    suppressing reds is caught rather than celebrated.
  * `permissive` — lawful, offers the tenant options. A red here is a false red.
  * `permissive_hard` — lawful, and *names the prohibited thing*: a CONFIDENTIALITY
    heading whose obligation runs against the landlord, a CONFESSION OF JUDGMENT
    clause that forbids one, late-charge wording lifted from the statute. Testing
    only the easy permissive cases would measure almost nothing, because the
    observed failure was of this shape.

The clauses are written against the statute text in `corpus/wa/statutes/`, not
invented, and each row records which subsection it targets and why it is labelled
that way.

    python -m evaluation.eval_permissive                                   # shipped index
    python -m evaluation.eval_permissive --collection wa_reference_230split  # the candidate
    python -m evaluation.eval_permissive --both                            # both, one artifact

≈ $0.025 per collection: one embedding and one judge call per clause, no scan
orchestration, no protections pass.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluation.provenance import stamp
from leasehound.metrics import UsageMeter
from leasehound.scan import MAX_PARALLEL_SCANS, scan_clause, scan_config

PAIRS_PATH = Path(__file__).parent / "permissive_pairs.jsonl"
RESULTS_PATH = Path(__file__).parent / "permissive_results.json"

SHIPPED = "wa_reference"
CANDIDATE = "wa_reference_230split"

# What each label expects of the verdict. Yellow is acceptable on a permissive
# clause — "fact-dependent" is a defensible reading of a lawful-but-unusual term —
# but red is not, because red is what a renter acts on.
PERMISSIVE_KINDS = ("permissive", "permissive_hard")


def load_pairs() -> list[dict]:
    return [json.loads(line) for line in
            PAIRS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def judge_all(pairs: list[dict], collection: str) -> tuple[list[dict], UsageMeter]:
    config = scan_config("wa", collection)
    meter = UsageMeter()

    def judge(pair: dict) -> dict:
        finding = scan_clause(pair["clause"], 0, config, meter)
        expected_red = pair["kind"] == "prohibited"
        red = finding["verdict"] == "red"
        return {
            "id": pair["id"],
            "prohibition": pair["prohibition"],
            "kind": pair["kind"],
            "verdict": finding["verdict"],
            "citations": finding["citations"],
            # The one field a reader should scan for. A prohibited clause that is
            # not red is a miss; a permissive clause that is red is a false red.
            "correct": red == expected_red,
            "explanation": finding["explanation"],
        }

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SCANS) as pool:
        return list(pool.map(judge, pairs)), meter


def score(rows: list[dict]) -> dict:
    prohibited = [r for r in rows if r["kind"] == "prohibited"]
    permissive = [r for r in rows if r["kind"] in PERMISSIVE_KINDS]
    false_reds = [r for r in permissive if r["verdict"] == "red"]
    return {
        "prohibited_total": len(prohibited),
        "prohibited_red": sum(1 for r in prohibited if r["verdict"] == "red"),
        "permissive_total": len(permissive),
        "false_reds": len(false_reds),
        "false_red_ids": sorted(r["id"] for r in false_reds),
        # Split out, because the easy and the hard permissive cases answer different
        # questions and averaging them hides which one moved.
        "false_reds_easy": sum(1 for r in false_reds if r["kind"] == "permissive"),
        "false_reds_hard": sum(1 for r in false_reds if r["kind"] == "permissive_hard"),
        "verdicts": {v: sum(1 for r in rows if r["verdict"] == v)
                     for v in ("red", "yellow", "green")},
    }


def run(collection: str, pairs: list[dict]) -> dict:
    print(f"\n{collection}: judging {len(pairs)} clauses…")
    rows, meter = judge_all(pairs, collection)
    summary = score(rows)
    print(f"  prohibited flagged red: {summary['prohibited_red']}/{summary['prohibited_total']}")
    print(f"  false reds on lawful clauses: {summary['false_reds']}/{summary['permissive_total']}"
          f"  (easy {summary['false_reds_easy']}, hard {summary['false_reds_hard']})")
    for row in rows:
        if not row["correct"]:
            print(f"    ! {row['id']:26} {row['kind']:16} -> {row['verdict']}"
                  f"  cites {row['citations']}")
    print(f"  ${meter.cost_usd:.4f} · {meter.llm_calls} LLM calls")
    return {"collection": collection, "summary": summary,
            "cost_usd": round(meter.cost_usd, 5), "clauses": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=SHIPPED,
                        help=f"index to retrieve from (default {SHIPPED})")
    parser.add_argument("--both", action="store_true",
                        help=f"run {SHIPPED} and {CANDIDATE} into one artifact, which is "
                             "the comparison that decides whether the split can ship")
    parser.add_argument("--write", action="store_true", help=f"write {RESULTS_PATH.name}")
    args = parser.parse_args()

    pairs = load_pairs()
    collections = [SHIPPED, CANDIDATE] if args.both else [args.collection]
    print(f"{len(pairs)} clauses × {len(collections)} collection(s) "
          f"≈ ${0.00085 * len(pairs) * len(collections):.3f}")

    runs = [run(collection, pairs) for collection in collections]
    if args.write:
        RESULTS_PATH.write_text(json.dumps(
            {"provenance": stamp(),
             "cost_usd": round(sum(r["cost_usd"] for r in runs), 5),
             "runs": runs}, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
