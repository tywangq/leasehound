"""Does the ask-mode router send real legal questions to the chitchat path?

Found by metering ask mode, not by reading it. The router is one nano call that
decides whether the whole retrieval pipeline runs, and it had no test of its own
because testing it needs an API. What the metrics log made visible was a question
that answered in 1.8 s for $0.00015 — two calls, no retrieval — when every
neighbouring question took 5-6 s and five calls. That is what a misroute looks
like from the outside: not an error, just an answer with no law in it.

The pattern the probe found: a habitability complaint stated as a FACT, with no
mention of the landlord and no legal vocabulary, was classified small_talk —
"there are cockroaches everywhere. what can i do?" went to the chitchat path 5
times out of 5, against the router's own definition of the category, which lists
only greetings, thanks, goodbyes and questions about the assistant. Naming the
landlord or the word "rights" fixed it, which is exactly backwards: the renter
least able to phrase the question legally is the one who most needs the statute.

Each case is asked N times because the router runs at the provider's default
temperature, and a single sample cannot tell a wrong answer from a coin flip —
"my heater has been broken for two weeks" came back legal 1 time in 5.

Router-only, so the whole probe is nano calls on ~200-token prompts: under a cent
for the default 5 repeats.

    python -m scripts.probe_router
    python -m scripts.probe_router --repeats 10
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from evaluation.provenance import stamp
from leasehound import answer
from leasehound.answer import ROUTER_MODEL, needs_retrieval

RESULTS_PATH = Path(__file__).parent.parent / "evaluation" / "router_results.json"

# Every case carries the answer it must get. The habitability group is the group
# that regressed; the controls are here so a prompt change that fixes them by
# routing EVERYTHING to retrieval cannot pass — that would just move the cost
# instead of fixing the routing.
CASES = [
    # A described problem, no landlord named, no legal vocabulary. The failing group.
    ("There are cockroaches everywhere. What can I do?", True, "habitability"),
    ("My toilet has been leaking for a month. What can I do?", True, "habitability"),
    ("My heater has been broken for two weeks. What can I do?", True, "habitability"),
    ("There's no hot water again. What can I do?", True, "habitability"),
    ("My apartment has mold. What can I do?", True, "habitability"),
    # The same problems phrased with a landlord or a right. These already worked.
    ("My heater is broken and the landlord won't fix it. What are my rights?",
     True, "habitability, legal vocabulary"),
    ("The landlord hasn't fixed my heat in two weeks. What can I do?",
     True, "habitability, landlord named"),
    # Ordinary legal questions.
    ("Can my landlord charge a late fee if rent is 3 days late?", True, "rules"),
    ("How much notice before my landlord can enter?", True, "rules"),
    ("When do I get my security deposit back after moving out?", True, "rules"),
    # Controls that must NOT reach retrieval, or the router has stopped routing.
    ("hi", False, "control: greeting"),
    ("thanks!", False, "control: thanks"),
    ("ok cool", False, "control: acknowledgement"),
    ("what do you do?", False, "control: about the assistant"),
    ("scan this sample lease for red flags", False, "control: app action"),
]


def probe(repeats: int, model: str) -> dict:
    rows = []
    # Patched rather than passed through, so the shipped call site takes no
    # parameter that exists only for this script. The point of the --model flag is
    # that the comparison in evaluation/README.md stays reproducible: a table
    # claiming nano is worse is worth nothing if nobody can re-measure nano.
    with patch.object(answer, "ROUTER_MODEL", model):
        for question, expected, kind in CASES:
            votes = Counter(needs_retrieval(question) for _ in range(repeats))
            agreed = votes[expected]
            rows.append({
                "question": question,
                "kind": kind,
                "expect_retrieval": expected,
                "correct": f"{agreed}/{repeats}",
                # A router that flaps is a router the user cannot rely on, so a
                # partial score is its own state, not rounded to pass or fail.
                "verdict": ("pass" if agreed == repeats
                            else "fail" if agreed == 0 else "flapping"),
            })

    def count(verdict, *kinds):
        # Exact kinds, not a prefix: "habitability" must not sweep in
        # "habitability, landlord named", which is a case that never broke.
        return sum(1 for r in rows if r["verdict"] == verdict
                   and (not kinds or r["kind"] in kinds))

    def total(*kinds):
        return sum(1 for _, _, kind in CASES if kind in kinds)

    controls = tuple(k for _, _, k in CASES if k.startswith("control"))

    return {
        "router_model": model,
        "repeats": repeats,
        "cases": len(CASES),
        "passed": count("pass"),
        "flapping": count("flapping"),
        "failed": count("fail"),
        # The group that regressed, scored on its own: an aggregate over all 15
        # cases hides it behind ten cases that never broke.
        "bare_habitability_passed": f"{count('pass', 'habitability')}/{total('habitability')}",
        "controls_passed": f"{count('pass', *controls)}/{total(*controls)}",
        "provenance": stamp(),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5,
                        help="samples per case; the router is not deterministic")
    parser.add_argument("--model", default=ROUTER_MODEL,
                        help="router model to probe (default: the shipped one)")
    parser.add_argument("--results", default=str(RESULTS_PATH))
    args = parser.parse_args()

    report = probe(args.repeats, args.model)
    for row in report["results"]:
        mark = {"pass": "OK ", "flapping": "~~ ", "fail": "!! "}[row["verdict"]]
        print(f"{mark}{row['correct']:>5}  {row['kind']:34} {row['question'][:52]}")
    print(f"\n{args.model}: {report['passed']}/{report['cases']} pass · "
          f"{report['flapping']} flapping · {report['failed']} fail · "
          f"bare habitability {report['bare_habitability_passed']} · "
          f"controls {report['controls_passed']}")
    Path(args.results).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Written to {args.results}")


if __name__ == "__main__":
    main()
