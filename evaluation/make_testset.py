"""Generate the retrieval test set: realistic tenant questions with statute-level ground truth.

Each test case is a plain-language question generated FROM one statute section's text,
so the ground truth (which section should be retrieved) is known by construction.
Sections are picked by substance (longest N) plus a must-include list of high-value
sections (prohibited provisions, deposits, entry, repairs).

Usage:
    python -m evaluation.make_testset --state wa --sections 25 --per-section 2
"""

import argparse
import json
from pathlib import Path

from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry
from tqdm import tqdm

from leasehound.ingest import fetch_documents
from leasehound.retrieval import UTILITY_MODEL, wait

TESTS_PATH = Path(__file__).parent / "tests.jsonl"

MUST_INCLUDE = [
    "RCW 59.18.230",  # prohibited lease provisions — the red-flag ground truth
    "RCW 59.18.150",  # landlord entry
    "RCW 59.18.060",  # landlord duties (repairs)
    "RCW 59.18.280",  # deposit refund
]


class TestQuestions(BaseModel):
    questions: list[str] = Field(description="Plain-language tenant questions")


def make_prompt(document: dict, per_section: int) -> str:
    return f"""
You write test questions for a tenant-rights assistant.

Below is one section of the Washington Residential Landlord-Tenant Act:
{document["section"]} — {document["title"]}

{document["text"]}

Write exactly {per_section} questions that a REAL TENANT might type into a chat box,
where the answer is contained in this section's text.

Rules:
- Everyday renter language: "late fee", "deposit", "my landlord", "can they..." —
  NO statute numbers, NO legal jargon, NO mention of "this section".
- Each question must be answerable from this section alone.
- Vary the angle: one about what's allowed/prohibited, one about a concrete scenario.
"""


@retry(wait=wait)
def questions_for(document: dict, per_section: int) -> list[str]:
    messages = [{"role": "user", "content": make_prompt(document, per_section)}]
    response = completion(model=UTILITY_MODEL, messages=messages, response_format=TestQuestions)
    return TestQuestions.model_validate_json(response.choices[0].message.content).questions


def pick_sections(documents: list[dict], n: int) -> list[dict]:
    by_section = {d["section"]: d for d in documents if d["section"]}
    picked = [by_section[s] for s in MUST_INCLUDE if s in by_section]
    rest = sorted(
        (d for d in by_section.values() if d["section"] not in MUST_INCLUDE),
        key=lambda d: len(d["text"]),
        reverse=True,
    )
    return (picked + rest)[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="wa")
    parser.add_argument("--sections", type=int, default=25)
    parser.add_argument("--per-section", type=int, default=2)
    args = parser.parse_args()

    documents = pick_sections(fetch_documents(args.state), args.sections)
    cases = []
    for document in tqdm(documents):
        for question in questions_for(document, args.per_section):
            cases.append(
                {
                    "question": question,
                    "section": document["section"],
                    "title": document["title"],
                    "state": args.state,
                }
            )

    with open(TESTS_PATH, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cases)} test cases to {TESTS_PATH}")


if __name__ == "__main__":
    main()
