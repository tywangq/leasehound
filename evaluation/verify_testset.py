"""Filter the generated test set: drop questions that aren't actually answerable
from their ground-truth section (cheap generator models drift off-section, which
poisons the ground truth — a retriever that finds the RIGHT section gets scored wrong).

Usage:
    python -m evaluation.verify_testset --state wa
"""

import argparse
import json
from pathlib import Path

from litellm import completion
from pydantic import BaseModel, Field
from tqdm import tqdm

from leasehound.ingest import fetch_documents
from leasehound.retrieval import UTILITY_MODEL, llm_retry

TESTS_PATH = Path(__file__).parent / "tests.jsonl"


class Verdict(BaseModel):
    answerable: bool = Field(
        description="True only if the question can be fully answered from this section's "
        "text alone, and this section is the natural place a lawyer would look"
    )
    reason: str = Field(description="One short sentence")


@llm_retry
def verify(question: str, section_text: str) -> Verdict:
    messages = [
        {
            "role": "user",
            "content": f"Statute section text:\n{section_text}\n\nTest question:\n{question}\n\n"
            "Judge whether this question is fully answerable from THIS section alone. "
            "If the question's topic is canonically governed by a different section "
            "(e.g. late fees, deposits, entry) and only incidentally mentioned here, "
            "answer false.",
        }
    ]
    response = completion(model=UTILITY_MODEL, messages=messages, response_format=Verdict)
    return Verdict.model_validate_json(response.choices[0].message.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="wa")
    args = parser.parse_args()

    sections = {d["section"]: d["text"] for d in fetch_documents(args.state) if d["section"]}
    with open(TESTS_PATH, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f]

    kept, dropped = [], []
    for case in tqdm(cases):
        verdict = verify(case["question"], sections[case["section"]])
        (kept if verdict.answerable else dropped).append((case, verdict.reason))

    with open(TESTS_PATH, "w", encoding="utf-8") as f:
        for case, _ in kept:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Kept {len(kept)}, dropped {len(dropped)}:")
    for case, reason in dropped:
        print(f"  [{case['section']}] {case['question']}  — {reason}")


if __name__ == "__main__":
    main()
