"""Rewrite the retrieval test set in the voice of a renter who never read a law.

Same ground truth, different phrasing. The original test set inherits statute
vocabulary by construction — questions were generated while looking at the
section text — which systematically understates the lexical gap between how
renters talk and how statutes are written. Rewriting every question with zero
legal vocabulary (keeping the same information need and concrete facts) turns
the set into an A/B experiment: any score drop from original to rewritten is
the vocabulary gap, and it lands exactly where the bridging stages
(plain-language augmentation, query rewriting) claim to help.

Usage:
    python -m evaluation.make_adversarial
    python -m evaluation.eval_retrieval --tests tests_adversarial.jsonl --name ...
"""

import json
from pathlib import Path

from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry
from tqdm import tqdm

from leasehound.retrieval import GENERATION_MODEL, wait

TESTS_PATH = Path(__file__).parent / "tests.jsonl"
OUT_PATH = Path(__file__).parent / "tests_adversarial.jsonl"


class Rewrite(BaseModel):
    question: str = Field(
        description="The same information need, reworded: casual spoken English, "
        "contractions fine, zero legal or statutory vocabulary (no 'provision', "
        "'statute', 'prohibited', 'landlord-tenant act', 'notice period', no section "
        "numbers). Keep every concrete fact the original relies on — amounts, day "
        "counts, who did what. One or two sentences."
    )


@retry(wait=wait)
def rewrite(question: str) -> str:
    message = (
        "Rewrite this renter's question as someone who has never read a law and "
        "doesn't know any legal words — texting a friend for advice. Same situation, "
        "same thing they want to know.\n\n" + question
    )
    response = completion(
        model=GENERATION_MODEL, messages=[{"role": "user", "content": message}],
        response_format=Rewrite,
    )
    return Rewrite.model_validate_json(response.choices[0].message.content).question


def main() -> None:
    with open(TESTS_PATH, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f]
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for case in tqdm(cases, desc="rewriting"):
            reworded = {**case, "question": rewrite(case["question"]), "original": case["question"]}
            f.write(json.dumps(reworded, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cases)} rewritten cases to {OUT_PATH}")


if __name__ == "__main__":
    main()
