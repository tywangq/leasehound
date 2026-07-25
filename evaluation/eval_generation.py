"""Generation-layer evaluation: is the final answer correct, grounded, and cited?

Reuses the retrieval test set: every question was generated from a known statute
section, so that section's text is the authoritative reference for judging the
answer. An LLM judge (temperature=0) grades each answer twice —

- correctness  against the ground-truth section text (consistent / contradicts /
  declined, where declined = the answer says the law provided doesn't cover it)
- groundedness against the retrieved extracts the generator actually saw
  (any legal rule asserted that appears in neither is an unsupported claim)

— while citation accuracy is checked mechanically (no LLM): does the answer cite
the ground-truth section? Declines are split by whether retrieval had actually
missed (honest) or the section was right there (overcautious).

Judge caveat: the judge is the same model family as the generator, but it grades
against reference text rather than its own taste, which limits self-preference.

Usage:
    python -m evaluation.eval_generation --name full-n82
    python -m evaluation.eval_generation --name naive-crag-n82 \
        --collection wa_reference_naive --no-dual --no-rerank
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry
from tqdm import tqdm

from leasehound.answer import make_messages
from leasehound.ingest import fetch_documents
from leasehound.retrieval import GENERATION_MODEL, PipelineConfig, fetch_context, wait
from leasehound.scan import base_section

TESTS_PATH = Path(__file__).parent / "tests.jsonl"
RESULTS_PATH = Path(__file__).parent / "generation_results.jsonl"
MAX_PARALLEL = 8
RCW_RE = re.compile(r"RCW \d+\.\d+\.\d+")


class AnswerJudgment(BaseModel):
    verdict: Literal["consistent", "contradicts", "declined"] = Field(
        description="consistent: the answer's legal substance agrees with the reference "
        "statute text. contradicts: the answer asserts something the reference text "
        "contradicts. declined: the answer says the provided law doesn't cover the "
        "question rather than answering it."
    )
    unsupported_claim: bool = Field(
        description="True if the answer asserts a specific legal rule that appears in "
        "NEITHER the retrieved extracts NOR the reference text. Hedged general advice "
        "('consider consulting a lawyer') is not a legal rule."
    )


def make_judge_prompt(question: str, reference: str, extracts: str, answer: str) -> str:
    return f"""
You are grading one answer from a tenant-law assistant.

Tenant's question:
{question}

Reference statute text (the authoritative ground truth for this question):
{reference[:8000]}

Retrieved extracts (everything the assistant was shown when answering):
{extracts[:8000]}

Assistant's answer:
{answer}

Grade the answer. Judge legal substance, not style or completeness of citations.
"""


@retry(wait=wait)
def judge_answer(question: str, reference: str, extracts: str, answer: str) -> AnswerJudgment:
    response = completion(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": make_judge_prompt(question, reference, extracts, answer)}],
        response_format=AnswerJudgment, temperature=0,
    )
    return AnswerJudgment.model_validate_json(response.choices[0].message.content)


@retry(wait=wait)
def generate_answer(question: str, config: PipelineConfig) -> tuple[str, list]:
    chunks = fetch_context(question, config)
    response = completion(model=GENERATION_MODEL, messages=make_messages(question, [], chunks))
    return response.choices[0].message.content, chunks


def evaluate_case(case: dict, config: PipelineConfig, sections: dict) -> dict:
    answer, chunks = generate_answer(case["question"], config)
    reference = sections[case["section"]]
    extracts = "\n\n".join(c.page_content for c in chunks)
    judgment = judge_answer(case["question"], reference, extracts, answer)
    retrieved_sections = {c.metadata.get("section") for c in chunks}
    return {
        "question": case["question"],
        "section": case["section"],
        "verdict": judgment.verdict,
        "unsupported_claim": judgment.unsupported_claim,
        "cites_gt": case["section"] in {base_section(c) for c in RCW_RE.findall(answer)},
        "retrieval_had_gt": case["section"] in retrieved_sections,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--collection", default="wa_reference")
    parser.add_argument("--no-dual", dest="dual", action="store_false")
    parser.add_argument("--no-grader", dest="grader", action="store_false")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false")
    parser.add_argument("--limit", type=int, help="Only run the first N questions (smoke test)")
    parser.add_argument("--tests", default="tests.jsonl", help="Test set file in evaluation/")
    args = parser.parse_args()
    config = PipelineConfig(collection=args.collection, dual_query=args.dual,
                            grader=args.grader, rerank=args.rerank)

    with open(Path(__file__).parent / args.tests, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f]
    if args.limit:
        cases = cases[: args.limit]
    sections = {d["section"]: d["text"] for d in fetch_documents("wa") if d["section"]}

    rows = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = [pool.submit(evaluate_case, case, config, sections) for case in cases]
        for future in tqdm(as_completed(futures), total=len(futures), desc=args.name):
            rows.append(future.result())

    n = len(rows)
    declined = [r for r in rows if r["verdict"] == "declined"]
    summary = {
        "name": args.name,
        "collection": config.collection,
        "n": n,
        "consistent": round(sum(r["verdict"] == "consistent" for r in rows) / n, 4),
        "contradicts": round(sum(r["verdict"] == "contradicts" for r in rows) / n, 4),
        "declined": round(len(declined) / n, 4),
        "declined_despite_retrieval": sum(r["retrieval_had_gt"] for r in declined),
        "grounded": round(sum(not r["unsupported_claim"] for r in rows) / n, 4),
        "cites_gt": round(sum(r["cites_gt"] for r in rows) / n, 4),
    }
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))
    for r in rows:
        if r["verdict"] != "consistent" or r["unsupported_claim"]:
            print(f"  [{r['verdict']}{' +unsupported' if r['unsupported_claim'] else ''}] "
                  f"({r['section']}) {r['question'][:70]}")


if __name__ == "__main__":
    main()
