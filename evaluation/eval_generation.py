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
from tqdm import tqdm

from evaluation.provenance import stamp
from leasehound.answer import make_messages
from leasehound.ingest import fetch_documents
from leasehound.retrieval import (
    GENERATION_MODEL,
    PipelineConfig,
    fetch_context,
    llm_retry,
)
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


@llm_retry
def judge_answer(question: str, reference: str, extracts: str, answer: str) -> AnswerJudgment:
    response = completion(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": make_judge_prompt(question, reference, extracts, answer)}],
        response_format=AnswerJudgment, temperature=0,
    )
    return AnswerJudgment.model_validate_json(response.choices[0].message.content)


# The closed-book config: the shipped system prompt with the retrieval rules removed
# and no statute text attached. Deliberately generous to the baseline — same persona,
# same instruction to cite section numbers inline — because the question is what
# retrieval adds over a model that is trying, not over a model told not to.
#
# Scan mode has had this comparison since July (eval_baseline.py: 3/14 citations
# correct closed-book against 18/18 with retrieval). Ask mode did not, which left the
# README's central claim — "the law shouldn't come from parametric memory" — measured
# on only one of the two modes it is made about. It is also the claim a reader is most
# likely to doubt about ask mode specifically, since "a chatbot already answers this"
# is the obvious objection to a tenant-law Q&A.
CLOSED_BOOK_PROMPT = """
You are a tenant-rights assistant for Washington State, answering questions about the
Residential Landlord-Tenant Act (RCW 59.18).

Answer the tenant's question in plain language, and cite the RCW section number inline
(e.g. "under RCW 59.18.230(2)(i) ..."). You provide legal information, not legal advice.
"""


@llm_retry
def generate_answer(question: str, config: PipelineConfig,
                    closed_book: bool = False) -> tuple[str, list]:
    if closed_book:
        response = completion(model=GENERATION_MODEL, messages=[
            {"role": "system", "content": CLOSED_BOOK_PROMPT},
            {"role": "user", "content": question}])
        return response.choices[0].message.content, []
    chunks = fetch_context(question, config)
    response = completion(model=GENERATION_MODEL, messages=make_messages(question, [], chunks))
    return response.choices[0].message.content, chunks


def evaluate_case(case: dict, config: PipelineConfig, sections: dict,
                  closed_book: bool = False) -> dict:
    answer, chunks = generate_answer(case["question"], config, closed_book)
    reference = sections[case["section"]]
    extracts = "\n\n".join(c.page_content for c in chunks)
    judgment = judge_answer(case["question"], reference, extracts, answer)
    retrieved_sections = {c.metadata.get("section") for c in chunks}
    cited = {base_section(c) for c in RCW_RE.findall(answer)}
    return {
        "question": case["question"],
        "section": case["section"],
        "verdict": judgment.verdict,
        "unsupported_claim": judgment.unsupported_claim,
        "cites_gt": case["section"] in cited,
        # Sections cited that this corpus does not contain. Named for what it measures
        # rather than "invented": a citation outside the corpus is not necessarily
        # fake — RCW 19.86, the Consumer Protection Act, is real and relevant and not
        # in here — but it is unverifiable by this system, which is the property that
        # matters when the claim being tested is that citations should be checkable.
        "cites_outside_corpus": sorted(c for c in cited if c not in sections),
        "retrieval_had_gt": case["section"] in retrieved_sections,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--collection", default="wa_reference")
    parser.add_argument("--no-dual", dest="dual", action="store_false")
    parser.add_argument("--no-grader", dest="grader", action="store_false")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false")
    parser.add_argument("--bm25", action="store_true",
                        help="merge a BM25 lexical channel into retrieval (hybrid)")
    parser.add_argument("--closed-book", action="store_true",
                        help="no retrieval at all: the model answers from memory, which "
                             "is what pasting the question into a chat window does")
    parser.add_argument("--limit", type=int, help="Only run the first N questions (smoke test)")
    parser.add_argument("--tests", default="tests.jsonl", help="Test set file in evaluation/")
    args = parser.parse_args()
    config = PipelineConfig(collection=args.collection, dual_query=args.dual,
                            grader=args.grader, rerank=args.rerank, bm25=args.bm25)

    with open(Path(__file__).parent / args.tests, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f]
    if args.limit:
        cases = cases[: args.limit]
    sections = {d["section"]: d["text"] for d in fetch_documents("wa") if d["section"]}

    rows = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = [pool.submit(evaluate_case, case, config, sections, args.closed_book)
                   for case in cases]
        for future in tqdm(as_completed(futures), total=len(futures), desc=args.name):
            rows.append(future.result())

    n = len(rows)
    declined = [r for r in rows if r["verdict"] == "declined"]
    summary = {
        "name": args.name,
        "collection": "(none — closed book)" if args.closed_book else config.collection,
        "n": n,
        "consistent": round(sum(r["verdict"] == "consistent" for r in rows) / n, 4),
        "contradicts": round(sum(r["verdict"] == "contradicts" for r in rows) / n, 4),
        "declined": round(len(declined) / n, 4),
        "declined_despite_retrieval": sum(r["retrieval_had_gt"] for r in declined),
        # Closed-book, the judge sees no extracts, so this collapses to "asserts no
        # rule the ground-truth statute contains" — a stricter question than the one it
        # answers for a retrieval config, and not the same number. Renamed rather than
        # reported under the same key, because a reader comparing rows would otherwise
        # be comparing two different measurements.
        ("unsupported_free" if args.closed_book else "grounded"):
            round(sum(not r["unsupported_claim"] for r in rows) / n, 4),
        "cites_gt": round(sum(r["cites_gt"] for r in rows) / n, 4),
        "answers_citing_outside_corpus": sum(bool(r["cites_outside_corpus"]) for r in rows),
        "sections_cited_outside_corpus": sorted(
            {s for r in rows for s in r["cites_outside_corpus"]}),
        **stamp(),
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
