"""Scan mode: clause-by-clause red-flag analysis against state statutes.

For each clause: retrieve the governing statute chunks (the clause text itself is
the query), then an LLM judge classifies the clause red / yellow / green — grounded
ONLY in the retrieved extracts, with section citations. Clauses are judged
concurrently — each judgment is an independent retrieval + API call.

Usage:
    python -m leasehound.scan examples/sample_lease.md
    python -m leasehound.scan mylease.pdf --state wa --out report.md
"""

import argparse
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Literal

from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry
from tqdm import tqdm

from leasehound.retrieval import (
    GENERATION_MODEL,
    PipelineConfig,
    Result,
    fetch_unranked,
    wait,
)
from leasehound.upload import load_clauses

STATUTES_PER_CLAUSE = 6
MAX_PARALLEL_SCANS = 8  # bounded so a long lease doesn't trip API rate limits


class ClauseVerdict(BaseModel):
    verdict: Literal["red", "yellow", "green"] = Field(
        description="red = conflicts with the provided statutes (likely prohibited or "
        "unenforceable); yellow = potentially problematic or fact-dependent; "
        "green = no conflict found in the provided extracts"
    )
    citations: list[str] = Field(
        description="RCW section numbers from the provided extracts that ground this "
        "verdict, e.g. ['RCW 59.18.230']. Empty for green verdicts with no relevant law"
    )
    explanation: str = Field(
        description="1-3 sentences in plain language a tenant understands"
    )


def make_judge_prompt(clause: str, chunks: list[Result]) -> str:
    extracts = "\n\n".join(
        f"[{c.metadata.get('section', '?')}]\n{c.page_content}" for c in chunks
    )
    return f"""
You are reviewing ONE clause of a residential lease against Washington State's
Residential Landlord-Tenant Act.

Lease clause:
{clause}

Relevant statute extracts — the ONLY law you may rely on:
{extracts}

Classify the clause. Rules:
- "red" only when the clause conflicts with the provided extracts; cite the section(s).
- Cite section numbers ONLY from the provided extracts, never from memory.
- If the extracts don't address this clause's topic, verdict is "green" and say the
  provided law doesn't address it.
- Plain language; this is legal information, not legal advice.
"""


@retry(wait=wait)
def judge_clause(clause: str, chunks: list[Result]) -> ClauseVerdict:
    messages = [{"role": "user", "content": make_judge_prompt(clause, chunks)}]
    # temperature=0: verdicts are classifications — the same lease should get
    # the same report every scan, and the 7/7 acceptance test should be stable.
    response = completion(
        model=GENERATION_MODEL, messages=messages,
        response_format=ClauseVerdict, temperature=0,
    )
    return ClauseVerdict.model_validate_json(response.choices[0].message.content)


# Negative-space check: protections the law requires that a lease might simply
# omit. The checklist is curated by hand with citations — the LLM only judges
# whether each item is addressed in the lease text, it never invents requirements.
PROTECTION_CHECKLIST = [
    {
        "name": "Deposit withholding terms",
        "requirement": "If any security deposit is collected, the rental agreement must state "
        "the terms and conditions under which any portion of it may be withheld.",
        "citation": "RCW 59.18.260",
    },
    {
        "name": "Move-in condition checklist",
        "requirement": "If any security deposit is collected, a written checklist or statement "
        "describing the unit's condition, signed by both parties, is required.",
        "citation": "RCW 59.18.260",
    },
    {
        "name": "Deposit location disclosure",
        "requirement": "If any security deposit is collected, the tenant must be told the name "
        "and address of the depository where the deposit is held.",
        "citation": "RCW 59.18.270",
    },
    {
        "name": "Fire safety information",
        "requirement": "The landlord must provide written fire safety and protection "
        "information (smoke detection, escape routes) at the start of the tenancy.",
        "citation": "RCW 59.18.060",
    },
    {
        "name": "Mold information",
        "requirement": "The landlord must provide written information about the health "
        "hazards of indoor mold and how to control mold growth.",
        "citation": "RCW 59.18.060",
    },
]


class ProtectionStatus(BaseModel):
    index: int = Field(description="1-based number of the checklist item")
    status: Literal["present", "missing", "not_applicable"] = Field(
        description="present = the lease text addresses it; missing = it applies to this "
        "lease but nothing in the text addresses it; not_applicable = its precondition "
        "doesn't hold (e.g. no security deposit is collected)"
    )
    evidence: str = Field(
        description="Short quote or clause reference when present, otherwise empty"
    )


class ProtectionReport(BaseModel):
    checks: list[ProtectionStatus]


def make_protections_prompt(lease_text: str) -> str:
    items = "\n".join(
        f"{i}. {p['name']} ({p['citation']}): {p['requirement']}"
        for i, p in enumerate(PROTECTION_CHECKLIST, start=1)
    )
    return f"""
You are checking a Washington State residential lease for REQUIRED tenant protections
that may be missing. For each numbered checklist item, decide whether the lease text
addresses it.

Rules:
- "present" only when the lease text actually addresses the item; quote the evidence.
- "missing" when the item applies to this lease but nothing in the text addresses it.
- "not_applicable" when the item's precondition doesn't hold (e.g. no deposit collected).
- Judge ONLY from the lease text below; do not assume documents were provided separately.

Checklist:
{items}

Lease text:
{lease_text[:24000]}

Respond with one status per checklist item, in order.
"""


@retry(wait=wait)
def check_protections(clauses: list[str]) -> list[dict]:
    messages = [{"role": "user", "content": make_protections_prompt("\n\n".join(clauses))}]
    response = completion(
        model=GENERATION_MODEL, messages=messages,
        response_format=ProtectionReport, temperature=0,
    )
    checks = ProtectionReport.model_validate_json(response.choices[0].message.content).checks
    results = []
    for check in checks:
        if 1 <= check.index <= len(PROTECTION_CHECKLIST):
            item = PROTECTION_CHECKLIST[check.index - 1]
            results.append({**item, "status": check.status, "evidence": check.evidence})
    return results


# Documents ABOUT leases (this repo's README, a scan report) read as
# lease-flavored: a bool ("is this a lease?") misfired on them, and — unlike
# the answer router — so did the nano model even with the confusable case as
# its own named category. Whole-document classification needs the generation
# model; it runs once per scan, so the extra cost is negligible.
class DocumentCheck(BaseModel):
    kind: Literal["lease_agreement", "document_about_leases", "other"] = Field(
        description="lease_agreement: the text IS (part of) a residential lease or "
        "rental agreement — contractual terms binding a landlord and a tenant to a "
        "dwelling (parties, premises, rent, obligations, signatures). "
        "document_about_leases: the text discusses, explains, analyzes, or reports on "
        "leases or tenant law without being a contract itself — a legal guide, an "
        "article, software documentation, a scan report of a lease. "
        "other: anything else — a resume, an invoice, another kind of contract, "
        "random text."
    )


@retry(wait=wait)
def looks_like_lease(clauses: list[str]) -> bool:
    """Sanity check before burning a full scan on a document that isn't a lease."""
    message = "Classify the following document.\n\n" + "\n\n".join(clauses)[:6000]
    response = completion(
        model=GENERATION_MODEL, messages=[{"role": "user", "content": message}],
        response_format=DocumentCheck, temperature=0,
    )
    check = DocumentCheck.model_validate_json(response.choices[0].message.content)
    return check.kind == "lease_agreement"


def scan_config(state: str = "wa") -> PipelineConfig:
    return PipelineConfig(
        collection=f"{state}_reference", dual_query=False, grader=False, rerank=False,
        retrieval_k=STATUTES_PER_CLAUSE,
    )


def base_section(citation: str) -> str:
    """Normalize subsection citations like 'RCW 59.18.150(6)' for URL lookup."""
    match = re.match(r"RCW \d+\.\d+\.\d+", citation)
    return match.group(0) if match else citation


def scan_clause(clause: str, index: int, config: PipelineConfig) -> dict:
    chunks = fetch_unranked(clause[:1200], config)
    verdict = judge_clause(clause, chunks)
    url_by_section = {c.metadata.get("section"): c.metadata.get("url") for c in chunks}
    return {
        "index": index,
        "clause": clause,
        "verdict": verdict.verdict,
        "citations": verdict.citations,
        "urls": {s: url_by_section.get(base_section(s), "") for s in verdict.citations},
        "explanation": verdict.explanation,
    }


def scan_clauses(clauses: list[str], config: PipelineConfig) -> Iterator[dict]:
    """Judge every clause concurrently, yielding each finding as its verdict arrives.

    The judgments are independent I/O-bound API calls (one embedding + one
    completion each), so a thread pool cuts the wall-clock scan time roughly
    by the pool size. Findings arrive in completion order — callers sort by
    index. Closing the generator (the user calls off a scan) cancels every
    clause still queued; requests already in flight finish in the background
    and are discarded.
    """
    executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_SCANS)
    try:
        futures = [
            executor.submit(scan_clause, clause, index, config)
            for index, clause in enumerate(clauses, start=1)
        ]
        for future in as_completed(futures):
            yield future.result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def scan_lease(path: str | Path, state: str = "wa") -> tuple[list[dict], list[dict]]:
    config = scan_config(state)
    clauses = load_clauses(path)
    if not clauses:
        raise SystemExit(
            f"No text could be extracted from {path} — "
            "a scanned/photo PDF has no text layer; try a text-based .pdf, .md, or .txt."
        )
    if not looks_like_lease(clauses):
        raise SystemExit(
            f"{path} doesn't look like a residential lease — LeaseHound only scans leases."
        )
    print(f"Scanning {len(clauses)} clauses from {path}")
    findings = list(tqdm(scan_clauses(clauses, config), total=len(clauses),
                         desc="🐕 sniffing clauses"))
    findings.sort(key=lambda f: f["index"])
    print("🐕 Checking required protections…")
    protections = check_protections(clauses)
    return findings, protections


BADGE = {"red": "🚩", "yellow": "⚠️", "green": "✅"}


def render_report(
    findings: list[dict], source: str, state: str, protections: list[dict] | None = None
) -> str:
    counts = {v: sum(1 for f in findings if f["verdict"] == v) for v in ("red", "yellow", "green")}
    missing = [p for p in (protections or []) if p["status"] == "missing"]
    header = [
        f"{BADGE['red']} {counts['red']} red flags",
        f"{BADGE['yellow']} {counts['yellow']} caution",
        f"{BADGE['green']} {counts['green']} clear",
    ]
    if protections is not None:
        header.append(f"🔍 {len(missing)} missing protections")
    lines = [
        "# LeaseHound scan report",
        "",
        f"Document: `{source}` · Jurisdiction: {state.upper()} · Date: {date.today().isoformat()}",
        "",
        "**" + " · ".join(header) + "**",
        "",
        "> Legal information, not legal advice.",
        "",
    ]
    for verdict in ("red", "yellow", "green"):
        matching = [f for f in findings if f["verdict"] == verdict]
        if not matching or verdict == "green":
            continue
        lines.append(f"## {BADGE[verdict]} {verdict.capitalize()}")
        lines.append("")
        for f in matching:
            preview = " ".join(f["clause"].split())[:120]
            lines.append(f"### Clause {f['index']}: {preview}…")
            lines.append("")
            lines.append(f["explanation"])
            for section in f["citations"]:
                url = f["urls"].get(section, "")
                lines.append(f"- {section}" + (f" — {url}" if url else ""))
            lines.append("")
    if missing:
        lines.append("## 🔍 Missing protections")
        lines.append("")
        lines.append("Required by law but not found in this lease — worth asking your landlord:")
        lines.append("")
        for p in missing:
            lines.append(f"- **{p['name']}** — {p['requirement']} ({p['citation']})")
        lines.append("")
    green_indexes = [str(f["index"]) for f in findings if f["verdict"] == "green"]
    if green_indexes:
        lines.append(f"## {BADGE['green']} Clear")
        lines.append("")
        lines.append(f"Clauses {', '.join(green_indexes)} — no conflict found with the retrieved statutes.")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Lease document (.pdf, .md, .txt)")
    parser.add_argument("--state", default="wa")
    parser.add_argument("--out", default="scan_report.md")
    args = parser.parse_args()

    findings, protections = scan_lease(args.file, args.state)
    report = render_report(findings, args.file, args.state, protections)
    Path(args.out).write_text(report, encoding="utf-8")
    counts = {v: sum(1 for f in findings if f["verdict"] == v) for v in ("red", "yellow", "green")}
    missing = sum(1 for p in protections if p["status"] == "missing")
    print(f"\n🚩 {counts['red']} red · ⚠️ {counts['yellow']} yellow · ✅ {counts['green']} green"
          f" · 🔍 {missing} missing protections")
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
