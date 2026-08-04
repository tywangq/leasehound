"""HTTP contract for the scanner: the pipeline as a service, Gradio as one client.

This exists because the pipeline had exactly one door, and it was a web UI. That
made the whole thing read as a Gradio app with a model behind it, when the Gradio
part is the thinnest layer in the repo — `scan.scan_steps` is the product, and it
already serves a command line and a browser. This is the third caller, and adding
it is what forced the orchestration to stop being duplicated per caller.

Two decisions worth stating.

The response models below are declared here rather than reusing the pipeline's
internal dicts. An HTTP contract that is really "whatever shape our dicts happen
to be today" breaks its callers on refactors that should not concern them, and the
pipeline's `findings` are plain dicts precisely so the pipeline can change them.

The scan and ask routes are behind a shared secret from LEASEHOUND_API_TOKEN, and
when that variable is unset they return 503 rather than running. This is the same
spend reasoning as the rest of the project: Gradio's queue bounds what browser
visitors can start, and it bounds nothing at all about `curl` in a loop, so an
open unauthenticated /v1/scan on a public demo is an unmetered hole straight into
the API key. The hosted demo therefore leaves the variable unset — /docs stays
readable, which is the part worth showing, and the paid routes stay shut. Set it
locally or in CI to exercise them.

    LEASEHOUND_API_TOKEN=dev python -m leasehound.app   # UI at /, API at /v1, docs at /docs
    curl -s -H "X-API-Token: dev" -F file=@examples/sample_lease.md \
        localhost:7860/v1/scan | jq '.summary'
"""

import os
import secrets
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from leasehound.answer import answer_question
from leasehound.scan import (
    CORPUS_SNAPSHOT,
    MAX_CLAUSES,
    NoTextExtracted,
    ScanResult,
    count_verdicts,
    render_report,
    run_scan,
    scan_config,
)
from leasehound.upload import read_document

TOKEN_ENV = "LEASEHOUND_API_TOKEN"
DEFAULT_STATE = "wa"


class Clause(BaseModel):
    index: int
    clause: str
    verdict: Literal["red", "yellow", "green"]
    citations: list[str]
    urls: dict[str, str] = Field(default_factory=dict)
    explanation: str


class Protection(BaseModel):
    name: str
    status: Literal["present", "missing", "not_applicable"]
    requirement: str | None = None
    citation: str | None = None
    evidence: str | None = None


class ScanSummary(BaseModel):
    source: str
    state: str
    split_mode: str
    clauses_total: int = Field(description="Clauses the document split into, before the cap")
    clauses_judged: int = Field(description=f"Clauses actually judged (cap: {MAX_CLAUSES})")
    partial: bool = Field(description="True when the cap left clauses unjudged")
    gate_flagged: bool = Field(
        description="The document did not read as a residential lease. Advisory: the "
                    "scan still ran, and the verdicts are less trustworthy.")
    refused: bool = Field(
        default=False,
        description="The document read as unrelated to renting, so no clause was "
                    "judged: `red`/`yellow`/`green` are all zero because nothing was "
                    "looked at, NOT because the lease is clean. Re-send with "
                    "`scan_anyway=true` to override.")
    red: int
    yellow: int
    green: int
    missing_protections: int
    cost_usd: float | None = Field(description="What this request spent")
    seconds: float | None = None


class ScanResponse(BaseModel):
    summary: ScanSummary
    findings: list[Clause]
    protections: list[Protection]
    report_markdown: str = Field(description="The same report the web UI pins")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    report_markdown: str | None = Field(
        default=None, max_length=200_000,
        description="A prior scan report, so follow-up questions can be about this lease")


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(description="Statute sections the answer was grounded in")
    retrieved: bool = Field(
        description="False when the router classified the message as chitchat or a scan "
                    "request, so no statutes were fetched and `sources` is empty by design")
    llm_calls: int | None = Field(
        default=None,
        description="Calls this question made: the router, the retrieval stages, the answer")
    cost_usd: float | None = Field(default=None, description="What this request spent")
    seconds: float | None = None


class Health(BaseModel):
    status: Literal["ok"]
    corpus_snapshot: str
    collection: str
    paid_routes_enabled: bool


def require_token(x_api_token: Annotated[str | None, Header()] = None) -> None:
    """Gate the routes that spend money. See the module docstring for why."""
    expected = os.environ.get(TOKEN_ENV)
    if not expected:
        raise HTTPException(
            503,
            f"The scanning routes are disabled because {TOKEN_ENV} is not set. This is "
            "deliberate on the public demo: nothing here rate-limits an unauthenticated "
            "caller, so an open scan endpoint is an unmetered hole into the API key. Use "
            "the web UI, or set the variable and send it as X-API-Token.",
        )
    # compare_digest so a wrong token cannot be found one character at a time.
    if not secrets.compare_digest(x_api_token or "", expected):
        raise HTTPException(401, "Bad or missing X-API-Token.")


def scan_response(result: ScanResult, source: str, state: str) -> ScanResponse:
    counts = count_verdicts(result.findings)
    record = result.record or {}
    return ScanResponse(
        summary=ScanSummary(
            source=source, state=state, split_mode=result.split_mode,
            clauses_total=result.clauses_total, clauses_judged=result.clauses_judged,
            partial=result.partial, gate_flagged=result.gate_flagged,
            refused=result.refused,
            red=counts["red"], yellow=counts["yellow"], green=counts["green"],
            missing_protections=sum(
                1 for p in result.protections if p["status"] == "missing"),
            cost_usd=record.get("cost_usd"), seconds=record.get("seconds"),
        ),
        findings=[Clause(**f) for f in result.findings],
        protections=[Protection(**p) for p in result.protections],
        report_markdown=render_report(
            result.findings, source, state, result.protections,
            result.gate_flagged, result.clauses_total, result.refused),
    )


api = FastAPI(
    title="LeaseHound",
    version="0.1.0",
    summary="Scan a residential lease for clauses that are void under state law.",
    description=__doc__,
)


@api.get("/v1/health", response_model=Health, tags=["meta"])
def health() -> Health:
    """Liveness, plus the two facts that decide whether a verdict is current."""
    return Health(
        status="ok",
        corpus_snapshot=CORPUS_SNAPSHOT,
        collection=scan_config(DEFAULT_STATE).collection,
        paid_routes_enabled=bool(os.environ.get(TOKEN_ENV)),
    )


@api.post("/v1/scan", response_model=ScanResponse, tags=["scan"],
          dependencies=[Depends(require_token)])
async def scan(file: Annotated[UploadFile, File(description=".pdf, .md or .txt")],
               state: str = DEFAULT_STATE,
               scan_anyway: bool = False) -> ScanResponse:
    """Scan an uploaded lease. Over the clause cap the scan is partial, not refused —
    `summary.partial` says so and `clauses_judged` says how far it got. A document
    that reads as unrelated to renting IS refused: `summary.refused` says so, and
    `scan_anyway=true` overrides it."""
    name = Path(file.filename or "upload").name
    # Written to a temp file and deleted immediately, for two reasons: pypdf reads a
    # file, and this is the same extraction path the CLI and the UI use rather than a
    # second one that could disagree about what a document says.
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix, delete=False) as handle:
        handle.write(await file.read())
        upload = Path(handle.name)
    try:
        text = read_document(upload)
    except Exception as broken:
        raise HTTPException(400, f"Could not read {name}: {broken}") from broken
    finally:
        upload.unlink(missing_ok=True)

    try:
        result = run_scan(text, name, state=state, scan_anyway=scan_anyway)
    except NoTextExtracted as empty:
        raise HTTPException(
            422, f"No text layer in {name} — a scanned or photo PDF has no extractable "
                 "text. Try a text-based .pdf, .md or .txt.") from empty
    return scan_response(result, name, state)


@api.post("/v1/ask", response_model=AskResponse, tags=["ask"],
          dependencies=[Depends(require_token)])
def ask(request: AskRequest) -> AskResponse:
    """Ask about Washington tenant law, optionally with a scan report as context.

    The pipeline streams tokens; this collects them, because a JSON response body
    is not a stream and pretending otherwise would just buffer in a worse place.
    """
    history = ([{"role": "assistant", "content": request.report_markdown}]
               if request.report_markdown else [])
    answered = answer_question(request.question, history,
                               report_context=bool(request.report_markdown))
    # Draining the stream is also what completes the metering, so `record` is only
    # populated after this line — same ordering as /v1/scan, where run_scan drains
    # the step iterator before the record exists.
    answer = "".join(answered.stream)
    record = answered.record or {}
    return AskResponse(
        answer=answer,
        sources=sorted({c.metadata.get("section", "") for c in answered.chunks} - {""}),
        retrieved=answered.routed,
        llm_calls=record.get("llm_calls"),
        cost_usd=record.get("cost_usd"),
        seconds=record.get("seconds"),
    )
