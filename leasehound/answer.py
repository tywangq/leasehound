"""Ask mode: grounded Q&A over the statute corpus.

Retrieval happens in retrieval.py; this module only turns retrieved chunks
into a cited, plain-language answer.

Metered like scan mode is: one log row per answered question, covering the
router call, every retrieval stage, and the answer itself. That last one is why
this arrived late — the answer streams, and a streamed completion reports no
token usage until its final chunk, so there was nothing to hand the meter at
the point the function returns. The fix is stream_options={"include_usage":
True} plus a meter that can book usage without a response object; the log is
written when the stream is drained, not when the call is made.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from litellm import completion
from pydantic import BaseModel, Field

from leasehound.jurisdiction import (
    UNKNOWN_JURISDICTION,
    Jurisdiction,
    jurisdiction_mismatch,
)
from leasehound.metrics import UsageMeter, log_ask
from leasehound.retrieval import (
    GENERATION_MODEL,
    PipelineConfig,
    Result,
    fetch_context,
    llm_retry,
)


class Route(BaseModel):
    # Two answers from the router call, for the price of the one it already makes —
    # the same trade scan's gate makes for the same reason. Ask mode had the gap scan
    # mode just closed: the corpus is Washington's, and a renter in Oregon typing
    # "my landlord kept my whole deposit" got RCW 59.18 with nothing saying it does
    # not govern them. The context line says the hound knows Washington law, which is
    # a disclosure; it is not an answer to a question that named somewhere else.
    jurisdiction: Jurisdiction = Field(
        description="Two-letter code for the US state the USER'S OWN MESSAGE says "
        "they rent in, or whose law they are asking about — 'I'm in Portland, "
        "Oregon' is 'or', 'does California law let my landlord...' is 'ca'. Answer "
        "'unknown' when the message names no state, which is the common case. Judge "
        "ONLY from the user's message: not from the conversation history, not from "
        "any document quoted in it, and not from the statutes under discussion — this "
        "assistant only knows Washington law, so Washington appearing in an earlier "
        "answer is not evidence about where the user lives."
    )
    category: Literal["legal_question", "scan_request", "small_talk"] = Field(
        description="legal_question: asks about rental law, tenant rights, a lease "
        "clause, or what the scan report means. ALSO any message that merely "
        "DESCRIBES a housing problem, or something the landlord did or failed to do "
        "— 'there are cockroaches everywhere', 'my heater has been broken for two "
        "weeks', 'they kept my whole deposit' — even when it names no law, claims no "
        "right, and never mentions the landlord. Answering requires statutes. "
        "scan_request: asks to scan, check, or upload a lease document ('scan this "
        "sample lease for red flags', 'please scan my lease') — an app action, no "
        "statutes needed. "
        "small_talk: greetings ('hi'), thanks, bare acknowledgements ('ok cool'), "
        "goodbye, or questions about the assistant itself ('what do you do?'). "
        "Nothing that describes the user's home, tenancy, or landlord belongs here, "
        "however casually it is phrased. "
        "When unsure, legal_question."
    )


CHITCHAT_PROMPT = """
You are LeaseHound, a friendly assistant for Washington State tenant rights.
The user's message doesn't call for legal research — reply briefly and warmly
(1-2 sentences), and mention what you can help with: scanning a lease for
red flags, or answering questions about renting in Washington.

If the user asked to scan a lease but no lease is attached, tell them to attach
the file with the paperclip (or click the sample-lease example) and send it —
you can't scan from words alone.
"""


# The router runs on the generation model, not the utility one, for the same reason
# scan.py's is-this-a-lease gate does: it is a whole-message classification where
# nano is confidently wrong, and one call per question makes the difference
# negligible (~$0.00006 against a $0.0023 question). Measured, by
# scripts/probe_router.py: nano sent "there are cockroaches everywhere, what can I
# do?" to the chitchat path 4 times out of 5 — mini gets all 15 probe cases right,
# controls included, so this is not just routing everything to retrieval.
#
# Prompt wording alone was not enough. Spelling out that a described problem IS a
# legal question moved the leaking-toilet case from 0/5 to 5/5 and left cockroaches
# at 1/5, which is the shape of a model at its limit rather than an unclear spec.
ROUTER_MODEL = GENERATION_MODEL

# The only law this corpus holds, and therefore the only law any answer is about. Not
# a caller parameter the way scan's `state` is — ask mode has no document to take a
# jurisdiction from and no second corpus to switch to — so it is stated once here
# rather than defaulted in three signatures.
CORPUS_STATE = "wa"


@llm_retry
def route(question: str, history: list | None = None, meter=None) -> Route:
    """Router: skip the whole retrieval pipeline for messages that aren't legal questions.

    Returns the whole route rather than the bool it used to, for the reason
    `classify_document` does the same in scan.py: the call already knows more than
    the caller was being told, and the part it was throwing away — which state the
    asker named — is the part that decides whether the answer is about them.
    """
    message = (
        f"Conversation so far:\n{history or '(none)'}\n\n"
        f"User message:\n{question}\n\n"
        "Classify the user message."
    )
    response = completion(
        model=ROUTER_MODEL,
        messages=[{"role": "user", "content": message}],
        response_format=Route,
    )
    if meter is not None:
        meter.add_completion(response)
    return Route.model_validate_json(response.choices[0].message.content)


def needs_retrieval(question: str, history: list | None = None, meter=None) -> bool:
    """The router's routing half alone, for callers that want nothing else."""
    return route(question, history, meter).category == "legal_question"


SYSTEM_PROMPT = """
You are a tenant-rights assistant for Washington State, answering based on the
Residential Landlord-Tenant Act (RCW 59.18).

Rules:
- Ground every claim in the provided statute extracts and cite the RCW section number
  inline (e.g. "under RCW 59.18.230(2)(i) ..."). Cite inline only — never append a
  sources or "statutes cited" list at the end; the app adds that footer itself.
- If the extracts don't cover the question, say so plainly — never guess at law.
- Plain language; explain legalese when you must quote it.
- You provide legal information, not legal advice; note this when the user's question
  asks what they personally should do.

Statute extracts:
{context}
"""


def make_messages(question: str, history: list, chunks: list[Result]) -> list[dict]:
    context = "\n\n".join(
        f"[{c.metadata.get('section', c.metadata.get('source', ''))}] ({c.metadata.get('url', '')})\n"
        f"{c.page_content}"
        for c in chunks
    )
    return (
        [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
        + (history or [])
        + [{"role": "user", "content": question}]
    )


def stream_text(response, meter=None, model: str = GENERATION_MODEL) -> Iterator[str]:
    """Unwrap a streaming completion into plain text deltas, booking its usage.

    With include_usage the provider sends one extra final chunk that carries the
    token totals and an EMPTY choices list — so the delta lookup below has to be
    guarded, or metering the stream would crash every answer on its last chunk.
    Booked once: some providers attach usage to a content chunk instead of a bare
    one, and counting it twice would inflate the mode's cost by a whole call.
    """
    booked = False
    for chunk in response:
        usage = getattr(chunk, "usage", None)
        if usage is not None and meter is not None and not booked:
            meter.add_streamed_completion(usage, model)
            booked = True
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


@dataclass
class AskResult:
    """What one answered question produced, and what it cost.

    The scan side learned this shape first: a two-tuple grew a third element and
    became ScanResult. Same pressure here — `record` is the metrics row, and it
    cannot be a return value because it does not exist yet when this returns.

    `record` is None until `stream` is exhausted (or closed), because that is when
    the answer call's token usage arrives. Read it after draining the stream.
    """

    stream: Iterator[str]
    chunks: list[Result]
    meter: UsageMeter
    # False means the router sent this down the chitchat path: one call, no
    # retrieval, no sources footer. Empty `chunks` means the same thing, but only
    # by coincidence — retrieval returning nothing would look identical.
    routed: bool
    # The state the question named, or "unknown" — which is the common case and is
    # not a mismatch. Kept separate from the state whose law this corpus holds,
    # because the interesting case is the two disagreeing.
    jurisdiction: str = UNKNOWN_JURISDICTION
    record: dict | None = field(default=None)


# Ask the provider for token totals on the stream. Without this a streamed
# response reports no usage at all, which is the whole reason ask mode went
# unmetered while scan mode was measured to five decimal places.
STREAM_USAGE = {"include_usage": True}


@llm_retry
def answer_question(
    question: str, history: list | None = None, config: PipelineConfig | None = None,
    report_context: bool = False,
) -> AskResult:
    """Route the message, then hand back the token stream and what it cost.

    Retrieval completes before this returns; the caller iterates the stream to
    show the answer as it's generated. Chunks are empty on the chitchat path,
    which is the caller's signal to skip the sources footer.

    The metrics row is written when the caller finishes (or abandons) the stream —
    abandoning still logs, because the retrieval calls were paid for whether or
    not anyone read the answer.

    `report_context` only labels that row: a scan report in the history is most of
    the prompt, so the two kinds of question do not cost the same and averaging
    them together hides it. Passed in rather than sniffed out of `history`, which
    would tie this module to the exact wording a caller wraps the report in.
    """
    config = config or PipelineConfig()
    meter = UsageMeter()
    routing = route(question, history, meter)
    routed = routing.category == "legal_question"
    if routed:
        chunks = fetch_context(question, config, history, meter)
        messages = make_messages(question, history or [], chunks)
    else:
        chunks = []
        messages = (
            [{"role": "system", "content": CHITCHAT_PROMPT}]
            + (history or [])
            + [{"role": "user", "content": question}]
        )
    response = completion(model=GENERATION_MODEL, messages=messages, stream=True,
                          stream_options=STREAM_USAGE)
    result = AskResult(stream=iter(()), chunks=chunks, meter=meter, routed=routed,
                       jurisdiction=routing.jurisdiction)

    def metered() -> Iterator[str]:
        try:
            yield from stream_text(response, meter, GENERATION_MODEL)
        finally:
            result.record = log_ask(
                meter, retrieved=len(chunks), routed=routed,
                with_report=report_context,
                jurisdiction=(routing.jurisdiction
                              if jurisdiction_mismatch(routing.jurisdiction, CORPUS_STATE)
                              else None))

    result.stream = metered()
    return result
