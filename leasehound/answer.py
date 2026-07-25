"""Ask mode: grounded Q&A over the statute corpus.

Retrieval happens in retrieval.py; this module only turns retrieved chunks
into a cited, plain-language answer.
"""

from collections.abc import Iterator
from typing import Literal

from litellm import completion
from pydantic import BaseModel, Field

from leasehound.retrieval import (
    GENERATION_MODEL,
    UTILITY_MODEL,
    PipelineConfig,
    Result,
    fetch_context,
    llm_retry,
)


class Route(BaseModel):
    category: Literal["legal_question", "scan_request", "small_talk"] = Field(
        description="legal_question: asks about rental law, tenant rights, a lease "
        "clause, or what the scan report means — answering requires statutes. "
        "scan_request: asks to scan, check, or upload a lease document ('scan this "
        "sample lease for red flags', 'please scan my lease') — an app action, no "
        "statutes needed. "
        "small_talk: greetings ('hi'), thanks, bare acknowledgements ('ok cool'), "
        "goodbye, or questions about the assistant itself ('what do you do?'). "
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


@llm_retry
def needs_retrieval(question: str, history: list | None = None) -> bool:
    """Router: skip the whole retrieval pipeline for messages that aren't legal questions."""
    message = (
        f"Conversation so far:\n{history or '(none)'}\n\n"
        f"User message:\n{question}\n\n"
        "Classify the user message."
    )
    response = completion(
        model=UTILITY_MODEL,
        messages=[{"role": "user", "content": message}],
        response_format=Route,
    )
    return Route.model_validate_json(response.choices[0].message.content).category == "legal_question"


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


def stream_text(response) -> Iterator[str]:
    """Unwrap a streaming completion into plain text deltas."""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


@llm_retry
def answer_question(
    question: str, history: list | None = None, config: PipelineConfig | None = None
) -> tuple[Iterator[str], list[Result]]:
    """Route the message, then return (token stream, statute chunks the answer cites).

    Retrieval completes before this returns; the caller iterates the stream to
    show the answer as it's generated. Chunks are empty on the chitchat path,
    which is the caller's signal to skip the sources footer.
    """
    config = config or PipelineConfig()
    if not needs_retrieval(question, history):
        messages = (
            [{"role": "system", "content": CHITCHAT_PROMPT}]
            + (history or [])
            + [{"role": "user", "content": question}]
        )
        response = completion(model=GENERATION_MODEL, messages=messages, stream=True)
        return stream_text(response), []
    chunks = fetch_context(question, config, history)
    response = completion(
        model=GENERATION_MODEL,
        messages=make_messages(question, history or [], chunks),
        stream=True,
    )
    return stream_text(response), chunks
