"""Shared retrieval layer: fetch statute chunks from Chroma, configurably refined.

Both query modes sit on top of this module — answer.py (ask mode) and scan.py
(scan mode) — as does the ablation suite. Model names and shared clients live
here so there is exactly one place to swap them.

Every refinement stage is a switch so the ablation study can turn them on one
at a time:

    PipelineConfig(collection="wa_reference_naive", dual_query=False,
                   grader=False, rerank=False)          # ablation row 1 (baseline)
    PipelineConfig()                                    # full pipeline (row 6)
"""

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from leasehound import bm25

load_dotenv(override=True)

# Any litellm model id works — e.g. "ollama/llama3.1" for fully local inference.
UTILITY_MODEL = os.getenv("LEASEHOUND_UTILITY_MODEL", "openai/gpt-4.1-nano")  # rewrite/grade/rerank/chunking
GENERATION_MODEL = os.getenv("LEASEHOUND_GENERATION_MODEL", "openai/gpt-4.1-mini")  # answers and verdicts
EMBEDDING_MODEL = os.getenv("LEASEHOUND_EMBEDDING_MODEL", "text-embedding-3-large")
DB_PATH = str(Path(__file__).parent.parent / "vector_db")

# One retry policy for every LLM/API call. Bounded — without a stop, a
# persistent failure (revoked key, exhausted quota) retries forever and the
# UI event waiting on it hangs. reraise hands callers the real error instead
# of tenacity's RetryError wrapper.
llm_retry = retry(
    wait=wait_exponential(multiplier=1, min=10, max=240),
    stop=stop_after_attempt(5),
    reraise=True,
)
openai = OpenAI()

# The Chroma client is created lazily: ingest worker processes import this
# module for Result/constants and must not each open the database. The lock
# keeps parallel scan threads from racing to create it on first use.
_chroma = None
_chroma_lock = threading.Lock()


def _get_collection(name: str):
    global _chroma
    with _chroma_lock:
        if _chroma is None:
            _chroma = PersistentClient(path=DB_PATH)
    return _chroma.get_collection(name)


@dataclass
class PipelineConfig:
    collection: str = "wa_reference"
    dual_query: bool = True
    grader: bool = True
    rerank: bool = True
    # Lexical channel merged into the dense results (see bm25.py). Off by
    # default so every existing ablation row keeps meaning what it measured.
    bm25: bool = False
    # Hand the judge every chunk of a retrieved section, not just the chunk that
    # matched. Off by default until measured (see eval_scan_retrieval.py).
    section_completion: bool = False
    retrieval_k: int = 20
    final_k: int = 10


class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(description="Chunk ids from most to least relevant")


class Sufficiency(BaseModel):
    sufficient: bool = Field(description="Whether the excerpts can fully answer the question")


@llm_retry
def rewrite_query(question: str, history: list | None = None, angle: str = "specific") -> str:
    instruction = {
        "specific": "Rewrite as a short, specific search query most likely to surface "
        "relevant statute text. Focus on the concrete details.",
        "statutory": "Rewrite using formal landlord-tenant statutory vocabulary "
        "(e.g. 'rental agreement', 'prohibited provision', 'security deposit', "
        "'notice to enter') so it matches legal language.",
    }[angle]
    message = f"""
You help retrieve sections of a state landlord-tenant law to answer a tenant's question.

Conversation so far:
{history or "(none)"}

Tenant's question:
{question}

{instruction}
Respond ONLY with the query text, nothing else.
"""
    response = completion(model=UTILITY_MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


# How deep each channel looks before the RRF merge, when the merge would
# otherwise see less than this. Measured, not guessed: for the exculpation
# clause that motivated hybrid retrieval, the governing section sits at rank 15
# in the dense channel and rank 8 in the lexical one. Scan mode shows the judge
# six chunks — so merging two six-deep lists would discard, before the merge,
# the very chunk this stage exists to recover. Looking deeper is free: the
# embedding is one API call regardless of how many rows Chroma returns, BM25 is
# local, and the merge is truncated back to retrieval_k for the prompt.
HYBRID_CANDIDATES = 20


def candidate_k(config: PipelineConfig) -> int:
    return max(config.retrieval_k, HYBRID_CANDIDATES)


def bm25_search(query: str, config: PipelineConfig, k: int | None = None) -> list[Result]:
    """Lexical half of hybrid retrieval. No API call, so it costs latency only."""
    def load():
        stored = _get_collection(config.collection).get(
            include=["documents", "metadatas"]
        )
        return stored["documents"], stored["metadatas"]

    index = bm25.get_index(config.collection, load)
    return [
        Result(page_content=doc, metadata=meta)
        for doc, meta in index.search(query, k if k is not None else config.retrieval_k)
    ]


def fetch_unranked(query: str, config: PipelineConfig, meter=None) -> list[Result]:
    collection = _get_collection(config.collection)
    response = openai.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    if meter is not None:
        meter.add_embedding(response, EMBEDDING_MODEL)
    embedding = response.data[0].embedding
    depth = candidate_k(config) if config.bm25 else config.retrieval_k
    results = collection.query(query_embeddings=[embedding], n_results=depth)
    dense = [
        Result(page_content=doc, metadata=meta)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
    if not config.bm25:
        return complete_sections(dense, config) if config.section_completion else dense
    lexical = bm25_search(query, config, k=depth)
    # Same RRF merge the dual-query stage uses, then truncated back to
    # retrieval_k — hybrid costs no extra prompt tokens and no extra API call.
    merged = merge_chunks(dense, lexical)[: config.retrieval_k]
    return complete_sections(merged, config) if config.section_completion else merged


# A section averages ~3.7 chunks, so completing every retrieved section would
# multiply the judge's prompt. Only the best few are expanded.
SECTIONS_TO_COMPLETE = 3


def chunk_order(chunk_id: str) -> int:
    """Ingest writes ids as `<state>-<n>` in document order, and the metadata
    carries no chunk index, so the id suffix is what puts a section back together
    in reading order."""
    tail = chunk_id.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def complete_sections(chunks: list[Result], config: PipelineConfig) -> list[Result]:
    """Replace the top sections' single hits with all of that section's chunks.

    Retrieving the right *section* is not the same as retrieving the rule: a
    section split across four chunks can surface the one chunk that does not
    contain the prohibition, and the judge — which may only cite what it is
    given — then correctly returns green for what it saw. That was the measured
    cause of the scanner's one missed violation. Handing over the whole section
    costs prompt tokens and no extra API call.
    """
    collection = _get_collection(config.collection)
    ordered: list[str] = []
    for chunk in chunks:
        section = chunk.metadata.get("section")
        if section and section not in ordered:
            ordered.append(section)
    completing = ordered[:SECTIONS_TO_COMPLETE]

    expanded: list[Result] = []
    for section in completing:
        got = collection.get(where={"section": section})
        siblings = sorted(zip(got["ids"], got["documents"], got["metadatas"]),
                          key=lambda row: chunk_order(row[0]))
        expanded += [Result(page_content=doc, metadata=meta) for _, doc, meta in siblings]
    # Sections past the cap keep whichever chunk retrieval actually chose.
    kept = [c for c in chunks if c.metadata.get("section") not in completing]
    return expanded + kept


def merge_chunks(*chunk_lists: list[Result], rrf_k: int = 60) -> list[Result]:
    """Reciprocal-rank-fusion merge: a chunk ranked high in ANY list surfaces early.

    A plain append would strand the second list's chunks below top-k, silently
    turning dual-query retrieval into a no-op whenever no reranker follows —
    which is exactly what the ablation study caught.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, Result] = {}
    for chunks in chunk_lists:
        for rank, chunk in enumerate(chunks):
            key = chunk.page_content
            scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank + 1)
            first_seen.setdefault(key, chunk)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [first_seen[key] for key in ordered]


@llm_retry
def grade_context(question: str, chunks: list[Result]) -> bool:
    """CRAG-style self-grading: do the retrieved chunks actually cover the question?"""
    excerpts = "\n\n".join(chunk.page_content[:500] for chunk in chunks[:8])
    messages = [
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nRetrieved statute excerpts:\n{excerpts}\n\n"
            "Do these excerpts contain the information needed to answer the question?",
        }
    ]
    response = completion(model=UTILITY_MODEL, messages=messages, response_format=Sufficiency)
    return Sufficiency.model_validate_json(response.choices[0].message.content).sufficient


@llm_retry
def rerank(question: str, chunks: list[Result]) -> list[Result]:
    system_prompt = (
        "You are a document re-ranker. Rank the provided chunks by relevance to the "
        "question, most relevant first. Reply only with the ranked chunk ids, "
        "including every id you were given."
    )
    user_prompt = f"Question:\n\n{question}\n\nChunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID {index + 1}:\n\n{chunk.page_content}\n\n"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=UTILITY_MODEL, messages=messages, response_format=RankOrder)
    order = RankOrder.model_validate_json(response.choices[0].message.content).order
    return [chunks[i - 1] for i in order if 1 <= i <= len(chunks)]


def fetch_context(question: str, config: PipelineConfig, history: list | None = None) -> list[Result]:
    chunks = fetch_unranked(question, config)
    if config.dual_query:
        rewritten = rewrite_query(question, history)
        chunks = merge_chunks(chunks, fetch_unranked(rewritten, config))
    if config.grader and not grade_context(question, chunks):
        statutory = rewrite_query(question, history, angle="statutory")
        chunks = merge_chunks(chunks, fetch_unranked(statutory, config))
    if config.rerank:
        chunks = rerank(question, chunks)
    return chunks[: config.final_k]
