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


def fetch_unranked(query: str, config: PipelineConfig, meter=None) -> list[Result]:
    collection = _get_collection(config.collection)
    response = openai.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    if meter is not None:
        meter.add_embedding(response, EMBEDDING_MODEL)
    embedding = response.data[0].embedding
    results = collection.query(query_embeddings=[embedding], n_results=config.retrieval_k)
    return [
        Result(page_content=doc, metadata=meta)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


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
