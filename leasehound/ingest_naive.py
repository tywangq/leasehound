"""Naive baseline ingestion: fixed-size character chunking, no LLM, no augmentation.

This exists purely as the ablation baseline — the same corpus lands in a parallel
collection ("<state>_reference_naive") so retrieval quality can be compared
against the LLM-chunked pipeline on identical questions.

Usage:
    python -m leasehound.ingest_naive --state wa
"""

import argparse

from leasehound.ingest import create_embeddings, fetch_documents
from leasehound.retrieval import Result

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def naive_split(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def create_chunks(documents: list[dict]) -> list[Result]:
    chunks = []
    for document in documents:
        metadata = {
            "source": document["source"],
            "type": document["type"],
            "state": document["state"],
            "section": document["section"],
            "title": document["title"],
            "url": document["url"],
        }
        for piece in naive_split(document["text"]):
            chunks.append(Result(page_content=piece, metadata=metadata))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="wa")
    args = parser.parse_args()

    documents = fetch_documents(args.state)
    chunks = create_chunks(documents)
    print(f"Created {len(chunks)} naive chunks")
    create_embeddings(chunks, args.state, suffix="_naive")
    print("Naive ingestion complete")


if __name__ == "__main__":
    main()
