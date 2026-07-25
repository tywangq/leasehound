"""Ingest the Layer-1 reference corpus into a per-state Chroma collection.

Corpus layout:  corpus/<state>/<doc_type>/**/*.md   (see scripts/fetch_corpus.py)
Collections:    one per state, named "<state>_reference" — states are first-class
                so new jurisdictions are added by dropping a corpus folder and re-running.

Usage:
    python -m leasehound.ingest --state wa
    python -m leasehound.ingest --state wa --limit 3 --workers 1   # cheap smoke test
"""

import argparse
import re
from multiprocessing import Pool
from pathlib import Path

from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry
from tqdm import tqdm

from leasehound.retrieval import DB_PATH, EMBEDDING_MODEL, UTILITY_MODEL, Result, openai, wait

EMBED_BATCH_SIZE = 100
AVERAGE_CHUNK_SIZE = 400  # legal sections are dense; chunks lean larger than prose
WORKERS = 3

REPO_ROOT = Path(__file__).parent.parent
CORPUS_PATH = REPO_ROOT / "corpus"

SECTION_RE = re.compile(r"^# (RCW [\d.]+) — (.+)$", re.M)
SOURCE_RE = re.compile(r"^Source: (\S+)$", re.M)


class Chunk(BaseModel):
    headline: str = Field(
        description="Brief heading for this chunk. MUST begin with the statute citation "
        "(e.g. 'RCW 59.18.230'), followed by a few plain-English words on what it covers"
    )
    summary: str = Field(
        description="2-3 sentences in everyday renter vocabulary (deposit, late fee, repairs, "
        "eviction, mold...) summarizing what this chunk means for a tenant. This bridges "
        "colloquial questions to statutory language, so avoid legalese here"
    )
    original_text: str = Field(
        description="The original statute text of this chunk, exactly as provided, unchanged"
    )

    def as_result(self, document, plain: bool = False):
        metadata = {
            "source": document["source"],
            "type": document["type"],
            "state": document["state"],
            "section": document["section"],
            "title": document["title"],
            "url": document["url"],
        }
        content = (
            self.original_text
            if plain
            else self.headline + "\n\n" + self.summary + "\n\n" + self.original_text
        )
        return Result(page_content=content, metadata=metadata)


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents(state: str) -> list[dict]:
    state_path = CORPUS_PATH / state
    if not state_path.is_dir():
        raise SystemExit(f"No corpus for state '{state}' at {state_path}")

    documents = []
    for folder in sorted(state_path.iterdir()):
        if not folder.is_dir():
            continue
        for file in sorted(folder.rglob("*.md")):
            text = file.read_text(encoding="utf-8")
            section_match = SECTION_RE.search(text)
            source_match = SOURCE_RE.search(text)
            documents.append(
                {
                    "type": folder.name,
                    "state": state,
                    "source": file.relative_to(REPO_ROOT).as_posix(),
                    "section": section_match.group(1) if section_match else "",
                    "title": section_match.group(2) if section_match else file.stem,
                    "url": source_match.group(1) if source_match else "",
                    "text": text,
                }
            )

    print(f"Loaded {len(documents)} documents for state '{state}'")
    return documents


def make_prompt(document: dict) -> str:
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You split one section of a state statute into overlapping chunks for a legal knowledge base.

This section is {document["section"]} ("{document["title"]}") from the Washington State
Residential Landlord-Tenant Act. Document type: {document["type"]}.

A tenant-facing assistant will use these chunks to (a) answer plain-language questions about
tenant rights and (b) check lease clauses against the law. Chunk with these rules:

- Cover the ENTIRE document across your chunks — leave nothing out, with roughly 25% overlap
  between adjacent chunks. Aim for at least {how_many} chunks, more if the structure calls for it.
- Never split a numbered subsection (like "(2)(c)") across chunks mid-provision; keep each
  provision's text intact within a chunk.
- If the section enumerates a list of prohibited provisions, remedies, or duties, give each
  enumerated item its own chunk (plus surrounding context) so specific questions retrieve precisely.
- headline: MUST begin with "{document["section"]}", then a few plain-English words.
- summary: everyday renter vocabulary a non-lawyer would actually search for — translate the
  legalese; mention concrete scenarios (late fees, deposits, repairs, entry notice, eviction).
- original_text: the statute text exactly as written, unchanged.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


MAX_CHUNK_CHARS = 3000  # a chunk longer than this defeats retrieval granularity
MIN_CHARS_FOR_MULTI = 2000  # docs longer than this must not come back as a single chunk
VALIDATION_ATTEMPTS = 3


@retry(wait=wait)
def call_chunker(document: dict) -> list[Chunk]:
    messages = [{"role": "user", "content": make_prompt(document)}]
    response = completion(model=UTILITY_MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    return Chunks.model_validate_json(reply).chunks


def validate_chunks(document: dict, chunks: list[Chunk]) -> str | None:
    """Cheap models occasionally ignore chunking instructions; catch it instead of trusting it."""
    if len(document["text"]) > MIN_CHARS_FOR_MULTI and len(chunks) == 1:
        return f"{len(document['text'])} chars returned as a single chunk"
    oversized = max((len(c.original_text) for c in chunks), default=0)
    if oversized > MAX_CHUNK_CHARS:
        return f"chunk of {oversized} chars exceeds {MAX_CHUNK_CHARS}"
    return None


def process_document(document: dict) -> list[Result]:
    for attempt in range(VALIDATION_ATTEMPTS):
        doc_as_chunks = call_chunker(document)
        problem = validate_chunks(document, doc_as_chunks)
        if problem is None:
            break
        print(f"[retry {attempt + 1}] {document['section']}: {problem}")
    else:
        print(f"[warning] {document['section']}: accepted after {VALIDATION_ATTEMPTS} failed validations")
    return [chunk.as_result(document, plain=PLAIN_MODE) for chunk in doc_as_chunks]


PLAIN_MODE = False


def _init_worker(plain: bool):
    global PLAIN_MODE
    PLAIN_MODE = plain


def create_chunks(documents: list[dict], workers: int, plain: bool = False) -> list[Result]:
    chunks = []
    with Pool(processes=workers, initializer=_init_worker, initargs=(plain,)) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks


def create_embeddings(chunks: list[Result], state: str, suffix: str = "") -> None:
    collection_name = f"{state}_reference{suffix}"
    chroma = PersistentClient(path=DB_PATH)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)
    collection = chroma.get_or_create_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    for start in tqdm(range(0, len(texts), EMBED_BATCH_SIZE)):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        emb = openai.embeddings.create(model=EMBEDDING_MODEL, input=batch).data
        collection.add(
            ids=[f"{state}-{i}" for i in range(start, start + len(batch))],
            embeddings=[e.embedding for e in emb],
            documents=batch,
            metadatas=[chunk.metadata for chunk in chunks[start : start + len(batch)]],
        )
    print(f"Collection '{collection_name}' created with {collection.count()} chunks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="wa", help="State corpus to ingest (e.g. wa)")
    parser.add_argument("--limit", type=int, help="Only ingest the first N documents (smoke test)")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--plain", action="store_true", help="original text only, no augmentation (ablation collection <state>_reference_plain)")
    args = parser.parse_args()

    documents = fetch_documents(args.state)
    if args.limit:
        documents = documents[: args.limit]
        print(f"Smoke test: limited to {len(documents)} documents")

    chunks = create_chunks(documents, args.workers, plain=args.plain)
    print(f"Created {len(chunks)} chunks")
    create_embeddings(chunks, args.state, suffix="_plain" if args.plain else "")
    print("Ingestion complete")


if __name__ == "__main__":
    main()
