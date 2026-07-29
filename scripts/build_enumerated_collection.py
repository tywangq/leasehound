"""Re-index one section's enumerated catalog as one chunk per item, into a copy.

The roadmap's top item, made measurable. RCW 59.18.230 enumerates ten prohibited
lease provisions in subsection (2), and it is the only section that never reaches
the judge in scan mode (31 of 31 partial misses in eval_scan_retrieval). The
hypothesis: one embedding covering ten unrelated prohibitions is a smear, so no
single lease clause matches it strongly.

Two design choices matter, and both are about not paying twice.

It writes a COPY of the shipped collection rather than re-ingesting. Every chunk
outside the target section keeps its existing embedding, so building this costs
one embedding per new unit — the 355 untouched chunks cost nothing, and the
shipped collection is left exactly as it was in case the experiment loses.

It builds the units by PARSING, not by prompting. ingest.py's chunker is already
told "give each enumerated item its own chunk" and returned four length-shaped
chunks for .230 anyway; an instruction a model ignores is not a strategy. The
plain-language summary is still generated, because the point is to isolate
granularity — units without the renter-voice augmentation every other chunk
carries would confound the two.

    python -m scripts.build_enumerated_collection                    # wa_reference_230split
    python -m scripts.build_enumerated_collection --section "RCW 59.18.150"
"""

import argparse
import json
from pathlib import Path

from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field

from leasehound.ingest import CORPUS_PATH, SECTION_RE, split_enumerated_catalog
from leasehound.retrieval import (
    DB_PATH,
    EMBEDDING_MODEL,
    UTILITY_MODEL,
    llm_retry,
    openai,
)

DEFAULT_SECTION = "RCW 59.18.230"
RESULTS_PATH = Path(__file__).parent.parent / "evaluation" / "enumerated_index.json"


class UnitSummary(BaseModel):
    headline: str = Field(description="Begins with the citation and label, then a few "
                          "plain-English words naming the single thing this item prohibits "
                          "or requires, e.g. 'RCW 59.18.230(2)(f) — waiving landlord liability'")
    summary: str = Field(description="2-3 sentences in everyday renter vocabulary about this "
                         "ONE item only. Name the lease wording a tenant would actually see.")


@llm_retry
def summarize(section: str, title: str, label: str, text: str) -> UnitSummary:
    prompt = f"""One enumerated item from {section} ("{title}") of Washington State's
Residential Landlord-Tenant Act, item {label}.

This item will be retrieved on its own to check a single lease clause, so describe
ONLY what this item covers — do not summarize the rest of the section.

{text}
"""
    response = completion(model=UTILITY_MODEL, messages=[{"role": "user", "content": prompt}],
                         response_format=UnitSummary)
    return UnitSummary.model_validate_json(response.choices[0].message.content)


def corpus_file(section: str) -> Path:
    slug = section.lower().replace(" ", "-").replace(".", "-")
    matches = sorted(CORPUS_PATH.rglob(f"{slug}.md"))
    if not matches:
        raise SystemExit(f"No corpus file for {section} (looked for {slug}.md)")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="wa")
    parser.add_argument("--section", default=DEFAULT_SECTION)
    parser.add_argument("--suffix", default="_230split")
    args = parser.parse_args()

    source_name = f"{args.state}_reference"
    target_name = f"{source_name}{args.suffix}"
    chroma = PersistentClient(path=DB_PATH)
    source = chroma.get_collection(source_name)
    everything = source.get(include=["documents", "metadatas", "embeddings"])

    kept = [i for i, meta in enumerate(everything["metadatas"])
            if meta.get("section") != args.section]
    replaced = len(everything["ids"]) - len(kept)
    if not replaced:
        raise SystemExit(f"{args.section} has no chunks in {source_name} — nothing to replace")

    path = corpus_file(args.section)
    text = path.read_text(encoding="utf-8")
    title_match = SECTION_RE.search(text)
    title = title_match.group(2) if title_match else args.section
    units = split_enumerated_catalog(text)
    if not units:
        raise SystemExit(f"{args.section} holds no enumerated catalog to split")

    # Metadata has to match the replaced chunks exactly, or the eval would be
    # measuring a metadata change instead of a granularity change.
    template = next(meta for meta in everything["metadatas"]
                    if meta.get("section") == args.section)

    print(f"{args.section}: {replaced} chunks -> {len(units)} enumerated units")
    documents, metadatas = [], []
    for unit in units:
        described = summarize(args.section, title, unit.label, unit.text)
        documents.append(f"{described.headline}\n\n{described.summary}\n\n{unit.text}")
        metadatas.append(dict(template))
        print(f"  {unit.label}  {described.headline}")

    embeddings = [e.embedding for e in
                  openai.embeddings.create(model=EMBEDDING_MODEL, input=documents).data]

    if target_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(target_name)
    target = chroma.get_or_create_collection(target_name)
    # Reuse every untouched embedding rather than re-embedding 355 chunks.
    target.add(
        ids=[everything["ids"][i] for i in kept],
        embeddings=[everything["embeddings"][i] for i in kept],
        documents=[everything["documents"][i] for i in kept],
        metadatas=[everything["metadatas"][i] for i in kept],
    )
    target.add(
        ids=[f"{args.state}-{args.section.split('.')[-1]}{u.label}" for u in units],
        embeddings=embeddings, documents=documents, metadatas=metadatas,
    )
    print(f"\nCollection '{target_name}': {target.count()} chunks "
          f"({len(kept)} reused + {len(units)} new)")

    RESULTS_PATH.write_text(json.dumps({
        "collection": target_name, "section": args.section,
        "chunks_replaced": replaced, "units_created": len(units),
        "embeddings_paid_for": len(units),
        "labels": [u.label for u in units],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
