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

import numpy
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field

from leasehound.ingest import CORPUS_PATH, SECTION_RE, split_enumerated_catalog
from leasehound.retrieval import (
    DB_PATH,
    EMBEDDING_MODEL,
    UTILITY_MODEL,
    chunk_order,
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


def clean(text: str) -> str:
    """Drop the stray control characters the cheap model sprinkles into headlines.

    The headline is embedded along with everything else, so "RCW 59.18.230(2)(e) 7
    attorney's fees" is not a cosmetic problem — it is noise inside the vector.
    """
    return " ".join("".join(c for c in text if c.isprintable()).split())


@llm_retry
def summarize(section: str, title: str, label: str, text: str) -> UnitSummary:
    prompt = f"""One enumerated item from {section} ("{title}") of Washington State's
Residential Landlord-Tenant Act, item {label}.

This item will be retrieved on its own to check a single lease clause, so describe
ONLY what this item covers — do not summarize the rest of the section.

{text}
"""
    # temperature=0 so a rebuild reproduces the collection this experiment measured.
    # Without it every rebuild re-embeds different text and the published retrieval
    # numbers describe an index nobody can reconstruct.
    response = completion(model=UTILITY_MODEL, messages=[{"role": "user", "content": prompt}],
                         response_format=UnitSummary, temperature=0)
    described = UnitSummary.model_validate_json(response.choices[0].message.content)
    return UnitSummary(headline=clean(described.headline), summary=clean(described.summary))


def splice_units(entries: list[dict], section: str, units: list[dict]) -> list[dict]:
    """Put the units where the section's chunks were, and renumber every id.

    Ids carry document order — `retrieval.chunk_order` reads the integer suffix to
    reassemble a section for `section_completion`, and anything it can't parse sorts
    to 0. So the units cannot get ids like `wa-230(2)(a)`: that reads as position 0
    and silently scrambles reading order the moment both features are on at once.
    Renumbering the whole collection sequentially keeps the one scheme that works.
    """
    positions = [i for i, e in enumerate(entries) if e["metadata"].get("section") == section]
    if not positions:
        raise SystemExit(f"{section} has no chunks to replace")
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise SystemExit(f"{section}'s chunks are not contiguous — refusing to guess a position")
    spliced = entries[: positions[0]] + units + entries[positions[-1] + 1:]
    state = entries[0]["id"].rsplit("-", 1)[0]
    for i, entry in enumerate(spliced):
        entry["id"] = f"{state}-{i}"
    return spliced


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
    # Sorted into document order first: Chroma's get() makes no ordering promise, and
    # the splice below needs the section's chunks to be adjacent to mean anything.
    entries = sorted(
        ({"id": i, "document": d, "metadata": m, "embedding": e}
         for i, d, m, e in zip(everything["ids"], everything["documents"],
                               everything["metadatas"], everything["embeddings"])),
        key=lambda e: chunk_order(e["id"]),
    )
    replaced = sum(1 for e in entries if e["metadata"].get("section") == args.section)
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
    documents, labels = [], []
    for unit in units:
        described = summarize(args.section, title, unit.label, unit.text)
        documents.append(f"{described.headline}\n\n{described.summary}\n\n{unit.text}")
        labels.append(unit.label)
        print(f"  {unit.label}  {described.headline}")

    # Chroma rejects a mix of ndarray and list, and the reused embeddings come back
    # from get() as ndarrays — so the new ones are converted rather than appended raw.
    embeddings = [numpy.asarray(e.embedding, dtype=numpy.float32) for e in
                  openai.embeddings.create(model=EMBEDDING_MODEL, input=documents).data]
    new_entries = [{"id": "", "document": doc, "metadata": dict(template), "embedding": emb}
                   for doc, emb in zip(documents, embeddings)]
    spliced = splice_units(entries, args.section, new_entries)

    if target_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(target_name)
    target = chroma.get_or_create_collection(target_name)
    # Every untouched embedding is reused rather than re-embedded, so the whole
    # experiment costs one embedding per new unit.
    target.add(
        ids=[e["id"] for e in spliced],
        embeddings=[e["embedding"] for e in spliced],
        documents=[e["document"] for e in spliced],
        metadatas=[e["metadata"] for e in spliced],
    )
    print(f"\nCollection '{target_name}': {target.count()} chunks "
          f"({len(entries) - replaced} reused + {len(units)} new, ids renumbered in order)")

    RESULTS_PATH.write_text(json.dumps({
        "collection": target_name, "section": args.section,
        "chunks_replaced": replaced, "units_created": len(units),
        "embeddings_paid_for": len(units),
        "labels": labels,
        "ids_renumbered": True,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
