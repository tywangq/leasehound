"""Export just the collection the app serves, for the container to ship.

The demo image bakes the vector store in — that is what makes the container
stateless and lets it scale to zero. But `vector_db/` is a *development* store:
alongside `wa_reference` (the 359 chunks the app queries) it accumulated the
three ablation collections the evaluation needs — `_naive`, `_plain`, and
`_230split`, the last one an experiment that was measured and rejected. Copying
the directory into the image shipped 1655 chunks to serve 359, plus whatever
segment directories earlier rebuilds had orphaned, and every megabyte is paid
for again on each cold start.

So the image gets a purpose-built store instead of a copy of a dev artifact.
This writes a fresh Chroma at `vector_db_runtime/`, holding one collection:

    python -m scripts.export_runtime_db
    python -m scripts.export_runtime_db --collection wa_reference --force

It costs nothing. Every chunk already has an embedding in the source store, so
this reads the vectors and writes them back — no model is called, which also
means the exported store is byte-for-byte the same retrieval the evaluation
measured rather than a re-ingestion that would need re-measuring.

Chunks are written in document order, because `retrieval.chunk_order` reads the
integer suffix of an id to reassemble a section for `section_completion`. The
ids are copied verbatim rather than renumbered: renumbering here would make the
container's ids disagree with the ids in the eval artifacts.
"""

import argparse
import shutil
from pathlib import Path

from chromadb import PersistentClient

from leasehound.retrieval import DB_PATH, PipelineConfig, chunk_order

DEFAULT_TARGET = Path(__file__).parent.parent / "vector_db_runtime"


def directory_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1)


def export(source_path: str, target_path: Path, name: str) -> int:
    source = PersistentClient(path=source_path).get_collection(name)
    everything = source.get(include=["documents", "metadatas", "embeddings"])
    entries = sorted(
        ({"id": i, "document": d, "metadata": m, "embedding": e}
         for i, d, m, e in zip(everything["ids"], everything["documents"],
                               everything["metadatas"], everything["embeddings"])),
        key=lambda e: chunk_order(e["id"]),
    )

    target = PersistentClient(path=str(target_path)).get_or_create_collection(name)
    target.add(
        ids=[e["id"] for e in entries],
        embeddings=[e["embedding"] for e in entries],
        documents=[e["document"] for e in entries],
        metadatas=[e["metadata"] for e in entries],
    )
    return target.count()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=PipelineConfig.collection,
                        help="The collection the app serves; everything else stays behind")
    parser.add_argument("--source", default=DB_PATH)
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--force", action="store_true",
                        help="Replace an existing export instead of refusing")
    args = parser.parse_args()

    target_path = Path(args.target)
    if target_path.exists():
        # Refuse by default: this path is what gets deployed, and silently
        # merging a new export into an old one would ship both.
        if not args.force:
            raise SystemExit(f"{target_path} already exists — pass --force to replace it")
        shutil.rmtree(target_path)

    source_mb = directory_size_mb(Path(args.source))
    count = export(args.source, target_path, args.collection)
    target_mb = directory_size_mb(target_path)
    print(f"{args.collection}: {count} chunks -> {target_path}")
    print(f"{source_mb} MB (development store, all collections) "
          f"-> {target_mb} MB (runtime store)")


if __name__ == "__main__":
    main()
