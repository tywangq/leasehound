"""What produced a number: models, corpus snapshot, and commit.

The corpus already carries a snapshot date because statute text moves, and
`scripts/check_corpus_drift.py` watches it. Models move too — a provider can
change what `gpt-4.1-mini` points at without changing the string — and until
this module existed no result file recorded which models it was measured with.
That made every published table unreproducible in the one way that matters: you
could not tell whether a changed score meant changed code or a changed model.

So every eval artifact now carries the same stamp. Nothing here calls an API;
it reads the constants the pipeline is actually configured with, which is also
why it picks up the `LEASEHOUND_*_MODEL` overrides rather than the defaults.
"""

import hashlib
import subprocess
from datetime import datetime, timezone

from leasehound.retrieval import EMBEDDING_MODEL, GENERATION_MODEL, UTILITY_MODEL
from leasehound.scan import CORPUS_SNAPSHOT, judge_fingerprint


def index_fingerprint(collection: str = "wa_reference") -> dict | None:
    """Which index a number was measured against, and how healthy it was.

    The gap this closes was found by the first CI run that had an API key. It
    rebuilt the store from `corpus/` — same commit, same models, same corpus
    snapshot, every field of the stamp identical — and scored `section_hit@1`
    **0.667 against the shipped store's 0.833**, on the retrieval-only eval, which
    makes no completions at all. Nothing in the stamp could have told you why,
    because the index was the one input it did not name.

    Ingestion is LLM semantic chunking, and `process_document` accepts a document
    whose validation never passed: three tries, then `[warning] ... accepted after
    3 failed validations`, and the section ships as one unsplit block. RCW 59.18.150
    — the entry statute — came back as a single 9,959-character chunk in that build.
    A blob that size embeds poorly against a specific question, which is where the
    16 points went.

    `degraded` counts the chunks over `MAX_CHUNK_CHARS`, so a store carrying that
    damage says so rather than looking like any other store. The shipped one is not
    clean either — 7 chunks over, the largest 13,701 characters — which is the point:
    the question was never "did this build fail" but "which build is this".

    Returns None when no store is present (CI's free jobs, a fresh clone), because a
    stamp that invents an index is worse than one that admits it has none.
    """
    from leasehound.retrieval import DB_PATH

    try:
        from chromadb import PersistentClient

        documents = PersistentClient(path=DB_PATH).get_collection(collection).get()["documents"]
    except Exception:
        return None
    return fingerprint_documents(documents, collection)


def fingerprint_documents(documents: list[str], collection: str = "wa_reference") -> dict:
    """The fingerprint itself, over chunk text — separated from fetching it.

    Split out because the version that read Chroma directly could only be tested where
    a store exists, and CI's free `test` job has none: both tests passed locally and
    failed on the first push. Skipping them there would have been worse — a test that
    only ever skips guards nothing, which is the same reason `tests/test_css.py` has a
    job of its own rather than a skip marker.
    """
    from leasehound.ingest import MAX_CHUNK_CHARS

    # Content, not ids: two builds of the same corpus produce different ids for the
    # same text, and it is the text the embeddings are of.
    digest = hashlib.sha256(
        "\n".join(sorted(documents)).encode("utf-8")).hexdigest()[:12]
    return {
        "collection": collection,
        "chunks": len(documents),
        "digest": digest,
        "degraded": sum(1 for d in documents if len(d) > MAX_CHUNK_CHARS),
    }


def commit() -> str | None:
    """Short SHA of the code that produced the run, or None outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def stamp(models: bool = True) -> dict:
    """Provenance for one run.

    `models=False` is for runs that call no model and read no corpus — the
    concurrency harness stubs the whole LLM layer, so naming three models there
    would imply they had something to do with the number. Commit and date still
    apply, and they are the fields that make a re-run comparable, so the same
    stamping path covers both kinds of run rather than growing a second one.
    """
    origin = {
        "commit": commit(),
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    if not models:
        return origin
    return {
        "generation_model": GENERATION_MODEL,
        "utility_model": UTILITY_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "corpus_snapshot": CORPUS_SNAPSHOT,
        # The same argument as the models one field up, applied to the other input
        # that moves: every scan verdict comes from one prompt and one response
        # schema, both of which have been edited to fix particular failures. Without
        # this, a changed prompt and a changed model look identical from the artifact.
        "judge_prompt": judge_fingerprint(),
        # The third input that moves, and the one that hid the longest. Models and
        # prompts were named here from the start; the index they are queried against
        # was not, and two builds of it from the same corpus are not the same index.
        "index": index_fingerprint(),
        **origin,
    }
