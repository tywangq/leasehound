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

import subprocess
from datetime import datetime, timezone

from leasehound.retrieval import EMBEDDING_MODEL, GENERATION_MODEL, UTILITY_MODEL
from leasehound.scan import CORPUS_SNAPSHOT


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


def stamp() -> dict:
    return {
        "generation_model": GENERATION_MODEL,
        "utility_model": UTILITY_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "corpus_snapshot": CORPUS_SNAPSHOT,
        "commit": commit(),
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
