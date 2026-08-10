"""Boot check for the demo image, run inside the container by ci.yml.

The deployed artifact used to be the least-tested thing in the repo: CI linted
and tested the source, and nobody built the container until a deploy. What can
break there and nowhere else is the dependency set — the image installs
requirements-lock.txt rather than pyproject's ranges, and then removes two
packages chromadb declares but this deployment never uses. Both of those are
assertions about the runtime, so they are checked in the runtime.

Mounted rather than copied, because scripts/ is excluded from the build context:

    docker run --rm -e OPENAI_API_KEY=dummy \
        -v "$PWD/scripts/image_smoke_test.py:/app/smoke.py" leasehound:ci python /app/smoke.py

Importing the app is the whole point of the first check: it pulls in gradio,
litellm, openai and chromadb together, which is where a bad resolution shows up.

The second check OPENS the store, and works against the placeholder CI builds with,
because what it is testing is not the corpus — it is whether the runtime user can
write where Chroma needs to. That distinction cost a live demo: a `USER hound` line
landed while `COPY vector_db_runtime` still ran as root, so every scan died on
"attempt to write a readonly database" while the page itself served fine, CI stayed
green, and the UI reported only "the hound tripped over an error". An empty
placeholder exercises exactly that path, since Chroma initialises a new store there
and needs the same write access.

The third check WRITES what a scan writes. Same story as the second, one path over:
the store fix shipped, a visitor's scan then got all the way through 15 of 15 clauses
and died on `mkdir /app/logs`. Two write paths, two deploys, one cause — so this
exercises the writes rather than reasoning about them, and it costs nothing because
log_scan takes an empty meter.

What is still verified by hand before a deploy is RETRIEVAL — that the corpus in the
image answers a query — because the real store is not in git.
"""

import sys
from pathlib import Path

REMOVED_BY_DOCKERFILE = ("kubernetes", "onnxruntime")
# Where the Dockerfile puts the store, and where leasehound.retrieval looks for it.
STORE = Path("/app/vector_db")


def main() -> None:
    import leasehound.app  # noqa: F401  — the import IS the test

    imported = [name for name in REMOVED_BY_DOCKERFILE if name in sys.modules]
    if imported:
        raise SystemExit(
            f"The Dockerfile uninstalls {list(REMOVED_BY_DOCKERFILE)} because booting the app "
            f"does not import them. It now imports {imported}, so that is no longer true — "
            f"stop removing it, or the demo will fail on a visitor instead of here."
        )
    print(f"Image boots. Pinned dependencies import together; "
          f"{len(sys.modules)} modules loaded, none of {list(REMOVED_BY_DOCKERFILE)}.")

    import chromadb

    try:
        chromadb.PersistentClient(path=str(STORE))
    except Exception as opening:
        raise SystemExit(
            f"Chroma cannot open {STORE} as uid {__import__('os').getuid()}: "
            f"{type(opening).__name__}: {opening}\n"
            f"The store is opened read-write, so it has to belong to the user the "
            f"image runs as — see the --chown on its COPY in the Dockerfile. Without "
            f"this check the image boots, serves the page, and fails every scan."
        ) from opening
    print(f"Chroma opens {STORE} read-write as the runtime user.")

    from leasehound.metrics import LOG_PATH, UsageMeter, log_scan

    try:
        log_scan(UsageMeter(), "smoke.md", 0, verdicts={"red": 0, "yellow": 0, "green": 0})
    except OSError as writing:
        raise SystemExit(
            f"A scan cannot write its own metrics line: {type(writing).__name__}: "
            f"{writing}\nlog_scan appends to {LOG_PATH} and creates its parent, so that "
            f"directory has to belong to the user the image runs as. This failed after "
            f"every clause had been judged, which is the most expensive place to fail."
        ) from writing
    print(f"A scan can write {LOG_PATH} as the runtime user.")


if __name__ == "__main__":
    main()
