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
The store is not exercised here — CI builds the image against a placeholder,
since the real one is not in git and re-embedding it per commit would cost money.
Retrieval against the real store is verified locally before a deploy.
"""

import sys

REMOVED_BY_DOCKERFILE = ("kubernetes", "onnxruntime")


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


if __name__ == "__main__":
    main()
