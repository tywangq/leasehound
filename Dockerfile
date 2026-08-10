# Demo image for Cloud Run. The vector store ships inside the image — a
# read-only artifact, not state — so the container needs no volume and scales to
# zero cleanly. OPENAI_API_KEY comes from the service environment.
#
# Two choices here are deliberate.
#
# The store is the one scripts/export_runtime_db.py writes, not vector_db/. The
# development store also holds the three ablation collections the evaluation
# needs and the app never queries, so copying it shipped 1655 chunks and ~49 MB
# to serve 359 chunks in 9.8 MB — and every megabyte is paid again on each cold
# start, which is the one number the demo's first visitor actually feels.
#
# Dependencies come from requirements-lock.txt, not from pyproject's ranges. The
# store is a Chroma on-disk artifact coupled to the version that wrote it, so
# "chromadb>=1.0" is a promise this image cannot keep: a rebuild months from now
# resolves whatever is current and may not read its own data. CI's test job
# deliberately stays on the ranges — that run is the canary for upstream drift,
# while this one has to be reproducible.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-lock.txt ./
# chromadb declares two dependencies this deployment cannot use: kubernetes, for
# running Chroma as a distributed server, and onnxruntime, for its built-in local
# embedding function — 133 MB together, against a PersistentClient that embeds
# through the OpenAI API. Neither is in sys.modules after the app boots, which is
# what makes removing them safe and what the CI smoke test re-checks on every
# commit: if a future chromadb starts importing either one, the import fails
# loudly there instead of quietly in front of a visitor.
RUN pip install --no-cache-dir -r requirements-lock.txt \
    && pip uninstall -y --no-cache-dir kubernetes onnxruntime

COPY pyproject.toml LICENSE ./
COPY leasehound ./leasehound
RUN pip install --no-cache-dir --no-deps .

COPY vector_db_runtime ./vector_db
COPY examples ./examples

# Drop root, after the installs and before anything runs. The first code to touch a
# visitor's upload is pypdf parsing an arbitrary PDF, and this container was doing that
# as uid 0 — Cloud Run's sandbox is a second line of defence, not a reason to skip the
# first. Nothing at runtime writes inside /app: the vector store is read-only, the
# report temp files and Gradio's own scratch space go to /tmp and $HOME.
RUN useradd --create-home --uid 10001 hound
USER hound
ENV HOME=/home/hound

ENV GRADIO_SERVER_NAME=0.0.0.0
# Cloud Run injects PORT; anywhere else the default keeps parity with local dev.
CMD ["sh", "-c", "GRADIO_SERVER_PORT=${PORT:-7860} python -m leasehound.app"]
