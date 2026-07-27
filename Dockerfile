# Demo image for Cloud Run. The vector DB ships inside the image — it's a
# ~42MB read-only artifact, not state — so the container needs no volume and
# scales to zero cleanly. OPENAI_API_KEY comes from the service environment.
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml LICENSE ./
COPY leasehound ./leasehound
RUN pip install --no-cache-dir .

COPY vector_db ./vector_db
COPY examples ./examples

ENV GRADIO_SERVER_NAME=0.0.0.0
# Cloud Run injects PORT; anywhere else the default keeps parity with local dev.
CMD ["sh", "-c", "GRADIO_SERVER_PORT=${PORT:-7860} python -m leasehound.app"]
