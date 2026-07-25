"""Chunking helpers: naive splitting and the LLM-output validators."""

from leasehound.ingest import Chunk, document_metadata, validate_chunks
from leasehound.ingest_naive import CHUNK_OVERLAP, CHUNK_SIZE, naive_split


def make_chunk(text: str) -> Chunk:
    return Chunk(headline="RCW 59.18.010 test", summary="A summary.", original_text=text)


DOCUMENT = {
    "source": "corpus/wa/statutes/rcw-59-18-010.md",
    "type": "statutes",
    "state": "wa",
    "section": "RCW 59.18.010",
    "title": "Test section",
    "url": "https://example.test",
    "text": "x" * 2500,
}


def test_naive_split_covers_text_with_overlap():
    text = "abcdefghij" * 250  # 2500 chars
    chunks = naive_split(text)
    assert chunks[0] == text[:CHUNK_SIZE]
    assert chunks[1][:CHUNK_OVERLAP] == chunks[0][-CHUNK_OVERLAP:]
    assert "".join(c[CHUNK_OVERLAP:] for c in chunks).endswith(text[-100:])


def test_naive_split_short_text_is_one_chunk():
    assert naive_split("short") == ["short"]


def test_validate_flags_long_doc_returned_as_single_chunk():
    problem = validate_chunks(DOCUMENT, [make_chunk("one chunk only")])
    assert problem is not None and "single chunk" in problem


def test_validate_flags_oversized_chunk():
    problem = validate_chunks(DOCUMENT, [make_chunk("y" * 4000), make_chunk("ok")])
    assert problem is not None and "exceeds" in problem


def test_validate_accepts_reasonable_chunks():
    chunks = [make_chunk("y" * 800), make_chunk("z" * 900)]
    assert validate_chunks(DOCUMENT, chunks) is None


def test_document_metadata_projects_exactly_the_index_fields():
    metadata = document_metadata(DOCUMENT)
    assert "text" not in metadata
    assert metadata["section"] == "RCW 59.18.010"
    assert set(metadata) == {"source", "type", "state", "section", "title", "url"}
