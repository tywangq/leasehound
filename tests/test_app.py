"""App helpers: cited-only sources footer, upload cleanup (privacy), scan cache."""

import json
from pathlib import Path

import leasehound.app as app
from leasehound.app import (
    MODEL_FOOTER,
    SAMPLE_LEASE,
    cleanup_upload,
    sources_footer,
    strip_footer,
)
from leasehound.retrieval import Result


def chunk(section: str) -> Result:
    return Result(page_content="text", metadata={"section": section, "url": f"https://law/{section}"})


RETRIEVED = [chunk("RCW 59.18.170"), chunk("RCW 59.18.230"), chunk("RCW 59.18.060")]


def test_footer_lists_only_statutes_the_answer_cites():
    answer = "Late fees can only start on day six under RCW 59.18.170(2)."
    footer = sources_footer(answer, RETRIEVED)
    assert "RCW 59.18.170" in footer
    assert "RCW 59.18.230" not in footer
    assert "RCW 59.18.060" not in footer


def test_footer_empty_when_answer_cites_nothing():
    # An off-topic question retrieves chunks but the answer declines — no footer.
    assert sources_footer("I only cover Washington State law.", RETRIEVED) == ""


def test_footer_ignores_citations_not_in_the_retrieved_set():
    assert sources_footer("See RCW 99.99.999 for details.", RETRIEVED) == ""


def test_strip_footer_round_trips_the_appended_footer():
    # The footer must come back off before history re-enters the LLM, or the
    # model mimics it and answers grow a second "Statutes cited" block.
    answer = "Late fees start on day six under RCW 59.18.170(2)."
    message = {"role": "assistant", "content": answer + sources_footer(answer, RETRIEVED)}
    assert strip_footer(message) == {"role": "assistant", "content": answer}


def test_strip_footer_leaves_other_messages_alone():
    user = {"role": "user", "content": "what about late fees?"}
    plain = {"role": "assistant", "content": "Day six, per RCW 59.18.170(2)."}
    assert strip_footer(user) == user
    assert strip_footer(plain) == plain


def test_model_written_footer_is_stripped_from_the_answer_tail():
    # Belt and suspenders: even if the model writes its own sources list
    # (this happened — mimicry before strip_footer existed), it gets dropped
    # so the mechanical footer isn't a duplicate. Both wordings, since the
    # model may have learned either from an older conversation.
    body = "Two days' notice is required (RCW 59.18.150(6))."
    for header in ("Statutes cited in this answer:", "Statutes cited:"):
        imitation = body + f"\n\n{header}\n\n* [RCW 59.18.150(6)](https://law/x)"
        assert MODEL_FOOTER.sub("", imitation).rstrip() == body


def test_cleanup_removes_upload_but_never_the_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(app.tempfile, "gettempdir", lambda: str(tmp_path))
    upload = tmp_path / "lease.pdf"
    upload.write_text("uploaded")
    cleanup_upload(upload)
    assert not upload.exists()

    outside = tmp_path.parent / "keep.pdf"
    outside.write_text("keep")
    monkeypatch.setattr(app.tempfile, "gettempdir", lambda: str(tmp_path / "elsewhere"))
    cleanup_upload(outside)
    assert outside.exists()
    outside.unlink()

    cleanup_upload(SAMPLE_LEASE)
    assert SAMPLE_LEASE.exists()


def test_scan_cache_is_bounded_lru():
    app._scan_cache.clear()
    for i in range(app.CACHE_MAX_ENTRIES + 5):
        app.cache_put(f"digest{i}", {"findings": [], "protections": []})
    assert len(app._scan_cache) == app.CACHE_MAX_ENTRIES
    assert app.cache_get("digest0") is None  # oldest evicted
    assert app.cache_get(f"digest{app.CACHE_MAX_ENTRIES + 4}") is not None
    app._scan_cache.clear()


def test_second_scan_of_identical_content_makes_no_api_calls(monkeypatch, tmp_path):
    import leasehound.metrics as metrics

    filler = "The parties agree to the terms set forth in this provision as written. "
    lease_text = f"Lease intro. {filler}\n\n" + "\n\n".join(
        f"{i}. CLAUSE HEADING. {filler}" for i in range(1, 5)
    )
    api_calls = {"count": 0}

    def fake_scan_clauses(clauses, config, meter=None):
        api_calls["count"] += 1
        yield {"index": 1, "clause": clauses[0], "verdict": "green",
               "citations": [], "urls": {}, "explanation": "fine"}

    monkeypatch.setattr(app, "read_document", lambda path: lease_text)
    monkeypatch.setattr(app, "looks_like_lease", lambda clauses, meter=None: True)
    monkeypatch.setattr(app, "scan_clauses", fake_scan_clauses)
    monkeypatch.setattr(app, "check_protections", lambda clauses, meter=None: [])
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "scan_metrics.jsonl")
    app._scan_cache.clear()

    first_history: list = []
    list(app.scan_flow(Path("first_upload.md"), "key-a", first_history, "", "", []))
    # Same content under a different name, path, and session: cache must serve it.
    second_history: list = []
    list(app.scan_flow(Path("renamed_copy.md"), "key-b", second_history, "", "", []))

    assert api_calls["count"] == 1
    assert any(m["content"] == app.CACHED_SNIFF for m in second_history)
    # The cached scan renders under the new file name, not the original's.
    log_lines = (tmp_path / "scan_metrics.jsonl").read_text().splitlines()
    assert json.loads(log_lines[1])["cache_hit"] is True
    assert json.loads(log_lines[1])["source"] == "renamed_copy.md"
    app._scan_cache.clear()
