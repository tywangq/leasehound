"""App helpers: cited-only sources footer, upload cleanup (privacy), scan cache."""

import json
from pathlib import Path

from fastapi import FastAPI

import leasehound.app as app
import leasehound.scan as scan
from leasehound.app import (
    MODEL_FOOTER,
    SAMPLE_LEASE,
    cleanup_upload,
    sources_footer,
    strip_footer,
)
from leasehound.metrics import UsageMeter
from leasehound.retrieval import Result


def gate_returns(kind: str = "lease_agreement", state: str = "wa"):
    """A stand-in for `classify_document`, which returns a model rather than a kind.

    Jurisdiction defaults to "wa" — the state every scan in these tests applies — so
    a test that says nothing about jurisdiction gets no mismatch warning.
    """
    return lambda clauses, meter=None: scan.DocumentCheck(kind=kind, jurisdiction=state)


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
    # Stubbed on leasehound.scan, not on app: there is one orchestration now, and
    # the UI reaches these stages through it rather than calling them itself.
    monkeypatch.setattr(scan, "classify_document", gate_returns())
    monkeypatch.setattr(scan, "scan_clauses", fake_scan_clauses)
    monkeypatch.setattr(scan, "check_protections", lambda clauses, meter=None: [])
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


def test_the_override_phrase_reaches_the_scan_and_is_not_answered_as_a_question(
        monkeypatch, tmp_path):
    """The seam between what the visitor types and what the scan is told. Without
    this, "scan anyway" could look right in chat and still scan normally — and it
    must not also be handed to ask mode, which would answer a control phrase as
    though it were a question about tenant law."""
    seen = {}

    def fake_scan_flow(path, key, history, report, scanned, context_base,
                       question="", scan_anyway=False):
        seen.update(question=question, scan_anyway=scan_anyway)
        yield app._out(history)

    monkeypatch.setattr(app, "scan_flow", fake_scan_flow)
    upload = tmp_path / "maybe_a_lease.md"
    upload.write_text("1. RENT. Tenant pays.", encoding="utf-8")

    list(app._respond({"text": "Scan Anyway please", "files": [str(upload)]},
                      [], "", ""))
    assert seen == {"question": "", "scan_anyway": True}, "case must not matter"

    list(app._respond({"text": "what about my deposit?", "files": [str(upload)]},
                      [], "", ""))
    assert seen == {"question": "what about my deposit?", "scan_anyway": False}


def test_a_question_naming_another_state_is_warned_about_above_the_answer(
        monkeypatch, tmp_path):
    """Above, not below. A renter reading "your landlord cannot do that" has acted on
    it by the time they reach a footnote — and the answer still arrives, because the
    Washington answer is not nothing, it is just not theirs."""
    import leasehound.answer as answer_module
    import leasehound.metrics as metrics

    def fake_answer(question, history=None, config=None, report_context=False):
        return answer_module.AskResult(
            stream=iter(["Under RCW 59.18.230, no."]), chunks=[], meter=UsageMeter(),
            routed=True, jurisdiction="ca")

    monkeypatch.setattr(app, "answer_question", fake_answer)
    monkeypatch.setattr(metrics, "ASK_LOG_PATH", tmp_path / "ask_metrics.jsonl")

    history: list = [{"role": "user", "content": "In California, can they keep my deposit?"}]
    list(app.answer_flow("In California, can they keep my deposit?", history, "", []))

    said = [m.get("content") or "" for m in history]
    warning = next(i for i, m in enumerate(said) if "🌎" in m)
    answered = next(i for i, m in enumerate(said) if "RCW 59.18.230, no." in m)
    assert warning < answered, "the warning has to arrive before the answer it qualifies"
    assert "CA" in said[warning]
    assert "Washington" in said[warning]


def test_an_out_of_state_lease_is_scanned_with_the_wrong_law_named_in_chat(
        monkeypatch, tmp_path):
    """The report carries the warning too, but a visitor watching the scan happen
    should not have to open the panel to learn that the law being applied is not the
    law that governs their tenancy."""
    import leasehound.metrics as metrics

    filler = "The parties agree to the terms set forth in this provision as written. "
    text = "\n\n".join(f"{i}. HEADING. {filler}" for i in range(1, 4))

    def fake_scan_clauses(clauses, config, meter=None):
        for i, clause in enumerate(clauses, start=1):
            yield {"index": i, "clause": clause, "verdict": "green",
                   "citations": [], "urls": {}, "explanation": "fine"}

    monkeypatch.setattr(app, "read_document", lambda path: text)
    monkeypatch.setattr(scan, "classify_document", gate_returns(state="ca"))
    monkeypatch.setattr(scan, "scan_clauses", fake_scan_clauses)
    monkeypatch.setattr(scan, "check_protections", lambda clauses, meter=None: [])
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "scan_metrics.jsonl")
    app._scan_cache.clear()

    history: list = []
    frames = list(app.scan_flow(Path("ca_lease.pdf"), "key", history, "", "", []))

    said = [m.get("content") or "" for m in history]
    assert any("CA law" in m for m in said), "the state has to be named, not implied"
    # Not the not-a-lease line: this IS a residential lease and the gate accepted it.
    assert not any(app.NOT_A_LEASE in m for m in said)
    # The scan finished and pinned a report, warning and all.
    report = frames[-1][2]
    assert "🌎" in report and "Judged against: WA law" in report


def test_a_refused_scan_leaves_ask_mode_on_law_only_context(monkeypatch, tmp_path):
    """The subtle half of refusing. Pinning an empty panel would read as "0 red
    flags", and setting the report context would tell ask mode it has the scan
    report of a document nobody judged — every follow-up answer would then be
    reasoning from a report that does not exist."""
    import leasehound.metrics as metrics

    filler = "The parties agree to the terms set forth in this provision as written. "
    text = "Intro.\n\n" + "\n\n".join(f"{i}. HEADING. {filler}" for i in range(1, 4))
    judged = {"count": 0}

    def fake_scan_clauses(clauses, config, meter=None):
        judged["count"] += 1
        yield {"index": 1, "clause": clauses[0], "verdict": "red",
               "citations": [], "urls": {}, "explanation": "bad"}

    monkeypatch.setattr(app, "read_document", lambda path: text)
    monkeypatch.setattr(scan, "classify_document", gate_returns("other"))
    monkeypatch.setattr(scan, "scan_clauses", fake_scan_clauses)
    monkeypatch.setattr(scan, "check_protections", lambda clauses, meter=None: [])
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "scan_metrics.jsonl")
    app._scan_cache.clear()

    history: list = []
    frames = list(app.scan_flow(Path("resume.pdf"), "key", history, "", "", []))

    assert judged["count"] == 0, "no clause may be judged after a refusal"
    assert any(app.NOT_ABOUT_RENTING in (m.get("content") or "") for m in history)
    # _out's positional contract: report, state, context are 3rd, 4th, 5th.
    _, _, report, state, context = frames[-1][:5]
    assert report == "" and state == ""
    assert context == app.LAW_ONLY_CONTEXT

    # And the override gets through: same document, same stubs, one word added.
    app._scan_cache.clear()
    override: list = []
    list(app.scan_flow(Path("resume.pdf"), "key", override, "", "", [], scan_anyway=True))
    assert judged["count"] == 1
    app._scan_cache.clear()


def test_a_long_document_that_is_refused_is_never_promised_a_scan(monkeypatch, tmp_path):
    """Found by uploading a long non-lease: the app said "this splits into N clauses,
    the first 60 will be judged" and only then said it was not a lease. The cap notice
    is a promise about a scan, and at that point the gate had not said there would be
    one — so it now waits for the pipeline's `judging` step, which fires only after the
    gate accepts."""
    import leasehound.metrics as metrics

    filler = "The parties agree to the terms set forth in this provision as written. "
    total = app.MAX_CLAUSES + 30
    text = "\n\n".join(f"{i}. HEADING. {filler}" for i in range(1, total + 1))
    judged = []

    def fake_scan_clauses(clauses, config, meter=None):
        judged.extend(clauses)
        return iter(())

    monkeypatch.setattr(app, "read_document", lambda path: text)
    monkeypatch.setattr(scan, "classify_document", gate_returns("other"))
    monkeypatch.setattr(scan, "scan_clauses", fake_scan_clauses)
    monkeypatch.setattr(scan, "check_protections", lambda clauses, meter=None: [])
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "scan_metrics.jsonl")
    app._scan_cache.clear()

    history: list = []
    frames = list(app.scan_flow(Path("very_long_resume.pdf"), "k", history, "", "", []))

    chat = " ".join(m["content"] for m in history)
    assert app.NOT_ABOUT_RENTING in chat, "it is still refused"
    assert judged == [], "and nothing is judged"
    assert str(app.MAX_CLAUSES) not in chat, (
        "the clause cap must not be mentioned for a scan that never happens")
    assert "won't get a verdict" not in chat
    # And no report was ever pinned. (_out's positional contract: report is 3rd.)
    assert not any(frame[2] for frame in frames if isinstance(frame[2], str))


def test_a_long_lease_is_scanned_partially_instead_of_being_turned_away(monkeypatch, tmp_path):
    """The demo's likeliest visitor action is uploading a real lease, and real WA
    housing agreements run to 270 clauses. Refusing over the cap meant that visitor
    got an apology; judging the first 60 and naming the rest costs the same."""
    import leasehound.metrics as metrics

    filler = "The parties agree to the terms set forth in this provision as written. "
    total = app.MAX_CLAUSES + 30
    lease_text = "\n\n".join(f"{i}. CLAUSE HEADING. {filler}" for i in range(1, total + 1))
    judged, protections_saw = [], []

    def fake_scan_clauses(clauses, config, meter=None):
        judged.extend(clauses)
        for i, clause in enumerate(clauses, start=1):
            yield {"index": i, "clause": clause, "verdict": "green",
                   "citations": [], "urls": {}, "explanation": "fine"}

    monkeypatch.setattr(app, "read_document", lambda path: lease_text)
    monkeypatch.setattr(scan, "classify_document", gate_returns())
    monkeypatch.setattr(scan, "scan_clauses", fake_scan_clauses)
    monkeypatch.setattr(scan, "check_protections",
                        lambda clauses, meter=None: protections_saw.extend(clauses) or [])
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "scan_metrics.jsonl")
    app._scan_cache.clear()

    history: list = []
    list(app.scan_flow(Path("uw_housing_agreement.pdf"), "key-long", history, "", "", []))

    assert len(judged) == app.MAX_CLAUSES, "the clause pass must stop at the cap"
    # The negative-space pass is exempt: "this lease omits X" needs the whole document.
    # Which is why the cap alone never bounded a scan — see the next test.
    assert len(protections_saw) == total
    chat = " ".join(m["content"] for m in history)
    assert f"clauses {app.MAX_CLAUSES + 1}–{total}" in chat, "the visitor is told what was skipped"
    assert app.HOUND_TRIPPED not in chat
    logged = json.loads((tmp_path / "scan_metrics.jsonl").read_text().splitlines()[0])
    assert logged["clauses"] == app.MAX_CLAUSES and logged["clauses_total"] == total
    app._scan_cache.clear()


def test_a_document_too_large_to_scan_is_turned_away_without_spending(monkeypatch, tmp_path):
    """The other end of the size question, and the opposite answer to the one above.

    Over the CLAUSE cap a scan degrades, because "clauses 1–60 were judged" is still
    true. Over the DOCUMENT limit it stops, because the protections pass reads
    everything and a window that never ran turns "missing" from partial into wrong.
    So this visitor gets a refusal — and the thing worth pinning is that the refusal
    is free: the check runs before the split and before the gate.
    """
    import leasehound.metrics as metrics

    oversized = "1. RENT. Tenant shall pay rent when due.\n\n" * 12_000
    assert len(oversized) > scan.MAX_DOCUMENT_CHARS
    spent = []

    monkeypatch.setattr(app, "read_document", lambda path: oversized)
    monkeypatch.setattr(scan, "classify_document",
                        lambda *a, **k: spent.append("gate"))
    monkeypatch.setattr(scan, "scan_clauses", lambda *a, **k: spent.append("judge"))
    monkeypatch.setattr(scan, "check_protections", lambda *a, **k: spent.append("protections"))
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "scan_metrics.jsonl")
    app._scan_cache.clear()

    history: list = []
    frames = list(app.scan_flow(Path("all_my_documents.pdf"), "key-big", history, "", "", []))

    assert spent == [], f"an oversized upload must cost nothing, but ran {spent}"
    chat = " ".join(m["content"] for m in history)
    assert "too big for one scan" in chat
    assert "nothing was spent" in chat.lower()
    # Pages, not characters: the visitor is a renter, not an integrator.
    assert str(scan.MAX_DOCUMENT_CHARS) not in chat
    # No report pinned, and the composer is usable again. (_out: report is 3rd.)
    assert not any(frame[2] for frame in frames if isinstance(frame[2], str))
    # Nothing reached the metrics log either: a scan that never ran has no cost row,
    # and one here would quietly pollute the published cost-per-scan figure.
    assert not (tmp_path / "scan_metrics.jsonl").exists()
    assert app.HOUND_TRIPPED not in chat, "this is a known answer, not a crash"
    app._scan_cache.clear()


def test_the_served_app_bounds_upload_size(monkeypatch):
    """The byte-level half of the size bound, pinned because it is one keyword
    argument away from silently reverting.

    scan.MAX_DOCUMENT_CHARS is the bound that matters for spend, but it can only fire
    after read_document has parsed the upload — and parsing is the memory event, on a
    one-instance deployment measured at ~300 MB peak. Gradio's default is None, i.e.
    unlimited, so dropping this argument during a refactor restores an OOM path
    without failing anything else: the app still boots, every other test still passes,
    and the only symptom is a dead demo under an upload nobody sends on purpose.
    """
    passed = {}

    def fake_mount(api_app, blocks, **kwargs):
        # A FRESH app, because the real mount_gradio_app returns one and serve() then
        # adds middleware to it. Handing back the shared `api` instance makes this test
        # pass alone and fail in the suite: test_api.py has already started that app
        # through TestClient, and FastAPI refuses middleware on a started application.
        passed.update(kwargs)
        return FastAPI()

    monkeypatch.setattr(app.gr, "mount_gradio_app", fake_mount)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    app.serve()

    assert passed["max_file_size"] == app.MAX_UPLOAD_SIZE
    # One source of truth, in upload.py, because api.py enforces the same number on
    # its own route and neither module may import the other (app.py mounts api.py).
    # Two constants that must agree are two constants that will not.
    from leasehound.upload import MAX_UPLOAD_BYTES
    assert app.MAX_UPLOAD_SIZE == MAX_UPLOAD_BYTES
    # Looser than the character limit — they bound different things — but sized to the
    # 1 GiB instance, since Cloud Run charges filesystem bytes to memory and admits up
    # to 80 concurrent requests regardless of what Gradio's queue allows through.
    assert MAX_UPLOAD_BYTES == 8 * 1024 * 1024
    assert MAX_UPLOAD_BYTES * 80 < 1024**3, "80 concurrent uploads must fit in 1 GiB"
