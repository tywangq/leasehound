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
    second_out = list(app.scan_flow(Path("renamed_copy.md"), "key-b",
                                    second_history, "", "", []))

    assert api_calls["count"] == 1
    assert any(m["content"] == app.CACHED_SNIFF for m in second_history)
    log_lines = (tmp_path / "scan_metrics.jsonl").read_text().splitlines()
    assert json.loads(log_lines[1])["cache_hit"] is True
    # The name is deliberately NOT in the log any more: on this surface the document
    # belongs to a visitor and its file name is routinely their own name or address
    # (see metrics.UNNAMED_CLIENTS). This assertion used to read the log; the claim it
    # was making — a cache hit is attributed to the document actually uploaded, not to
    # the one that warmed the cache — is checked where it is visible to the person who
    # uploaded it, which is the report.
    assert json.loads(log_lines[1])["source"] == "upload.md"
    reports = [out[2] for out in second_out
               if isinstance(out[2], str) and "LeaseHound scan report" in out[2]]
    assert reports, "a cache hit must still pin a report"
    assert "renamed_copy.md" in reports[-1]
    assert "first_upload.md" not in reports[-1]
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
    assert "🌎" in report and "Judged against WA law" in report


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
    #
    # A refusal is a no-op on the right-hand side now — it neither pins a report nor
    # rewrites the context, because the panel may be showing a PREVIOUS lease's report and
    # that report is still true. So the assertion is about what no frame may DO, rather
    # than about a cleared value: nothing may claim a report for the document that was
    # refused.
    texts = [f[4] for f in frames if isinstance(f[4], str)]
    assert app.report_context("resume.pdf") not in texts
    assert all(t == app.LAW_ONLY_CONTEXT for t in texts)
    assert all(not f[3] for f in frames if isinstance(f[3], str)), "no state may be pinned"
    assert all(not f[2] for f in frames if isinstance(f[2], str)), "no report may be pinned"

    # And the override gets through: same document, same stubs, one word added.
    app._scan_cache.clear()
    override: list = []
    list(app.scan_flow(Path("resume.pdf"), "key", override, "", "", [], scan_anyway=True))
    assert judged["count"] == 1
    app._scan_cache.clear()


def test_the_stop_button_waits_for_the_gate(monkeypatch, tmp_path):
    """"Call off the hound" used to appear the moment a scan started — beside the previous
    lease's report and its three action buttons, so the screen offered to cancel something
    while showing the result of something else. It arrives in the same frame that clears
    them, which is also after the gate: a refused document never flashes a cancel button
    for a scan that will not happen.
    """
    import leasehound.metrics as metrics

    filler = "The parties agree to the terms set forth in this provision as written. "
    text = "Intro.\n\n" + "\n\n".join(f"{i}. HEADING. {filler}" for i in range(1, 4))

    def fake_scan_clauses(clauses, config, meter=None):
        for index, clause in enumerate(clauses, start=1):
            yield {"index": index, "clause": clause, "verdict": "green",
                   "citations": [], "urls": {}, "explanation": "fine"}

    monkeypatch.setattr(app, "read_document", lambda path: text)
    monkeypatch.setattr(scan, "classify_document", gate_returns())
    monkeypatch.setattr(scan, "scan_clauses", fake_scan_clauses)
    monkeypatch.setattr(scan, "check_protections", lambda clauses, meter=None: [])
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "scan_metrics.jsonl")
    app._scan_cache.clear()

    history: list = []
    frames = list(app.scan_flow(Path("new.md"), "k2", history, "# old report", "k1", []))
    # _out's positional contract: ..., stop is 7th, report 3rd, actions 10th.
    first_stop = next(i for i, f in enumerate(frames) if f[6] not in (None,) and
                      getattr(f[6], "get", lambda k, d=None: None)("visible") is True)
    assert isinstance(frames[first_stop][2], str), (
        "the frame that shows the stop button must also replace the old report")
    assert not any(getattr(f[6], "get", lambda k, d=None: None)("visible") is True
                   for f in frames[:first_stop]), "nothing may offer to cancel before the gate"
    app._scan_cache.clear()


def test_a_locked_pdf_gets_its_own_message_and_touches_nothing(monkeypatch, tmp_path):
    """The handler was in the wrong function and nobody noticed until a locked PDF hit the
    deployed app: read_document is called OUTSIDE the try that wraps the scan, so
    EncryptedDocument went to respond()'s generic "tripped over an error" instead.

    Also pins the other half: a file that never opened cannot cost the report on screen.
    """
    from leasehound.upload import EncryptedDocument

    def locked(path):
        raise EncryptedDocument(path.name)

    monkeypatch.setattr(app, "read_document", locked)
    history: list = []
    previous = "# LeaseHound scan report\n\nDocument: `first.md`"
    frames = list(app.scan_flow(Path("locked.pdf"), "key2", history, previous, "key1", []))

    assert any(app.LOCKED_PDF in (m.get("content") or "") for m in history)
    assert not any(app.SNIFF_STARTING in (m.get("content") or "") for m in history), (
        "nothing may promise a scan of a file that never opened")
    for frame in frames:
        for value in (frame[2], frame[3], frame[4], frame[5]):
            assert not isinstance(value, str), "a locked PDF must not touch the panel"


def test_a_refusal_leaves_an_existing_report_alone(monkeypatch, tmp_path):
    """Uploading a non-lease while a report is on screen used to cost you the report.

    The panel, the state, the context and the buttons were all cleared at the start of
    every scan, so a document that could not even be opened took the last good result with
    it. Nothing on the right is touched now until there is something to replace it with.
    """
    import leasehound.metrics as metrics

    monkeypatch.setattr(app, "read_document", lambda path: "Intro.\n\n1. HEADING. Text.")
    monkeypatch.setattr(scan, "classify_document", gate_returns("other"))
    monkeypatch.setattr(scan, "check_protections", lambda clauses, meter=None: [])
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "scan_metrics.jsonl")
    app._scan_cache.clear()

    history: list = []
    previous = "# LeaseHound scan report\n\nDocument: `first.md`"
    frames = list(app.scan_flow(Path("resume.pdf"), "key2", history, previous, "key1", []))

    for frame in frames:
        report, state, context, source = frame[2], frame[3], frame[4], frame[5]
        for value in (report, state, context, source):
            assert not isinstance(value, str) or value == app.LAW_ONLY_CONTEXT, (
                "a refusal must not overwrite the report on screen")
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


# --- the report file, and the ask path's size bound ------------------------------


def test_the_report_directory_does_not_grow_without_end(tmp_path, monkeypatch):
    """It used to make a fresh temp dir per finished scan and delete none of them.

    The report quotes the clauses it judged, so this is the one place in the app that
    really does write lease-derived content to disk — and on Cloud Run that is not a
    disk, it is a slice of the same 1 GiB the scan is running in.
    """
    monkeypatch.setattr(app, "REPORTS_ROOT", tmp_path / "reports")
    paths = [app.report_file(f"# report {i}", f"lease_{i}.pdf")
             for i in range(app.MAX_REPORT_FILES + 12)]
    assert len(list((tmp_path / "reports").iterdir())) == app.MAX_REPORT_FILES
    # The newest download still works; the oldest is the one that went.
    assert Path(paths[-1]).read_text() == f"# report {len(paths) - 1}"
    assert not Path(paths[0]).exists()


def test_two_visitors_uploading_the_same_file_name_get_different_reports(
        tmp_path, monkeypatch):
    """A subdirectory per report, not one directory of files. The file NAME is the
    visitor's own, so a shared directory would let two people who both uploaded
    `lease.pdf` hand each other a report."""
    monkeypatch.setattr(app, "REPORTS_ROOT", tmp_path / "reports")
    mine = app.report_file("# my clauses", "lease.pdf")
    theirs = app.report_file("# their clauses", "lease.pdf")
    assert mine != theirs
    assert Path(mine).read_text() == "# my clauses"
    assert Path(mine).name == Path(theirs).name == "scan_report_lease.md"


def test_a_question_too_long_to_answer_is_refused_in_chat_for_nothing():
    """No stubs, deliberately: if anything in the pipeline were reachable this test
    would try to make a real API call and fail loudly. The refusal runs ahead of the
    router, which is ask mode's first call."""
    from leasehound.answer import MAX_QUESTION_CHARS

    history: list = [{"role": "user", "content": "x"}]
    outs = list(app.answer_flow("y" * (MAX_QUESTION_CHARS + 1), history, "", []))
    assert outs, "the refusal has to reach the screen"
    assert history[-1]["content"].startswith("🐕 That's a lot to read")
    assert f"{MAX_QUESTION_CHARS:,}" in history[-1]["content"]
    assert "Nothing was spent" in history[-1]["content"]


def test_the_report_context_bound_is_the_pipelines_bound_not_a_second_opinion():
    """The same context had two limits 33x apart: this file trimmed a scan report to
    6,000 characters while /v1/ask accepted 200,000 and passed it through untouched."""
    from leasehound.answer import MAX_HISTORY_MESSAGE_CHARS

    assert app.REPORT_CONTEXT_CHARS == MAX_HISTORY_MESSAGE_CHARS
