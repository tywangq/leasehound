"""App helpers: cited-only sources footer, upload cleanup (privacy), scan cache."""

import json
from pathlib import Path

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

    assert len(judged) == app.MAX_CLAUSES, "spend must stay bounded by the cap"
    # The negative-space pass is exempt: "this lease omits X" needs the whole document.
    assert len(protections_saw) == total
    chat = " ".join(m["content"] for m in history)
    assert f"clauses {app.MAX_CLAUSES + 1}–{total}" in chat, "the visitor is told what was skipped"
    assert app.HOUND_TRIPPED not in chat
    logged = json.loads((tmp_path / "scan_metrics.jsonl").read_text().splitlines()[0])
    assert logged["clauses"] == app.MAX_CLAUSES and logged["clauses_total"] == total
    app._scan_cache.clear()
