"""Ask mode's deterministic parts: prompt assembly, stream unwrapping, metering.

The models are stubbed rather than called — what is testable without an API is
the context the model receives (where grounding either holds or quietly breaks)
and the accounting around it (where the published cost figure broke).
"""

import json
from types import SimpleNamespace

import leasehound.answer as answer
import leasehound.metrics as metrics
from leasehound.answer import SYSTEM_PROMPT, make_messages, stream_text
from leasehound.retrieval import Result


def chunk(section: str, text: str = "Statute text.", url: str = "") -> Result:
    return Result(
        page_content=text,
        metadata={"section": section, "url": url or f"https://law/{section}"},
    )


def test_system_prompt_carries_the_retrieved_statutes_and_nothing_else():
    messages = make_messages("Can they keep my deposit?", [], [chunk("RCW 59.18.260")])
    system = messages[0]
    assert system["role"] == "system"
    assert "RCW 59.18.260" in system["content"]
    assert "Statute text." in system["content"]
    # The question is the user turn, not folded into the system prompt.
    assert messages[-1] == {"role": "user", "content": "Can they keep my deposit?"}


def test_every_chunk_reaches_the_prompt_with_its_section_and_url():
    chunks = [chunk("RCW 59.18.150", "Entry notice."), chunk("RCW 59.18.230", "Waivers.")]
    context = make_messages("q", [], chunks)[0]["content"]
    for c in chunks:
        assert c.metadata["section"] in context
        assert c.metadata["url"] in context
        assert c.page_content in context


def test_history_sits_between_the_system_prompt_and_the_new_question():
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    messages = make_messages("new question", history, [chunk("RCW 59.18.060")])
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1:3] == history


def test_no_retrieved_chunks_still_produces_a_well_formed_prompt():
    # An off-topic question can retrieve nothing; the prompt must stay valid so
    # the model declines rather than the call erroring out.
    messages = make_messages("what's the weather?", None, [])
    assert messages[0]["role"] == "system"
    assert len(messages) == 2


def test_prompt_forbids_ungrounded_answers_and_a_model_written_sources_list():
    # Both rules are load-bearing: the first is the grounding contract, the
    # second keeps the model from duplicating the footer the app appends.
    assert "never guess at law" in SYSTEM_PROMPT
    assert "never append a" in SYSTEM_PROMPT


def delta(content):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


def test_stream_text_yields_content_and_skips_empty_deltas():
    # Real streams open with a role-only delta whose content is None, and can
    # carry None mid-stream; passing those through would crash the caller's join.
    stream = [delta(None), delta("Under "), delta("RCW 59.18.230"), delta(None), delta(".")]
    assert "".join(stream_text(stream)) == "Under RCW 59.18.230."


def usage_chunk(prompt=800, completion=120):
    """The extra final chunk include_usage adds: token totals, and NO choices."""
    return SimpleNamespace(choices=[], usage=SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details=None))


def priced(monkeypatch, per_call=0.001):
    monkeypatch.setattr(metrics, "completion_cost", lambda completion_response: per_call)
    monkeypatch.setattr(metrics, "cost_per_token", lambda **kwargs: (per_call, 0.0))


def test_the_usage_only_final_chunk_carries_no_choices_to_index(monkeypatch):
    # Asking for usage on a stream buys one chunk with an EMPTY choices list. The
    # unguarded delta lookup this replaced would have raised IndexError on the last
    # chunk of every answer — i.e. metering ask mode would have broken ask mode.
    priced(monkeypatch)
    meter = metrics.UsageMeter()
    text = "".join(stream_text([delta("Answer."), usage_chunk()], meter, "m"))
    assert text == "Answer."
    assert meter.summary()["llm_calls"] == 1


def test_usage_is_booked_once_even_if_two_chunks_carry_it(monkeypatch):
    # Some providers attach usage to a content chunk as well as the final one.
    # Booking both would inflate the mode's cost by a whole call per answer.
    priced(monkeypatch)
    meter = metrics.UsageMeter()
    carrying = delta("Answer.")
    carrying.usage = SimpleNamespace(prompt_tokens=800, completion_tokens=120,
                                     prompt_tokens_details=None)
    "".join(stream_text([carrying, usage_chunk()], meter, "m"))
    assert meter.summary()["llm_calls"] == 1


def route(category="legal_question", jurisdiction="unknown"):
    """A stubbed router response. Both fields, because the router answers both.

    Faking only `category` made four tests spend four minutes each retrying a
    pydantic ValidationError when `jurisdiction` was added to the response model —
    llm_retry cannot tell a schema mismatch from a flaky provider. A stub of a
    structured response has to be the whole structure.
    """
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"category": category, "jurisdiction": jurisdiction})))],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=5,
                              prompt_tokens_details=None),
    )


def stub_ask(monkeypatch, tmp_path, category="legal_question", chunks=None,
             jurisdiction="unknown"):
    """Wire answer_question to stubs and return the list of calls it makes.

    Nothing here raises: answer_question is wrapped in llm_retry, so an assertion
    thrown from inside a stub would be retried for four minutes before failing.
    Calls are recorded and checked afterwards instead.
    """
    priced(monkeypatch)
    monkeypatch.setattr(metrics, "ASK_LOG_PATH", tmp_path / "ask_metrics.jsonl")
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([delta("Under "), delta("RCW 59.18.230."), usage_chunk()])
        return route(category, jurisdiction)

    monkeypatch.setattr(answer, "completion", fake_completion)
    monkeypatch.setattr(answer, "fetch_context",
                        lambda q, config, history=None, meter=None:
                        chunks if chunks is not None else [chunk("RCW 59.18.230")])
    return calls


def test_the_router_call_is_metered_not_just_the_retrieval_stages(monkeypatch, tmp_path):
    """The guard on the bug that made the published ask-mode cost too low.

    The measurement script reached past answer_question straight into
    fetch_context, so the router — which classifies every incoming message and
    therefore runs on every question — was absent from the totals while the
    docstring claimed a single known deviation from production. Metering the real
    entry point is what makes the router impossible to leave out.
    """
    stub_ask(monkeypatch, tmp_path)
    result = answer.answer_question("can they keep my deposit?")

    assert result.routed is True
    assert result.record is None, "usage arrives with the stream's last chunk, not before"
    assert "".join(result.stream) == "Under RCW 59.18.230."
    # The router, then the answer. fetch_context is stubbed here, so its own stages
    # are counted by its own tests — what matters is that the router is included.
    assert result.record["llm_calls"] == 2
    assert result.record["routed_to_retrieval"] is True
    assert result.record["retrieved_chunks"] == 1


def test_chitchat_is_one_router_call_plus_one_reply_and_no_retrieval(monkeypatch, tmp_path):
    stub_ask(monkeypatch, tmp_path, category="small_talk")
    result = answer.answer_question("hi")
    assert result.routed is False
    assert result.chunks == [], "no statutes, so the caller skips the sources footer"
    "".join(result.stream)
    assert result.record["llm_calls"] == 2
    assert result.record["routed_to_retrieval"] is False


def test_the_router_reports_the_state_the_asker_named_for_no_extra_call(
        monkeypatch, tmp_path):
    """Ask mode had the hole scan mode just closed: the corpus is Washington's, and a
    renter in Oregon describing their problem got RCW 59.18 with nothing saying it does
    not govern them. The router already classifies every message, so the answer rides
    along on a call that was being made anyway."""
    calls = stub_ask(monkeypatch, tmp_path, jurisdiction="or")
    result = answer.answer_question("I rent in Portland, Oregon — can they keep my deposit?")
    assert result.jurisdiction == "or"
    "".join(result.stream)
    # Two calls: the router and the answer. Naming a state costs neither a third.
    assert result.record["llm_calls"] == 2
    assert len([c for c in calls if not c.get("stream")]) == 1
    # And the log knows, because a demo answering Oregon renters with Washington law
    # is a fact about the demo.
    assert result.record["jurisdiction"] == "or"


def test_a_question_naming_no_state_is_not_a_mismatch(monkeypatch, tmp_path):
    """The common case, and the one that decides whether the warning is worth having:
    most questions never mention a state, and a warning on all of them is noise."""
    stub_ask(monkeypatch, tmp_path, jurisdiction="unknown")
    result = answer.answer_question("can my landlord charge a late fee?")
    "".join(result.stream)
    assert result.jurisdiction == "unknown"
    assert "jurisdiction" not in result.record

    stub_ask(monkeypatch, tmp_path, jurisdiction="wa")
    named_wa = answer.answer_question("in Seattle, can my landlord enter without notice?")
    "".join(named_wa.stream)
    assert "jurisdiction" not in named_wa.record, "naming Washington is not a mismatch"


def test_the_answer_call_asks_for_usage_or_nothing_would_be_metered(monkeypatch, tmp_path):
    calls = stub_ask(monkeypatch, tmp_path)
    "".join(answer.answer_question("q").stream)
    streamed = [c for c in calls if c.get("stream")]
    assert len(streamed) == 1
    assert streamed[0]["stream_options"] == {"include_usage": True}


def test_abandoning_the_stream_still_logs_what_was_already_spent(monkeypatch, tmp_path):
    # The retrieval calls were paid for whether or not anyone read the answer, so a
    # visitor who closes the tab must not make that spend invisible.
    stub_ask(monkeypatch, tmp_path)
    result = answer.answer_question("q")
    next(result.stream)
    result.stream.close()
    assert result.record is not None
    assert result.record["llm_calls"] >= 1
    assert (tmp_path / "ask_metrics.jsonl").read_text().strip()
