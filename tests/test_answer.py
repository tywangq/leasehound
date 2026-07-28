"""Ask mode's deterministic parts: prompt assembly and stream unwrapping.

The router and the generation call need the API, so they aren't tested here —
what is testable without one is the context the model receives, which is where
grounding either holds or quietly breaks.
"""

from types import SimpleNamespace

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
