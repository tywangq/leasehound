"""Gradio demo UI — one chat, artifact-style report panel.

A single multimodal input drives everything: attach a lease (or click the
sample-lease example) to scan it; just type to ask questions. Scan progress
streams into the chat clause by clause; the finished report pins to the
right-hand panel so it stays visible while you chat about it. The chat's
context is explicit — Washington law always, plus the latest scan report once
a scan has run.

Privacy: uploads are parsed in memory and the temp file is deleted right after
parsing; clause text goes to the OpenAI API for analysis and is never stored.

Usage:
    python -m leasehound.app     # http://localhost:7860
"""

import re
import tempfile
import traceback
from pathlib import Path

import gradio as gr

from leasehound.answer import answer_question
from leasehound.scan import (
    check_protections,
    count_verdicts,
    looks_like_lease,
    render_report,
    scan_clauses,
    scan_config,
)
from leasehound.upload import load_clauses

REPO_ROOT = Path(__file__).parent.parent
SAMPLE_LEASE = REPO_ROOT / "examples" / "sample_lease.md"
ICONS = Path(__file__).parent / "assets"

HERO = """
<div class="hero">
  <div class="wordmark">🐕 LeaseHound</div>
  <p class="tagline">Sniffs out the clauses that shouldn't be there.</p>
</div>
"""

CSS = """
.gradio-container {max-width: 1200px !important; margin: 0 auto !important;}
.hero {text-align: center; padding: 28px 0 8px;}
.hero .wordmark {font-size: 2.4em; font-weight: 700; letter-spacing: -0.5px;}
.hero .tagline {font-size: 1.25em; margin: 6px 0 4px;}
.step-sub {font-size: 0.9em; opacity: 0.75;}
.main-row {justify-content: center; align-items: stretch;}
/* Solo mode (no report column in the DOM): one centered chat column. With the
   report column present, both columns fall back to their equal flex scales.
   Transitions on the chat column plus a grow-in keyframe on the report column
   make the solo ↔ split switch a slide instead of a snap. */
.chat-col {transition: flex-basis 0.45s ease, flex-grow 0.45s ease;}
.main-row:not(:has(.report-col)) .chat-col {flex-grow: 0 !important; flex-basis: 760px !important; max-width: 760px;}
.main-row:has(.report-col) .chat-col {flex-basis: 0px !important;}
/* Verdict headers (🚩 Red / ⚠️ Yellow / ✅ Clear): body-sized bold, not big headings. */
.report-panel h1 {font-size: 1.15em; margin: 4px 0 10px;}
.report-panel h2 {font-size: 1em; font-weight: 700; margin: 12px 0 6px;}
/* The [copy][download][trash] row pins to the column's top-right corner and
   stays put while the report scrolls underneath. */
.report-col {position: relative;}
#report-actions {position: absolute; top: 6px; right: 6px; z-index: 10;
                 width: auto; min-width: 0; gap: 6px; flex-wrap: nowrap;}
/* Icon-only buttons: gradio gives button imgs max-width:100%, which resolves
   against the empty label's collapsed content box and squashes the icon to 0. */
#report-actions button {width: auto; min-width: 30px; flex-grow: 0; padding: 4px 8px;}
#report-actions button img {width: 13px; height: 13px; margin: 0;
                            max-width: none; flex-shrink: 0;}
/* Side-by-side mode: the report column stretches to the chat column's height
   and the report scrolls inside it, so both columns end flush. */
@media (min-width: 901px) {
  .report-col {display: flex !important; flex-direction: column;
               min-width: 0 !important; overflow: hidden;
               animation: report-in 0.5s ease;}
  @keyframes report-in {
    from {flex-grow: 0.001; opacity: 0;}
    to {flex-grow: 1; opacity: 1;}
  }
  .report-panel {flex: 1 1 0; min-height: 0; overflow-y: auto; padding: 0 4px;}
}
@media (max-width: 900px) {
  .report-panel {max-height: 65vh; overflow-y: auto; padding: 0 4px;}
}
.chat-col .multimodal-textbox .input-container {align-items: center !important;}
.chat-col .multimodal-textbox textarea {align-content: center;}
#example-scan button, #example-prompts button {text-align: left !important; justify-content: flex-start !important;}
/* The scan example renders its example_labels string in a bare div (14px by
   default); match the question examples' markdown <p> size. */
#example-scan .gallery-item {font-size: var(--text-lg, 16px);}
@media (max-width: 900px) {
  .main-row {flex-direction: column !important;}
  .main-row .chat-col {flex-grow: 1 !important; flex-basis: auto !important; max-width: 100% !important;}
}
footer {visibility: hidden;}
"""

LAW_ONLY_CONTEXT = (
    "🧠 The hound knows **Washington tenant law (RCW 59.18)**. "
    "Scan a lease and it will know your report too."
)
ALREADY_SNIFFED = "🐕 Already sniffed this one — the report is still on the right."
CALLED_OFF = "🐕 Called off — the hound stopped mid-sniff. Scan again whenever you're ready."
NOTHING_EXTRACTED = (
    "🐕 The hound couldn't find any text in this document — a scanned or photo PDF has "
    "no text layer. Try a text-based .pdf, .md, or .txt."
)
NOT_A_LEASE = (
    "🐕 The hound gave this document a good sniff, but it doesn't smell like a "
    "residential lease — and leases are all it scans. Attach a lease, or try the sample."
)
HOUND_TRIPPED = (
    "🐕 The hound tripped over an error and lost the scent — nothing was changed. "
    "Try again, or try a different file."
)
# The example chip shows the file name; once clicked, the file rides along as an
# attachment, so the message itself drops the "(sample_lease.md)" suffix.
SCAN_EXAMPLE = {
    "text": "Sniff this sample lease for red flags",
    "files": [str(SAMPLE_LEASE)],
}
SCAN_EXAMPLE_LABEL = "Sniff this sample lease for red flags (sample_lease.md)"
QUESTION_EXAMPLES = [
    {"text": "Can my landlord charge a late fee if rent is 3 days late?", "files": []},
    {"text": "How much notice before my landlord can enter?", "files": []},
    {"text": "When do I get my security deposit back after moving out?", "files": []},
]


def report_context(name: str) -> str:
    return f"🧠 The hound knows **Washington tenant law** + the **scan report of `{name}`**."


# No --- divider: markdown gives an <hr> generous margins on top of the
# paragraph gap, which reads as a hole between the answer and its sources.
FOOTER_MARK = "\n\n**Statutes cited in this answer:**"
# A model-written imitation of the footer (any trailing "Statutes cited …"
# block, however formatted) — stripped before the real footer is appended.
MODEL_FOOTER = re.compile(r"\n+(?:-{3,}\n+)?\**Statutes cited in this answer:?\**[\s\S]*\Z")


def sources_footer(answer: str, chunks) -> str:
    """List only the statutes the answer actually cites, not everything retrieved.

    An off-topic question still retrieves (irrelevant) chunks, and the answer
    honestly declines — the footer shouldn't then present those chunks as
    sources. Cited = mentioned in the answer AND present in the retrieved set.
    """
    url_by_section = {}
    for chunk in chunks:
        section = chunk.metadata.get("section")
        if section:
            url_by_section.setdefault(section, chunk.metadata.get("url", ""))
    seen, lines = set(), []
    for section in re.findall(r"RCW \d+\.\d+\.\d+", answer):
        if section in url_by_section and section not in seen:
            seen.add(section)
            url = url_by_section[section]
            lines.append(f"- [{section}]({url})" if url else f"- {section}")
    return FOOTER_MARK + "\n" + "\n".join(lines) if lines else ""


def strip_footer(message: dict) -> dict:
    """Drop the sources footer from an assistant message before it re-enters the LLM.

    The footer is display chrome. Fed back as conversation history, the model
    learns the pattern and writes its own copy at the end of the next answer —
    which then gets the real footer appended below it, doubling the block.
    """
    content = message.get("content")
    if message.get("role") == "assistant" and isinstance(content, str) and FOOTER_MARK in content:
        return {**message, "content": content.split(FOOTER_MARK)[0].rstrip()}
    return message


def cleanup_upload(path: Path) -> None:
    """Delete an uploaded temp file as soon as its text is extracted (privacy)."""
    if path != SAMPLE_LEASE and str(path).startswith(tempfile.gettempdir()):
        path.unlink(missing_ok=True)


def progress_line(done: int, total: int) -> str:
    return f"🐕 On the scent — {done}/{total} clauses sniffed…"


def report_file(report: str, source_name: str) -> str:
    """Write the report to a fresh temp dir so the download button can serve it."""
    folder = Path(tempfile.mkdtemp(prefix="leasehound_"))
    path = folder / f"scan_report_{Path(source_name).stem}.md"
    path.write_text(report, encoding="utf-8")
    return str(path)


# Every event yields the same 10 outputs. `col` shows/hides the report column, so
# the page is a single centered chat until there is something to pin. The input
# box is disabled while a scan runs — a submit queued mid-scan would carry a
# pre-scan snapshot of the cache state and silently rescan. `actions` is the
# [copy][download][trash] row: one switch so the three buttons always appear
# together, only once a finished report is pinned. `download` carries the file.
# [chatbot, message_box, report_output, report_state, context_line,
#  scanned_source, stop_button, report_col, download_button, report_actions]


def _out(history=gr.skip(), box=gr.skip(), report=gr.skip(), state=gr.skip(),
         context=gr.skip(), source=gr.skip(), stop=gr.skip(), col=gr.skip(),
         download=gr.skip(), actions=gr.skip()):
    return history, box, report, state, context, source, stop, col, download, actions


def answer_flow(question, history, report, context_base):
    """Append the hound's answer to an already-appended user message."""
    history.append({"role": "assistant", "content": "🐕 Thinking…"})
    yield _out(history)
    # Trim BEFORE prepending, so the report survives; strip footers so the
    # model doesn't mimic them (see strip_footer).
    context_history = [strip_footer(m) for m in list(context_base)[-10:]]
    if report:
        context_history = [
            {
                "role": "user",
                "content": "For context, here is the scan report of my lease:\n\n" + report[:6000],
            },
            {
                "role": "assistant",
                "content": "Got it — I'll answer your questions with your scan report in mind.",
            },
        ] + context_history
    stream, chunks = answer_question(question, context_history)
    answer = ""
    for delta in stream:
        answer += delta
        history[-1]["content"] = answer
        yield _out(history)
    answer = MODEL_FOOTER.sub("", answer).rstrip()
    history[-1]["content"] = answer + sources_footer(answer, chunks)
    yield _out(history)


def scan_flow(path, key, history, report, scanned, context_base, question=""):
    """Scan a lease with per-clause progress in chat; report lands in the side panel."""
    name = path.name
    if scanned == key and report:
        history.append({"role": "assistant", "content": ALREADY_SNIFFED})
        yield _out(history, report=report, state=report,
                   context=report_context(name), source=key,
                   stop=gr.update(visible=False), col=gr.update(visible=True),
                   download=gr.update(value=report_file(report, name)),
                   actions=gr.update(visible=True))
        if question:
            yield from answer_flow(question, history, report, context_base)
        return

    clauses = load_clauses(path)
    cleanup_upload(path)
    if not clauses:
        history.append({"role": "assistant", "content": NOTHING_EXTRACTED})
        yield _out(history, stop=gr.update(visible=False))
        return
    if not looks_like_lease(clauses):
        history.append({"role": "assistant", "content": NOT_A_LEASE})
        yield _out(history, stop=gr.update(visible=False))
        return

    total = len(clauses)
    config = scan_config("wa")
    history.append({"role": "assistant", "content": progress_line(0, total)})
    # A fresh scan makes any previous report stale immediately: law-only context
    # until the new report lands.
    yield _out(history, box=gr.update(interactive=False),
               report=f"🐕 Sniffing `{name}` — the report will appear here.",
               state="", context=LAW_ONLY_CONTEXT, source="",
               stop=gr.update(visible=True), col=gr.update(visible=True),
               actions=gr.update(visible=False))
    findings = []
    for finding in scan_clauses(clauses, config):
        findings.append(finding)
        history[-1]["content"] = progress_line(len(findings), total)
        yield _out(history)
    findings.sort(key=lambda f: f["index"])

    history[-1]["content"] = "🐕 One more pass — checking the protections the law requires…"
    yield _out(history)
    protections = check_protections(clauses)
    new_report = render_report(findings, name, "wa", protections)
    counts = count_verdicts(findings)
    missing = sum(1 for p in protections if p["status"] == "missing")
    history[-1]["content"] = (
        f"🐕 Sniff complete: 🚩 {counts['red']} red · ⚠️ {counts['yellow']} caution · "
        f"✅ {counts['green']} clear · 🔍 {missing} missing protections. "
        "Full report on the right — ask me about any clause."
    )
    yield _out(history, box=gr.update(interactive=True),
               report=new_report, state=new_report,
               context=report_context(name), source=key,
               stop=gr.update(visible=False), col=gr.update(visible=True),
               download=gr.update(value=report_file(new_report, name)),
               actions=gr.update(visible=True))
    if question:
        yield from answer_flow(question, history, new_report, context_base)


def respond(message, history, report, scanned):
    """Guard around the real handler: an uncaught exception inside a generator
    event dies silently — the traceback goes to the server log and the user
    sees nothing happen at all (how a missing PDF dependency shipped). Turn any
    failure into a hound apology and restore the pre-turn panel state."""
    history = list(history)
    try:
        yield from _respond(message, history, report, scanned)
    except Exception:
        traceback.print_exc()
        history.append({"role": "assistant", "content": HOUND_TRIPPED})
        name = SAMPLE_LEASE.name if scanned == "sample" else Path(scanned).name
        yield _out(history, box=gr.update(interactive=True),
                   report=report, state=report,
                   context=report_context(name) if report else LAW_ONLY_CONTEXT,
                   source=scanned, stop=gr.update(visible=False),
                   col=gr.update(visible=bool(report)),
                   actions=gr.update(visible=bool(report)))


def _respond(message, history, report, scanned):
    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    if not text and not files:
        yield _out(box=gr.update(value=None))
        return
    context_base = list(history)  # snapshot before this turn's messages
    if files:
        path = Path(files[0])
        if path.name == SAMPLE_LEASE.name:
            # The sample-lease example: scan the repo file itself and cache as "sample".
            path, key = SAMPLE_LEASE, "sample"
        else:
            key = str(path)
        user_content = f"📎 `{path.name}`" + (f"\n\n{text}" if text else "")
        history.append({"role": "user", "content": user_content})
        yield _out(history, box=gr.update(value=None))
        # The canned example text is an instruction the scan itself fulfills —
        # don't answer it again as a follow-up question afterwards.
        question = "" if text == SCAN_EXAMPLE["text"] else text
        yield from scan_flow(path, key, history, report, scanned, context_base, question=question)
    else:
        history.append({"role": "user", "content": text})
        yield _out(history, box=gr.update(value=None))
        yield from answer_flow(text, history, report, context_base)


def on_stop(history):
    history = list(history)
    if history and history[-1]["role"] == "assistant":
        history[-1]["content"] = CALLED_OFF
    # The report was already invalidated at scan start, so the panel column
    # disappears again rather than showing a stale "scanning…" line.
    return (history, gr.update(interactive=True), "", gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False))


def on_trash():
    """Discard the pinned report: back to a law-only, solo-chat page.

    Silent, matching the chatbot's built-in clear button — the column
    collapsing and the context line reverting are feedback enough. (A chat
    message would merge into the previous assistant bubble and pollute the
    answering context; a toast would be this app's only toast.)
    """
    return ("", "", LAW_ONLY_CONTEXT, "", gr.update(visible=False),
            gr.update(visible=False))


# Copy is client-side only (clipboard + a brief ✓ flash on the button) — no
# server round-trip. The js handler must return a list (Gradio maps it to outputs).
COPY_JS = """
(report) => {
    navigator.clipboard.writeText(report);
    const button = document.querySelector('#report-actions button');
    const check = document.createElement('span');
    check.textContent = '✓';
    button.appendChild(check);
    setTimeout(() => check.remove(), 1200);
    return [];
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="teal",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ),
    css=CSS,
    title="LeaseHound — lease red-flag scanner",
) as demo:
    gr.HTML(HERO)
    report_state = gr.State("")
    scanned_source = gr.State("")  # what the current report is for: a file path or "sample"

    with gr.Row(elem_classes="main-row"):
        with gr.Column(scale=1, elem_classes="chat-col"):
            context_line = gr.Markdown(LAW_ONLY_CONTEXT, elem_classes="step-sub")
            chatbot = gr.Chatbot(height=440, type="messages", show_label=False)
            message_box = gr.MultimodalTextbox(
                placeholder="Attach a lease, or ask about renting in Washington…",
                show_label=False,
                file_types=[".pdf", ".md", ".txt"],
                file_count="single",
            )
            gr.Examples(
                examples=[SCAN_EXAMPLE],
                example_labels=[SCAN_EXAMPLE_LABEL],
                inputs=message_box,
                label="Scan a lease",
                elem_id="example-scan",
            )
            gr.Examples(
                examples=QUESTION_EXAMPLES,
                inputs=message_box,
                label="Ask a question",
                elem_id="example-prompts",
            )
        with gr.Column(scale=1, visible=False, elem_classes="report-col") as report_col:
            stop_button = gr.Button(
                "✋ Call off the hound", variant="stop", size="sm", visible=False
            )
            # One row pinned at the panel's top-right corner (see the CSS), so
            # all three actions share the same position and appear together,
            # only once a finished report is up.
            with gr.Row(visible=False, elem_id="report-actions") as report_actions:
                copy_button = gr.Button("", icon=str(ICONS / "copy.svg"), size="sm")
                download_button = gr.DownloadButton("", icon=str(ICONS / "download.svg"), size="sm")
                trash_button = gr.Button("", icon=str(ICONS / "trash.svg"), size="sm")
            report_output = gr.Markdown("", elem_classes="report-panel")

    outs = [chatbot, message_box, report_output, report_state,
            context_line, scanned_source, stop_button, report_col, download_button,
            report_actions]
    submit_event = message_box.submit(
        respond, inputs=[message_box, chatbot, report_state, scanned_source], outputs=outs,
        show_progress="hidden",
    )
    stop_button.click(
        on_stop, inputs=[chatbot],
        outputs=[chatbot, message_box, report_output, report_col, stop_button,
                 report_actions],
        cancels=[submit_event],
    )
    trash_button.click(
        on_trash,
        outputs=[report_output, report_state, context_line,
                 scanned_source, report_col, report_actions],
        show_progress="hidden",
    )
    # Input is the Markdown component, not report_state: a js-only handler runs
    # entirely client-side, and gr.State values only exist on the server.
    copy_button.click(None, inputs=[report_output], js=COPY_JS)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
