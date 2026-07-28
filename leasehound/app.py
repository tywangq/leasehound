"""Gradio demo UI — one chat, artifact-style report panel.

A single multimodal input drives everything: attach a lease (or click the
sample-lease example) to scan it; just type to ask questions. Scan progress
streams into the chat clause by clause; the finished report pins to the
right-hand panel so it stays visible while you chat about it. The chat's
context is explicit — Washington law always, plus the latest scan report once
a scan has run.

Privacy: uploads are parsed in memory and the temp file is deleted right after
parsing; clause text goes to the OpenAI API for analysis and is never written
to disk. Finished scans are cached in process memory, keyed by document
content hash, so rescanning an identical document costs zero API calls.

Usage:
    python -m leasehound.app     # http://localhost:7860
"""

import hashlib
import re
import tempfile
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import gradio as gr

from leasehound.answer import answer_question
from leasehound.metrics import ScanMeter, log_scan
from leasehound.scan import (
    MAX_CLAUSES,
    check_protections,
    count_verdicts,
    looks_like_lease,
    render_report,
    scan_clauses,
    scan_config,
)
from leasehound.upload import read_document, split_clauses_with_mode

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
/* Desktop layout. Both columns get explicit widths (flex-grow stays 0) so the
   solo ↔ split switch is a deterministic slide: animating flex-grow instead
   makes widths depend on the grow *ratio*, which overshoots — the chat column
   balloons to full row width the instant its sibling leaves the DOM, then
   shrinks back. flex-basis is not !important so the entry keyframe can drive
   it (CSS animations lose to !important declarations). */
@media (min-width: 901px) {
  .chat-col {transition: flex-basis 0.45s ease;}
  .main-row:not(:has(.report-col)) .chat-col {flex-grow: 0 !important; flex-basis: 760px; max-width: 760px;}
  .main-row:has(.report-col) .chat-col {flex-grow: 0 !important; flex-basis: calc(50% - (var(--layout-gap, 8px) / 2));}
  /* Side-by-side, the report column stretches to the chat column's height and
     the report scrolls inside it, so both columns end flush. */
  .report-col {display: flex !important; flex-direction: column;
               flex-grow: 0 !important;
               flex-basis: calc(50% - (var(--layout-gap, 8px) / 2));
               min-width: 0 !important; overflow: hidden;
               transition: flex-basis 0.45s ease, opacity 0.45s ease;
               animation: report-in 0.45s ease;}
  @keyframes report-in {from {flex-basis: 0px; opacity: 0;}}
  .report-panel {flex: 1 1 0; min-height: 0; overflow-y: auto; padding: 0 4px;}
  /* Trash: TRASH_JS adds .closing so both columns slide to their solo
     positions *before* the server removes the report column from the DOM
     (on_trash sleeps past the transition). Must come after the :has rules —
     same specificity, so source order decides. */
  .main-row.closing .chat-col {flex-basis: 760px;}
  .main-row.closing .report-col {flex-basis: 0px; opacity: 0;}
}
@media (max-width: 900px) {
  .report-panel {max-height: 65vh; overflow-y: auto; padding: 0 4px;}
}
.chat-col .multimodal-textbox .input-container {align-items: center !important;}
.chat-col .multimodal-textbox textarea {align-content: center;}
/* Attached-file chip: gradio's 48px thumbnail dwarfs the 30px attach/send
   icons beside it. Direct-child only, so the delete ✕ keeps its own size. */
.chat-col .multimodal-textbox .thumbnail-item {width: 30px !important; height: 30px !important;}
.chat-col .multimodal-textbox .thumbnail-item > :is(svg, img) {width: 16px !important; height: 16px !important;}
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
ALREADY_SNIFFED = "🐕 Already sniffed this one — the report is still on the right (below on narrow screens)."
CACHED_SNIFF = (
    "🐕 The hound has sniffed this exact lease before — here's the saved report, "
    "no fresh API calls. Attach a different lease to watch a live scan."
)
CALLED_OFF = "🐕 Called off — the hound stopped mid-sniff. Scan again whenever you're ready."
NOTHING_EXTRACTED = (
    "🐕 The hound couldn't find any text in this document — a scanned or photo PDF has "
    "no text layer. Try a text-based .pdf, .md, or .txt."
)
NOT_A_LEASE = (
    "🐕 The hound gave this document a good sniff, but it doesn't smell like a "
    "residential lease — and leases are all it scans. Attach a lease, or try the sample."
)
TOO_MANY_CLAUSES = (
    "🐕 This document splits into {count} clauses — no residential lease is that long, "
    "so the hound stops at {limit}. If it really is a lease, try attaching just the lease body."
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
# "Statutes cited:" — right under the answer, "in this answer" says nothing.
FOOTER_MARK = "\n\n**Statutes cited:**"
# A model-written imitation of the footer (any trailing "Statutes cited …"
# block, however formatted or worded) — stripped before the real one is appended.
MODEL_FOOTER = re.compile(
    r"\n+(?:-{3,}\n+)?\**Statutes cited(?: in this answer)?:?\**[\s\S]*\Z"
)


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


# Cross-visitor scan cache, keyed by document CONTENT — not upload path, which
# is unique per upload, and not session, which is one browser tab. Most demo
# traffic scans the one sample lease; only its first scan after a cold start
# should cost API calls. Memory only (never disk), bounded, gone with the
# instance — the next first scan re-warms it. Stores findings + protections
# rather than the rendered report so a hit renders under its own file name.
CACHE_MAX_ENTRIES = 32
_scan_cache: OrderedDict[str, dict] = OrderedDict()
_scan_cache_lock = threading.Lock()


def cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def cache_get(digest: str) -> dict | None:
    with _scan_cache_lock:
        entry = _scan_cache.get(digest)
        if entry is not None:
            _scan_cache.move_to_end(digest)
        return entry


def cache_put(digest: str, entry: dict) -> None:
    with _scan_cache_lock:
        _scan_cache[digest] = entry
        _scan_cache.move_to_end(digest)
        while len(_scan_cache) > CACHE_MAX_ENTRIES:
            _scan_cache.popitem(last=False)


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

    text = read_document(path)
    cleanup_upload(path)
    clauses, split_mode = split_clauses_with_mode(text)
    if not clauses:
        history.append({"role": "assistant", "content": NOTHING_EXTRACTED})
        yield _out(history, stop=gr.update(visible=False))
        return
    if len(clauses) > MAX_CLAUSES:
        message = TOO_MANY_CLAUSES.format(count=len(clauses), limit=MAX_CLAUSES)
        history.append({"role": "assistant", "content": message})
        yield _out(history, stop=gr.update(visible=False))
        return

    digest = cache_key(text)
    cached = cache_get(digest)
    if cached:
        findings, protections = cached["findings"], cached["protections"]
        new_report = render_report(findings, name, "wa", protections)
        counts = count_verdicts(findings)
        missing = sum(1 for p in protections if p["status"] == "missing")
        log_scan(ScanMeter(), name, len(clauses), verdicts=counts, missing=missing,
                 split_mode=split_mode, cache_hit=True)
        history.append({"role": "assistant", "content": CACHED_SNIFF})
        yield _out(history, report=new_report, state=new_report,
                   context=report_context(name), source=key,
                   stop=gr.update(visible=False), col=gr.update(visible=True),
                   download=gr.update(value=report_file(new_report, name)),
                   actions=gr.update(visible=True))
        if question:
            yield from answer_flow(question, history, new_report, context_base)
        return

    meter = ScanMeter()  # the sanity check below is the scan's first API call
    if not looks_like_lease(clauses, meter):
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
    for finding in scan_clauses(clauses, config, meter):
        findings.append(finding)
        history[-1]["content"] = progress_line(len(findings), total)
        yield _out(history)
    findings.sort(key=lambda f: f["index"])

    history[-1]["content"] = "🐕 One more pass — checking the protections the law requires…"
    yield _out(history)
    protections = check_protections(clauses, meter)
    new_report = render_report(findings, name, "wa", protections)
    counts = count_verdicts(findings)
    missing = sum(1 for p in protections if p["status"] == "missing")
    log_scan(meter, name, total, verdicts=counts, missing=missing, split_mode=split_mode)
    cache_put(digest, {"findings": findings, "protections": protections})
    history[-1]["content"] = (
        f"🐕 Sniff complete: 🚩 {counts['red']} red · ⚠️ {counts['yellow']} caution · "
        f"✅ {counts['green']} clear · 🔍 {missing} missing protections. "
        "Full report on the right (below on narrow screens) — ask me about any clause."
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
    # TRASH_JS is sliding both columns to their solo positions right now; wait
    # it out so the DOM removal lands after the columns have stopped moving.
    time.sleep(0.5)
    return ("", "", LAW_ONLY_CONTEXT, "", gr.update(visible=False),
            gr.update(visible=False))


# Copy is client-side only — no server round-trip. Feedback: the copy icon
# itself becomes a checkmark in place for a moment (swapping the img src to a
# data URI; a check.svg on disk wouldn't be on Gradio's allowed-files list).
# The js handler must return a list (Gradio maps it to outputs).
CHECK_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="#6b7280" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<polyline points="20 6 9 17 4 12"/></svg>'
)
COPY_JS = f"""
(report) => {{
    navigator.clipboard.writeText(report);
    const img = document.querySelector('#report-actions button img');
    if (img) {{
        if (!img.dataset.copyIcon) img.dataset.copyIcon = img.src;
        img.src = 'data:image/svg+xml;utf8,' + encodeURIComponent({CHECK_ICON!r});
        setTimeout(() => {{ img.src = img.dataset.copyIcon; }}, 1200);
    }}
    return [];
}}
"""

# Examples send on click — one click, one message, like example chips in chat
# apps; they're complete questions, there's nothing to edit first. Gradio's
# Examples component only fills the box (run_on_click can't pass the state
# inputs respond needs), so a page-level listener waits for the fill to land
# and clicks send. The scan example also attaches a file — wait for its chip,
# or the submit would race the attachment and send text alone.
EXAMPLES_JS = """
() => {
    // Put the cursor in the message box on load, so a visitor can start typing
    // without clicking first. Deliberately not Gradio's autofocus=True: that
    // scrolls the focused input into view, measured at 713px down, which pushes
    // the wordmark and the one-click scan example off the top of the page.
    // preventScroll buys the focus without the jump.
    let focusTries = 0;
    const focusBox = setInterval(() => {
        const input = document.querySelector('.chat-col .multimodal-textbox textarea');
        if (input) {
            input.focus({preventScroll: true});
            clearInterval(focusBox);
        } else if (++focusTries > 40) {
            clearInterval(focusBox);
        }
    }, 50);

    document.addEventListener('click', (event) => {
        const chip = event.target.closest('#example-scan button, #example-prompts button');
        if (!chip) return;
        const needsFile = !!chip.closest('#example-scan');
        let tries = 0;
        const timer = setInterval(() => {
            const box = document.querySelector('.chat-col .multimodal-textbox');
            const filled = box.querySelector('textarea').value.trim()
                && (!needsFile || box.querySelector('.thumbnail-item'));
            if (filled) {
                clearInterval(timer);
                box.querySelector('.submit-button').click();
            } else if (++tries > 60) {
                clearInterval(timer);
            }
        }, 50);
    });
}
"""

# Starts the two-columns-to-one slide immediately on click; on_trash sleeps
# past the transition before removing the report column from the DOM. The
# class comes off on a timer (the column is long gone by then) so a future
# report isn't born collapsed.
TRASH_JS = """
() => {
    const row = document.querySelector('.main-row');
    if (row) {
        row.classList.add('closing');
        setTimeout(() => row.classList.remove('closing'), 700);
    }
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
    js=EXAMPLES_JS,
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
                # Cursor lands here on load: asking is the zero-friction entry
                # point, and the scan button is one click away either way.
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
        js=TRASH_JS,
    )
    # Input is the Markdown component, not report_state: a js-only handler runs
    # entirely client-side, and gr.State values only exist on the server.
    copy_button.click(None, inputs=[report_output], js=COPY_JS)

# Public-hosting guardrails: a bounded waiting room instead of an unbounded
# queue, and a few concurrent turns so one long scan doesn't serialize everyone.
demo.queue(max_size=16, default_concurrency_limit=4)

if __name__ == "__main__":
    # No explicit host/port: gradio falls back to GRADIO_SERVER_NAME /
    # GRADIO_SERVER_PORT (set by the container), else 127.0.0.1:7860 locally.
    demo.launch(inbrowser=False)
