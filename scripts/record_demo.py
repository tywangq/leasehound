"""Re-record docs/demo.gif and docs/screenshot.png by driving the real UI.

These two files are the first thing a visitor sees, and they had gone stale: they
were recorded before the gate warning, the partial-scan notice, the cache-hit
message and the "across clauses 1-N of M" coverage line existed, so the README's
own illustration showed a product that no longer matched the text beside it. The
reason they went stale is that re-recording them was a manual chore nobody was
going to repeat, which is the same failure this project keeps fixing elsewhere:
an artifact that cannot be regenerated will eventually be wrong.

So they are generated, by driving the actual app with Playwright and waiting on
the app's own strings rather than on sleeps. Frames come from the same run, so
the GIF and the still cannot disagree with each other.

The GIF is paced by the wall clock, with three exceptions and one addition, all
for the reader: it opens on a held frame of the untouched page, holds the click
and the finished report, and draws the mouse pointer back in (a screenshot never
contains it, so a click looked like the page changing by itself).

Playwright is a DEV-ONLY tool and is deliberately absent from pyproject and
requirements-lock.txt — nothing that ships needs a browser. Install it just for
this:

    .venv/bin/python -m pip install playwright && .venv/bin/playwright install chromium

Then, with the app running (`python -m leasehound.app`):

    python -m scripts.record_demo              # both assets, one live scan
    python -m scripts.record_demo --still-only # just the screenshot

One live scan and one question, so a full recording costs about $0.018. The scan
cache is keyed by document content, so a second recording against the same server
takes the cache-hit path and the GIF has no clause-by-clause progress in it at
all. That is a silent failure — the recording succeeds and the asset is wrong — so
this refuses to write a GIF it caught on the cached path, and says to restart.
"""

import argparse
import io
import time
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageDraw

DOCS = Path(__file__).parent.parent / "docs"
GIF_PATH = DOCS / "demo.gif"
STILL_PATH = DOCS / "screenshot.png"

DEFAULT_URL = "http://127.0.0.1:7860"

# The question the README's caption promises the screenshot shows. Kept here so
# the caption and the image cannot drift apart silently.
STILL_QUESTION = "Can my landlord charge a late fee if rent is 3 days late?"

# Strings the app itself emits. Waiting on these rather than on sleeps is what
# makes the recording deterministic: a slow provider makes it take longer, not
# capture the wrong moment.
ANSWER_DONE_MARK = "Statutes cited"
SAMPLE_CHIP = "Sniff this sample lease for red flags"
# Emitted when the content hash already has a report: a recording that sees this
# has no live scan in it, so the GIF would be a still of a finished panel.
CACHE_HIT_MARK = "Already sniffed"
CACHED_SNIFF_MARK = "saved report"

# 1280 wide is the layout's design width; 900 tall rather than 800 so the report
# panel's verdict summary and the answer fit in one frame without scrolling.
# Captured at 2x so the still survives GitHub's scaling, and the GIF downscales from
# those same frames, which is why it looks better than a 1x capture would.
VIEWPORT = {"width": 1280, "height": 900}
GIF_WIDTH = 900
GIF_COLORS = 128
# Frames are captured as fast as a 2x screenshot allows and each one is stamped, so
# playback follows the real gaps between them rather than a made-up frame rate. Long
# pauses are clamped, and PIL merges runs of identical frames into one longer frame,
# so the result is a little shorter than the wall clock (last run: 10.5s of playback
# for 15.1s of recording) — paced by the real thing, not equal to it.
GIF_MIN_MS, GIF_MAX_MS = 60, 1000

# Three deliberate pauses, in milliseconds, set by a Frame's own `hold_ms` rather
# than by the clock. The recording used to start on the frame AFTER the click, so
# the GIF opened on a page mid-scan: a reader had no idea what had been clicked, or
# where to look. These are one frame each, not a second's worth of captures — a held
# frame costs one frame's bytes however long it sits on screen, and identical frames
# would only be merged back into one anyway.
HOLD_BEFORE_CLICK_MS = 1400  # the untouched page, cursor resting on the chip
HOLD_ON_CLICK_MS = 500  # the click itself, so the eye lands on the chip
HOLD_ON_REPORT_MS = 1100  # the finished report, before the question starts typing

# Real time is the right pace for the scan; it is the wrong pace for a GIF, and the
# first recording with the holds showed why. Playback ran 19.4s of which the entire
# 0/15 -> 15/15 clause count was 530ms — six states, unreadable — while a single
# unchanging frame waiting on the gate held for 4.5s and another waiting on the
# answer held for 6.5s. Nearly two thirds of the recording was one still image.
#
# So identical consecutive frames are collapsed into one state, and a state's time on
# screen is bounded at both ends: nothing informative flashes past, and nothing static
# outstays its welcome. The wall-clock figure this script prints is unaffected and
# still describes the real run.
STATE_MIN_MS, STATE_MAX_MS = 260, 1200
# A recording that never ends is worse than a missing asset — it hangs CI-less
# local runs with no clue why. Each phase is bounded.
PHASE_TIMEOUT_S = 180


def chat_text(page) -> str:
    """Everything currently in the chat column, as plain text."""
    return page.locator(".chat-col").inner_text()


# Pinning the PAGE while the chat log scrolls itself. Streaming an answer grows the
# document, and the browser follows it — so the middle of the recording drifted down
# until the 🐕 LeaseHound wordmark was off-screen, and only came back when the final
# reframe ran. The chat log's own internal scrolling is wanted; the window's is not.
PIN_PAGE = "window.scrollTo(0, 0)"


class Frame(NamedTuple):
    """One captured moment, plus the two things a screenshot cannot record.

    A screenshot never contains the mouse pointer — the OS draws it, the page does
    not — so a recording of a click is a recording of a page that changes for no
    visible reason. `cursor` is where the pointer really was, in CSS pixels, and
    write_gif draws it back in.

    `hold_ms` overrides the real gap to the next frame. Everything else is paced by
    the wall clock, which is what makes the clause counter's pace honest; the three
    pauses are for the reader, and saying so explicitly keeps them out of the
    numbers this script prints about real time.
    """

    stamp: float
    png: bytes
    cursor: tuple[float, float] | None = None
    ring: bool = False  # draw a click ripple around the cursor
    hold_ms: int | None = None


def frame(page, cursor=None, ring: bool = False, hold_ms: int | None = None) -> Frame:
    """One stamped frame, taken with the page held at the top."""
    page.evaluate(PIN_PAGE)
    return Frame(time.time(), page.screenshot(type="png"), cursor, ring, hold_ms)


def centre_of(locator) -> tuple[float, float] | None:
    """Where a click on this element lands, in CSS pixels."""
    box = locator.bounding_box()
    return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2) if box else None


def wait_until(page, done, what: str, frames: list | None = None,
               timeout_s: int = PHASE_TIMEOUT_S) -> None:
    """Poll until `done()`, collecting stamped frames while waiting.

    Deliberately not page.wait_for_selector: what is being waited on is a message
    the app streams into a chat log, and the frames have to be captured *during*
    the wait or the GIF has nothing in it to show.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if frames is not None:
            frames.append(frame(page))
        else:
            page.evaluate(PIN_PAGE)
            page.wait_for_timeout(200)
        if done():
            if frames is not None:
                frames.append(frame(page))
            return
    raise TimeoutError(f"waited {timeout_s}s and {what} never happened")


def panel_pinned(page):
    # .first because the class lands on both gradio's block wrapper and the markdown
    # body inside it, and a bare locator is a strict-mode violation on two matches.
    panel = page.locator(".report-col .report-panel").first
    return lambda: panel.is_visible() and bool(panel.inner_text().strip())


def show_question_with_answer(page, question: str) -> None:
    """Scroll the chat log so the question sits with the answer it produced.

    The log auto-scrolls to the newest token, which pushes the question off the top
    — and a screenshot of an answer to an invisible question illustrates nothing.
    This is the third time it has needed work; the first fix was by hand, the second
    used scrollIntoView, which aligns an element's box but not its message wrapper,
    so the previous reply's last line stayed visible above the question and the hero
    image opened on a clipped bubble. Setting scrollTop against the scroll parent's
    own geometry is arithmetic rather than a hint, so it lands where it is told.
    """
    page.evaluate(
        """(text) => {
            const bubbles = [...document.querySelectorAll('.chat-col .message, .chat-col .bubble-wrap p')];
            const asked = bubbles.reverse().find(
                el => el.innerText.trim().startsWith(text.slice(0, 40)));
            if (asked) {
                // The nearest ancestor that actually scrolls; the bubble's own parent
                // usually does not.
                let box = asked.parentElement;
                while (box && box.scrollHeight <= box.clientHeight) box = box.parentElement;
                if (box) {
                    const top = asked.getBoundingClientRect().top
                              - box.getBoundingClientRect().top + box.scrollTop;
                    // A few pixels of air, so the bubble does not touch the top edge.
                    box.scrollTop = Math.max(0, top - 8);
                }
            }
            window.scrollTo(0, 0);
        }""",
        question,
    )
    page.wait_for_timeout(700)


def question_asked(page, question: str) -> bool:
    """Whether the question has actually reached the chat as a sent message."""
    return page.evaluate(
        """(q) => [...document.querySelectorAll('.chat-col .message')]
               .some(m => m.innerText.trim().startsWith(q.slice(0, 30)))""",
        question,
    )


def ask(page, question: str) -> None:
    """Fallback path: type the question. Only used if its chip has gone missing."""
    box = page.get_by_placeholder("Attach a lease, or ask about renting")
    box.click()
    box.fill(question)
    box.press("Enter")


def click_chip(page, text: str, frames: list | None, what: str):
    """Click an example chip the way a visitor does, with the pointer drawn in.

    Both of the recording's interactions are chip clicks, and only one of them used
    to look like one: the question was typed into the box by Playwright, so the GIF
    showed a scan being started by a click and then an answer arriving from nowhere.
    A reader watching for what to do next had one example, not two.
    """
    chip = page.get_by_text(text, exact=False).first
    chip.wait_for(state="visible", timeout=PHASE_TIMEOUT_S * 1000)
    # Measured and drawn with the page pinned at the top, because that is the layout
    # the frame shows. The question chips sit at y≈850 in a 900px viewport, so the
    # last of the three is below the fold and the first is only just above it.
    at = centre_of(chip)
    if frames is not None:
        frames.append(frame(page, cursor=at, hold_ms=HOLD_BEFORE_CLICK_MS))
    print(f"clicking {what}…")
    # Dispatched on the button from inside the page, not driven through the mouse.
    #
    # A chip does not submit itself: the app catches the click on `document`, polls
    # for the composer to fill, then presses send (see EXAMPLES_JS in app.py). A
    # synthesised mouse click has to survive Playwright's own scroll-into-view, which
    # races PIN_PAGE — every captured frame scrolls the window back to the top — and
    # during a recording it lost that race every time while working perfectly when
    # nothing was screenshotting. The listener is on `document`, so an in-page click
    # goes through exactly the handler a visitor's click reaches; the difference is
    # only in how the event is produced, and the pointer drawn into the frame is at
    # the coordinates the button really occupies.
    page.evaluate(
        """(label) => { const button = [...document.querySelectorAll(
               '#example-scan button, #example-prompts button')]
               .find(b => b.innerText.includes(label)); if (button) button.click(); }""",
        text[:40],
    )
    page.evaluate(PIN_PAGE)
    if frames is not None:
        # The ripple, then the same pointer without it — by the second frame the app
        # has replaced the chips with what the click set off.
        frames.append(frame(page, cursor=at, ring=True, hold_ms=HOLD_ON_CLICK_MS))
        frames.append(frame(page, cursor=at, hold_ms=HOLD_ON_CLICK_MS))
    return at


def shared_palette(images: list[Image.Image]) -> Image.Image:
    """One palette, built from every frame, with the octree quantiser.

    Two separate mistakes made the GIF greyscale while the PNG beside it in the
    README stayed in colour — the 🐕 came out a grey dog, the 🚩 a grey triangle,
    the ✅ a grey box.

    The first is the global palette. A GIF has one, and PIL takes it from whichever
    frame is saved first; frame 0 here is the app before anything happens, so the
    palette was derived from a near-blank page. Hence a strip of ALL frames,
    sampled at a third scale because a palette is about which colours occur, not
    where.

    The second is the quantiser, and it was the bigger one. **Median cut splits the
    colour space by pixel POPULATION**, and this UI is white — so a screenshot
    whose true-colour form has 3.2% coloured pixels came back with 2.6% after
    `MEDIANCUT`, with the loss concentrated in exactly the small, salient things:
    every emoji went grey. `FASTOCTREE` subdivides the colour space itself, is
    therefore indifferent to how much area a colour covers, and returns 3.3% — every
    emoji intact. Measured side by side at 128, 200 and 256 colours; 128 with
    FASTOCTREE is indistinguishable from 256, so the file stays small.
    """
    thumbs = [im.resize((im.width // 3, im.height // 3), Image.LANCZOS) for im in images]
    strip = Image.new("RGB", (thumbs[0].width, sum(t.height for t in thumbs)))
    offset = 0
    for thumb in thumbs:
        strip.paste(thumb, (0, offset))
        offset += thumb.height
    return strip.quantize(colors=GIF_COLORS, method=Image.FASTOCTREE)


# The pointer, as a polygon, in GIF pixels at GIF_WIDTH. Drawn rather than composited
# from an image file: a 20-pixel arrow is less code than an asset to keep in the repo,
# and it scales with GIF_WIDTH instead of going soft when that changes.
POINTER = [(0, 0), (0, 17), (4, 13), (7, 20), (11, 19), (8, 12), (14, 12)]
RING_RADIUS = 17


def draw_cursor(image: Image.Image, at: tuple[float, float], ring: bool) -> None:
    """Put the pointer back into a frame, at CSS coordinates.

    White outline under a black fill, because the pointer crosses both the white
    page and the teal chip and has to stay visible on both — the same reason every
    real cursor is drawn that way.
    """
    scale = image.width / VIEWPORT["width"]
    x, y = at[0] * scale, at[1] * scale
    draw = ImageDraw.Draw(image)
    if ring:
        # A click ripple. The chip is teal, so the ring is drawn in white with a
        # dark inner circle rather than the other way round.
        draw.ellipse([x - RING_RADIUS, y - RING_RADIUS, x + RING_RADIUS, y + RING_RADIUS],
                     outline=(255, 255, 255), width=4)
        draw.ellipse([x - RING_RADIUS, y - RING_RADIUS, x + RING_RADIUS, y + RING_RADIUS],
                     outline=(15, 23, 42), width=2)
    points = [(x + dx, y + dy) for dx, dy in POINTER]
    draw.polygon(points, fill=(15, 23, 42), outline=(255, 255, 255), width=2)


def durations(frames: list[Frame]) -> list[int]:
    """How long each frame sits on screen: the real gap, unless it is a held frame."""
    gaps = [
        min(GIF_MAX_MS, max(GIF_MIN_MS, round((b.stamp - a.stamp) * 1000)))
        for a, b in zip(frames, frames[1:])
    ]
    # The last frame is the payoff — the finished report beside the cited answer — so
    # it holds, rather than looping straight back to an empty page.
    gaps.append(2500)
    return [f.hold_ms if f.hold_ms is not None else gap
            for f, gap in zip(frames, gaps)]


def collapse(images: list[Image.Image], timings: list[int],
             deliberate: list[bool]) -> tuple[list, list[int]]:
    """Merge runs of identical frames into one state, then bound its time on screen.

    PIL merges identical frames on its own, but only by summing their durations,
    which is what let one unchanging frame sit for six and a half seconds. Doing it
    here means the sum can be clamped.

    Two exemptions from the ceiling, both for frames whose length is a decision rather
    than an accident: the three `hold_ms` pauses, and the final frame — the payoff, a
    finished report beside a cited answer, which holds instead of looping straight
    back to an empty page. Clamping the opening hold to 1.2s was this function's first
    bug, and it is the kind that looks like a tuning choice from outside.
    """
    states: list[Image.Image] = []
    held: list[int] = []
    fixed: list[bool] = []
    for image, milliseconds, is_fixed in zip(images, timings, deliberate):
        if states and image.tobytes() == states[-1].tobytes() and not is_fixed:
            held[-1] += milliseconds
        else:
            states.append(image)
            held.append(milliseconds)
            fixed.append(is_fixed)
    fixed[-1] = True
    return states, [ms if is_fixed else min(STATE_MAX_MS, max(STATE_MIN_MS, ms))
                    for ms, is_fixed in zip(held, fixed)]


def write_gif(frames: list[Frame], path: Path) -> None:
    """Assemble stamped PNG frames into an animated GIF at true speed.

    Downscaled and palettised because the honest constraint here is GitHub's: a
    README that takes ten seconds to show its own demo has not demonstrated
    anything.
    """
    rgb = []
    for captured in frames:
        image = Image.open(io.BytesIO(captured.png)).convert("RGB")
        image = image.resize(
            (GIF_WIDTH, round(image.height * GIF_WIDTH / image.width)), Image.LANCZOS)
        # After the resize, so the pointer is drawn at its final size rather than
        # downscaled into mush.
        if captured.cursor:
            draw_cursor(image, captured.cursor, captured.ring)
        rgb.append(image)

    master = shared_palette(rgb)
    # dither=NONE: this is flat UI, not a photograph. Floyd-Steinberg speckles large
    # areas of one colour and costs bytes for noise nobody wants to see.
    images = [frame.quantize(palette=master, dither=Image.Dither.NONE) for frame in rgb]
    images, timings = collapse(images, durations(frames),
                               [f.hold_ms is not None for f in frames])

    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=timings, loop=0, optimize=True, disposal=2)
    spanned = frames[-1].stamp - frames[0].stamp
    held = sum(f.hold_ms for f in frames if f.hold_ms is not None) / 1000
    print(f"  (recording spans {spanned:.1f}s of real time, plus {held:.1f}s of "
          f"deliberate pauses)")


def record(url: str, still_only: bool) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = context.new_page()
        # Not networkidle: Gradio holds a heartbeat connection open for the life of
        # the session, so the network is never idle and the wait times out. Wait for
        # the thing that actually has to be there instead — the example chip, which
        # arrives via demo.load and is therefore the real signal that the UI is up.
        page.goto(url, wait_until="load")

        frames: list[Frame] | None = None if still_only else []

        # The opening second, which the recording did not have: capture started on
        # the first poll AFTER the click, so the GIF began on a page already
        # scanning. A reader saw a counter moving and never saw what set it off.
        click_chip(page, SAMPLE_CHIP, frames, repr(SAMPLE_CHIP))
        # The report PANEL appearing, not a chat message: the fresh path ends with a
        # "Full report on the right" summary and the cached path does not, so keying
        # on that string worked exactly once and then hung for three minutes.
        wait_until(page, panel_pinned(page), "a report was pinned", frames)
        print("  scan finished")
        if frames is not None:
            # The finished report, on its own, before anything is asked of it. The
            # verdict counts are the point of the whole scan and they used to be on
            # screen for one frame's worth of real time.
            frames.append(frame(page, hold_ms=HOLD_ON_REPORT_MS))

        transcript = chat_text(page)
        if CACHE_HIT_MARK in transcript or CACHED_SNIFF_MARK in transcript:
            if frames is not None:
                browser.close()
                raise SystemExit(
                    "This server already has the sample lease cached, so the scan "
                    "returned instantly and there is no clause-by-clause progress to "
                    "record. Restart `python -m leasehound.app` and run this again "
                    "(or pass --still-only, which does not need the live progress)."
                )
            print("  (cache hit — fine for the still, which shows the finished state)")

        # The second click, and the reason the GIF shows two. Scanning and asking are
        # the product's two modes and both are one click from the same screen; typing
        # the question in made the answer look like it arrived on its own.
        # The panel appearing is not the scan being over. `panel_pinned` goes true on
        # the yield that pins the report, while the scan's Gradio event is still open
        # — and a chip click during an open event fills the composer and never gets
        # sent. It cost three recordings to find, because outside a recording the gap
        # closes before a human could click into it. The stop button going away and
        # the composer coming back are the app's own signals that it is idle.
        wait_until(page, lambda: page.evaluate(
            """() => { const box = document.querySelector('.chat-col .multimodal-textbox');
                   const stop = [...document.querySelectorAll('button')]
                       .find(b => b.innerText.includes('Call off the hound'));
                   return !box.querySelector('textarea').disabled
                          && !(stop && stop.offsetParent); }"""),
            "the scan finished settling", frames, timeout_s=30)
        if page.get_by_text(STILL_QUESTION, exact=False).count():
            click_chip(page, STILL_QUESTION, frames, f"the question chip: {STILL_QUESTION}")
            # Confirmed, not assumed. A chip does not submit itself: the click is
            # caught by a document-level listener that polls for the box to fill and
            # then presses send, and that handoff loses the race often enough against
            # a recording that screenshots continuously. Waiting on the question
            # appearing in the chat — rather than on the answer, three minutes later —
            # turns a silent failure into a two-second one with somewhere to go.
            try:
                wait_until(page, lambda: question_asked(page, STILL_QUESTION),
                           "the question was sent", frames, timeout_s=8)
            except TimeoutError:
                print("  (the chip click did not submit — typing the question instead)")
                ask(page, STILL_QUESTION)
        else:
            print(f"asking (chip not found, typing): {STILL_QUESTION}")
            ask(page, STILL_QUESTION)
        wait_until(page, lambda: ANSWER_DONE_MARK in chat_text(page),
                   "the answer finished", frames)
        print("  answer finished")

        # Settle: the sources footer is appended in a final yield after the last
        # token, and screenshotting mid-yield catches the answer without it.
        page.wait_for_timeout(1200)

        # The page is pinned to the top throughout now (see PIN_PAGE), so this is only
        # about the chat log's own scroll position: it auto-follows the newest token,
        # which pushed the question off the top of the log.
        show_question_with_answer(page, STILL_QUESTION)
        if frames is not None:
            frames.append(frame(page))

        DOCS.mkdir(exist_ok=True)
        page.screenshot(path=str(STILL_PATH))
        print(f"  {STILL_PATH.relative_to(DOCS.parent)} "
              f"({STILL_PATH.stat().st_size / 1024:.0f} KB)")

        if frames:
            write_gif(frames, GIF_PATH)
            print(f"  {GIF_PATH.relative_to(DOCS.parent)} "
                  f"({GIF_PATH.stat().st_size / 1024 / 1024:.1f} MB, {len(frames)} frames)")
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="a running leasehound.app")
    parser.add_argument("--still-only", action="store_true",
                        help="skip the GIF (still needs one live scan)")
    args = parser.parse_args()
    record(args.url, args.still_only)


if __name__ == "__main__":
    main()
