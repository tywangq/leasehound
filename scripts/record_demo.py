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

from PIL import Image

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


def frame(page) -> tuple[float, bytes]:
    """One stamped frame, taken with the page held at the top."""
    page.evaluate(PIN_PAGE)
    return time.time(), page.screenshot(type="png")


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


def ask(page, question: str) -> None:
    box = page.get_by_placeholder("Attach a lease, or ask about renting")
    box.click()
    box.fill(question)
    box.press("Enter")


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


def write_gif(frames: list[tuple[float, bytes]], path: Path) -> None:
    """Assemble stamped PNG frames into an animated GIF at true speed.

    Downscaled and palettised because the honest constraint here is GitHub's: a
    README that takes ten seconds to show its own demo has not demonstrated
    anything.
    """
    rgb = []
    for _, raw in frames:
        frame = Image.open(io.BytesIO(raw)).convert("RGB")
        rgb.append(frame.resize(
            (GIF_WIDTH, round(frame.height * GIF_WIDTH / frame.width)), Image.LANCZOS))

    master = shared_palette(rgb)
    # dither=NONE: this is flat UI, not a photograph. Floyd-Steinberg speckles large
    # areas of one colour and costs bytes for noise nobody wants to see.
    images = [frame.quantize(palette=master, dither=Image.Dither.NONE) for frame in rgb]

    stamps = [stamp for stamp, _ in frames]
    durations = [
        min(GIF_MAX_MS, max(GIF_MIN_MS, round((b - a) * 1000)))
        for a, b in zip(stamps, stamps[1:])
    ]
    # The last frame is the payoff — the finished report beside the cited answer — so
    # it holds, rather than looping straight back to an empty page.
    durations.append(2500)
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True, disposal=2)
    print(f"  (recording spans {stamps[-1] - stamps[0]:.1f}s of real time)")


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

        frames: list[tuple[float, bytes]] | None = None if still_only else []

        chip = page.get_by_text(SAMPLE_CHIP, exact=False).first
        chip.wait_for(state="visible", timeout=PHASE_TIMEOUT_S * 1000)
        print(f"clicking {SAMPLE_CHIP!r}…")
        chip.click()
        # The report PANEL appearing, not a chat message: the fresh path ends with a
        # "Full report on the right" summary and the cached path does not, so keying
        # on that string worked exactly once and then hung for three minutes.
        wait_until(page, panel_pinned(page), "a report was pinned", frames)
        print("  scan finished")

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

        print(f"asking: {STILL_QUESTION}")
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
