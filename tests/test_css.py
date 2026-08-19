"""The CSS, measured in a browser, because that is the only place it exists.

Every style bug this app has had was invisible to the rest of the suite and to reading
the file: a rule that gradio silently out-specified (`.prose :last-child` beat a 20px
margin, and the same 20px measured correctly in every render that had no .prose
ancestor), a `transform: none` that un-centred the element it was meant to hold still,
a token the dark-mode block forgot so the verdict chips went white-on-pale. Reading CSS
proves nothing about a page; a computed value does.

The page here is not the app: it imports the shipped UI_STYLE and lays out the same
elements, so it needs no vector store, no API key and no model. It does need a browser,
so the whole module skips when playwright or its chromium is not installed.

Kept to the seams that have actually broken. This is a regression net, not a snapshot
of every declaration — a test that pins a value nobody chose deliberately is a test
that will be deleted the first time it is right to change it.
"""
import json
import socket

import pytest

playwright_api = pytest.importorskip("playwright.sync_api",
                                     reason="browser checks need playwright")

import gradio as gr  # noqa: E402  (after the skip, so a missing browser skips cleanly)

from leasehound.app import QUESTION_EXAMPLES, SCAN_EXAMPLE, UI_STYLE  # noqa: E402
from leasehound.report_html import render_report_html  # noqa: E402

FINDING = {"index": 3, "verdict": "red", "clause": "3. LATE CHARGES. $150 if late.",
           "explanation": "Washington caps late fees.", "citations": ["RCW 59.18.170"],
           "urls": {"RCW 59.18.170": "https://app.leg.wa.gov/x"}}
PROTECTION = {"name": "Mold", "status": "missing", "requirement": "the pamphlet",
              "citation": "RCW 59.18.060"}
REPORT = render_report_html([FINDING], "sample_lease.md", "wa", [PROTECTION],
                            gate_flagged=True, clauses_total=9)

# Same shape as app.py's column, so the selectors under test resolve the same way.
BUILD = """
with gr.Blocks() as demo:
    with gr.Row(elem_classes="main-row"):
        with gr.Column(scale=1, elem_classes="chat-col"):
            with gr.Group(elem_classes="chat-surface"):
                gr.Chatbot(value=TURNS, height="min(420px, 45vh)", show_label=False,
                           buttons=["copy_all"])
                box = gr.MultimodalTextbox(show_label=False, file_count="single",
                                           file_types=[".pdf", ".md", ".txt"])
            gr.Examples(examples=[SCAN_EXAMPLE], example_labels=[SCAN_EXAMPLE["text"]],
                        inputs=box, label="Scan a lease", elem_id="example-scan")
            gr.Examples(examples=QUESTION_EXAMPLES, inputs=box, label="Ask a question",
                        elem_id="example-prompts")
        with gr.Column(scale=1, elem_classes="report-col"):
            gr.Button("✋ Call off the hound", size="sm", elem_id="stop-button")
            with gr.Row(elem_id="report-actions"):
                gr.Button("", size="sm")
            gr.HTML(REPORT, elem_classes="report-panel")
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def hexof(css_colour: str) -> str:
    """`rgb(…)` or `color(srgb …)` — chromium reports color-mix results as the latter."""
    parts = [float(n) for n in __import__("re").findall(r"[\d.]+", css_colour)][:3]
    scaled = [round(n * 255) if max(parts) <= 1 else round(n) for n in parts]
    return "#" + "".join(f"{n:02x}" for n in scaled)


def hue(hex_colour: str) -> float:
    import colorsys
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return colorsys.rgb_to_hls(*channels)[0] * 360


def relative_luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    low, high = sorted((relative_luminance(a), relative_luminance(b)))
    return (high + 0.05) / (low + 0.05)


@pytest.fixture(scope="module")
def page_factory():
    # Long enough to overflow the 420px/45vh box: the scroll-to-bottom control only
    # exists when there is something to scroll, and a skipped test guards nothing.
    turns = []
    for i in range(6):
        turns.append({"role": "user", "content": f"Can my landlord keep the deposit? ({i})"})
        turns.append({"role": "assistant", "content":
                      "Not without an itemized statement within 30 days of the end of the "
                      "tenancy — and a checklist signed at move-in, or the deposit was "
                      f"never lawfully collected. RCW 59.18.260 ({i})"})
    scope = {"gr": gr, "TURNS": turns, "SCAN_EXAMPLE": SCAN_EXAMPLE,
             "QUESTION_EXAMPLES": QUESTION_EXAMPLES, "REPORT": REPORT}
    exec(BUILD, scope)  # noqa: S102 — a Blocks layout, written once, right here
    demo = scope["demo"]
    port = free_port()
    demo.launch(server_port=port, server_name="127.0.0.1", quiet=True,
                prevent_thread_lock=True, **UI_STYLE)
    try:
        with playwright_api.sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as missing:  # chromium not downloaded
                pytest.skip(f"no chromium: {missing}")

            def open_page(scheme="light"):
                page = browser.new_page(viewport={"width": 1400, "height": 900},
                                        color_scheme=scheme)
                page.goto(f"http://127.0.0.1:{port}/", wait_until="domcontentloaded",
                          timeout=60_000)
                page.wait_for_selector(".report-panel .lh-chip", timeout=60_000)
                page.wait_for_timeout(1200)
                return page

            yield open_page
            browser.close()
    finally:
        demo.close()


def test_the_summary_chips_clear_the_first_notice(page_factory):
    """20px, and it has to be measured with .prose above it.

    The margin lived on .lh-chips, where gradio's `.prose :last-child {margin-bottom: 0
    !important}` deleted it — the chip row is the last child of .lh-head. It measured 20
    in every standalone render, because a standalone render does not have the rule.
    """
    page = page_factory()
    gap = page.evaluate("""() => {
      const r = s => document.querySelector('.report-panel ' + s).getBoundingClientRect();
      return +(r('.lh-note').y - r('.lh-chip').bottom).toFixed(1);
    }""")
    page.close()
    assert gap == 20


def test_every_pale_green_is_derived_from_the_one_accent(page_factory):
    """The wash is the accent mixed into the page, not a colour picked beside it.

    Two greens 9° apart is what "薄荷绿和深绿不搭" was; 12% of #0f766e lands on the
    accent's own hue. If someone reintroduces a literal, this is where it shows.
    """
    page = page_factory()
    page.hover("#example-prompts .gallery-item >> nth=1")
    page.wait_for_timeout(300)
    got = page.evaluate("""() => {
      const s = getComputedStyle(document.querySelectorAll('#example-prompts .gallery-item')[1]);
      const scan = getComputedStyle(document.querySelector('#example-scan .gallery-item'));
      return {hover_bg: s.backgroundColor, hover_border: s.borderTopColor,
              accent: scan.backgroundColor};
    }""")
    page.close()
    assert hexof(got["accent"]) == "#0f766e"
    assert hexof(got["hover_bg"]) == "#e2efee"      # 12% into #ffffff
    assert hexof(got["hover_border"]) == "#a9cecb"  # 36%


def test_nothing_in_the_chat_column_moves_on_hover(page_factory):
    """The scroll-to-bottom control slid 13px right when hovered.

    Not gradio's doing in the end: gradio centres the container with
    `left: 50%; transform: translate(-50%)`, and this file's own `transform: none`
    un-centred it. The container's transform is the thing to leave alone.
    """
    page = page_factory()
    page.evaluate("""() => {
      const box = document.querySelector('.chat-col .bubble-wrap')
               || document.querySelector('.chat-col .block .wrap');
      if (box) box.scrollTop = 30;
    }""")
    page.wait_for_timeout(600)
    if not page.query_selector(".scroll-down-button-container"):
        page.close()
        pytest.skip("gradio did not show the scroll-down control")
    before = page.evaluate("() => document.querySelector('.scroll-down-button-container')"
                           ".getBoundingClientRect().toJSON()")
    page.hover(".scroll-down-button-container button")
    page.wait_for_timeout(400)
    after = page.evaluate("() => document.querySelector('.scroll-down-button-container')"
                          ".getBoundingClientRect().toJSON()")
    page.close()
    assert (round(after["x"] - before["x"]), round(after["y"] - before["y"])) == (0, 0)


def test_the_chat_toolbar_buttons_carry_no_frame_of_their_own(page_factory):
    """They sit on a rounded white panel of gradio's; a border each read as a circle
    around every glyph at 16px. The report's three actions keep theirs — nothing is
    behind those."""
    page = page_factory()
    buttons = page.evaluate("""() => [...document.querySelectorAll(
        '.chat-col .icon-button-wrapper button')].map(b => {
      const s = getComputedStyle(b);
      return {border: s.borderTopWidth, radius: s.borderTopLeftRadius, shadow: s.boxShadow};
    })""")
    page.close()
    assert buttons, "no toolbar buttons rendered"
    for style in buttons:
        assert style["border"] == "0px"
        assert style["radius"] == "8px"
        assert style["shadow"] == "none"


def test_the_verdict_chips_stay_readable_in_dark_mode(page_factory):
    """--lh-card was missing from the dark block while .lh-finding and .lh-note carried
    the same value inline, so the chips — which read the token — stayed white and their
    lettering went light: about 1.4:1 on a #0a0f1e page."""
    page = page_factory("dark")
    got = page.evaluate("""() => {
      const chip = document.querySelector('.report-panel .lh-chip');
      const s = getComputedStyle(chip);
      return {bg: s.backgroundColor, fg: s.color,
              page: getComputedStyle(document.body).backgroundColor};
    }""")
    page.close()
    assert hexof(got["page"]) != "#ffffff", "the dark scheme did not engage"
    assert contrast(hexof(got["fg"]), hexof(got["bg"])) >= 4.5, json.dumps(got)


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_the_missing_marker_is_not_a_second_green(page_factory, scheme):
    """Four verdict colours, and two of them used to be 33° apart.

    `missing` was the theme's teal at 175°, next to `clear` at 142° — the closest pair in
    the legend, on 9px squares. It was also the product's own colour, so a category wore
    the brand, and #0d9488 on white is 3.74:1, under what a section heading needs. Blue
    is 79° away and reads in both schemes, which needs two steps: the light value alone
    measures 3.69:1 on the dark page.
    """
    page = page_factory(scheme)
    got = page.evaluate("""() => {
      const panel = document.querySelector('.report-panel');
      const tok = n => getComputedStyle(panel).getPropertyValue(n).trim();
      const head = document.querySelector('.report-panel h2.lh-missing');
      return {missing: tok('--lh-missing'), green: tok('--lh-green'),
              head_colour: head && getComputedStyle(head).color,
              behind: getComputedStyle(document.body).backgroundColor};
    }""")
    page.close()
    assert hue(got["missing"]) - hue(got["green"]) >= 60, got
    assert contrast(hexof(got["head_colour"]), hexof(got["behind"])) >= 4.5, got


@pytest.mark.parametrize("label,selector", [
    ("example chip", "#example-prompts .gallery-item"),
    ("user bubble", ".chat-col .message.user"),
    ("stop button", "#stop-button"),
    ("report action", "#report-actions button"),
    ("verdict chip", ".report-panel .lh-chip"),
])
def test_every_surface_stays_readable_in_dark_mode(page_factory, label, selector):
    """Four rules wrote `background: #ffffff` and `color: var(--body-text-color)`.

    Half a rule reading the theme is worse than none of it: in dark mode the text
    followed the token to #f1f5f9 and the fill stayed white, so the example chips, the
    user's bubble, the stop button and the report's actions all measured 1.1:1 — not low,
    invisible. The chips test above passed throughout, which is why this one is a list.
    """
    page = page_factory("dark")
    got = page.evaluate("""(sel) => {
      const el = document.querySelector(sel);
      if (!el) return null;
      const s = getComputedStyle(el);
      return {bg: s.backgroundColor, fg: s.color,
              page: getComputedStyle(document.body).backgroundColor};
    }""", selector)
    page.close()
    assert got, f"{label} did not render"
    assert hexof(got["page"]) != "#ffffff", "the dark scheme did not engage"
    # A transparent fill inherits whatever is behind it, which in dark mode is dark.
    if got["bg"].startswith("rgba") and got["bg"].endswith(", 0)"):
        pytest.skip(f"{label} has no fill of its own")
    assert contrast(hexof(got["fg"]), hexof(got["bg"])) >= 4.5, f"{label}: {got}"


def test_the_text_under_an_attachment_clears_the_chip_edge(page_factory, tmp_path):
    """6px, the chip's own corner radius. Flush with the tile's left edge is what a grid
    says and not what the eye reads, since the glyph sits 10px inside the tile."""
    lease = tmp_path / "lease.md"
    lease.write_text("1. RENT. Tenant shall pay rent.\n")
    page = page_factory()
    page.set_input_files(".multimodal-textbox input[type=file]", str(lease))
    page.wait_for_selector(".multimodal-textbox .thumbnails", timeout=30_000)
    page.wait_for_timeout(1000)
    offsets = page.evaluate("""() => {
      const box = document.querySelector('.multimodal-textbox');
      const tile = box.querySelector('.thumbnail-item') || box.querySelector('.thumbnails > *');
      const ink = tile.querySelector('svg, img'), ta = box.querySelector('textarea');
      const r = e => e.getBoundingClientRect();
      const pad = parseFloat(getComputedStyle(ta).paddingLeft);
      return {text_minus_tile: +((r(ta).x + pad) - r(tile).x).toFixed(1),
              glyph_off_centre: +((r(ink).x + r(ink).width / 2)
                                  - (r(tile).x + r(tile).width / 2)).toFixed(1)};
    }""")
    page.close()
    assert offsets["text_minus_tile"] == 6
    assert offsets["glyph_off_centre"] == 0   # and the glyph stays centred in its tile
