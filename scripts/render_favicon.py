"""Render the 🐕 wordmark to a PNG favicon, because the SVG one did not show up.

The favicon started as Gradio's logo — the wrong brand on the first surface a
visitor sees — and was replaced with an SVG holding the emoji as `<text>`, so the
tab glyph would match the hero exactly. It served correctly (HTTP 200,
`image/svg+xml`) and still showed nothing, for two reasons that stack:

  * **Safari does not support SVG favicons at all**, whatever they contain.
  * A `<text>` favicon depends on the client resolving an emoji font inside the
    favicon rendering context, which is more restricted than a page.

A PNG has neither problem. Rendering it needs the emoji font, so this drives a
real browser rather than guessing at glyph metrics — the same argument
`record_demo.py` makes: an asset nobody can regenerate will eventually be wrong,
and a hand-exported icon is exactly that.

Playwright is DEV-ONLY and deliberately absent from pyproject and
requirements-lock.txt; the shipped image needs no browser. Install it just for
this:

    .venv/bin/python -m pip install playwright && .venv/bin/playwright install chromium

Then:

    python -m scripts.render_favicon

Costs nothing. The PNG is committed, so a deploy does not need this script.
"""

import argparse
from pathlib import Path

ASSETS = Path(__file__).parent.parent / "leasehound" / "assets"
FAVICON_PATH = ASSETS / "favicon.png"

GLYPH = "🐕"
# 128 rather than 32: browsers and OS surfaces downscale a favicon to several sizes
# (tab, bookmark, touch icon), and downscaling is the direction that stays sharp.
SIZE = 128
# The glyph is drawn a little smaller than the box so the emoji's own bearings do not
# clip at the edges — a tab icon that touches its bounds looks cropped, not full.
GLYPH_RATIO = 0.82

PAGE = """
<style>
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  #glyph {{
    width: {size}px; height: {size}px;
    display: flex; align-items: center; justify-content: center;
    font-size: {font}px; line-height: 1;
    /* Named explicitly so the render does not silently fall back to a monochrome
       outline font, which is what a blank-looking icon usually turns out to be. */
    font-family: "Apple Color Emoji", "Noto Color Emoji", "Segoe UI Emoji", sans-serif;
  }}
</style>
<div id="glyph">{glyph}</div>
"""


def render(path: Path, size: int) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_context(viewport={"width": size, "height": size}).new_page()
        page.set_content(PAGE.format(size=size, font=round(size * GLYPH_RATIO),
                                     glyph=GLYPH))
        path.parent.mkdir(parents=True, exist_ok=True)
        # omit_background keeps the corners transparent, so the icon sits on a dark
        # tab strip without a white tile behind it.
        page.locator("#glyph").screenshot(path=str(path), omit_background=True)
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--out", type=Path, default=FAVICON_PATH)
    args = parser.parse_args()
    render(args.out, args.size)
    print(f"{args.out.relative_to(args.out.parent.parent.parent)} "
          f"({args.out.stat().st_size / 1024:.1f} KB, {args.size}x{args.size})")


if __name__ == "__main__":
    main()
