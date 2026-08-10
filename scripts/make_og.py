"""Generate the 1200x630 share card at docs/og.png.

Why a generated card and not the app screenshot: LinkedIn, Slack and Twitter all
crop to roughly 1.91:1. docs/screenshot.png is 1.42:1, so a quarter of its height
was being cut, and at card size the UI text was too small to read anyway. A card
has one job — say what this is in the two seconds before someone decides to click.

The title and description are read out of app.py rather than retyped, so the card
cannot drift from the meta tags it sits next to. Re-run after changing either:

    python scripts/make_og.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "leasehound" / "app.py"
OUT = ROOT / "docs" / "og.png"
PAW = ROOT / "leasehound" / "assets" / "favicon.png"

W, H = 1200, 630
BG = "#f8fafc"
INK = "#0f172a"
MUTED = "#64748b"
TEAL = "#0d9488"
RULE = "#e2e8f0"
RED = "#dc2626"

SANS = "/System/Library/Fonts/HelveticaNeue.ttc"
MONO = "/System/Library/Fonts/SFNSMono.ttf"


def font(path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, index=index)


def literal(name: str) -> str:
    """Pull a module-level string constant out of app.py without importing it.

    Importing would pull in gradio and the whole model stack for two strings.
    """
    tree = ast.parse(APP.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise SystemExit(f"{name} not found in {APP}")


def wrap(draw, text, fnt, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def main() -> None:
    title = literal("SOCIAL_TITLE")
    # "LeaseHound — lease red-flag scanner" -> the two halves are styled apart.
    name, _, kicker = title.partition("—")
    description = literal("SOCIAL_DESCRIPTION")
    # The em dash is the sentence's hinge; the card gives each half its own line.
    lede, _, promise = description.partition("—")

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Teal spine, echoing the app's primary hue.
    d.rectangle([0, 0, 12, H], fill=TEAL)

    pad_x, y = 84, 74

    paw = Image.open(PAW).convert("RGBA").resize((78, 78), Image.LANCZOS)
    img.paste(paw, (pad_x, y - 6), paw)

    d.text((pad_x + 98, y + 4), name.strip(), font=font(SANS, 62, 1), fill=INK)
    d.text(
        (pad_x + 100, y + 88),
        kicker.strip().upper(),
        font=font(SANS, 23, 0),
        fill=MUTED,
    )

    y += 168
    d.line([pad_x, y, W - pad_x, y], fill=RULE, width=2)

    y += 46
    body = font(SANS, 35, 0)
    for line in wrap(d, lede.strip(), body, W - pad_x * 2):
        d.text((pad_x, y), line, font=body, fill=INK)
        y += 48

    y += 8
    strong = font(SANS, 35, 1)
    for line in wrap(d, promise.strip().capitalize(), strong, W - pad_x * 2):
        d.text((pad_x, y), line, font=strong, fill=TEAL)
        y += 48

    # Evidence strip. These three are the numbers the README and resume both
    # carry; a card claiming accuracy without one is just a slogan.
    y = H - 132
    d.line([pad_x, y - 34, W - pad_x, y - 34], fill=RULE, width=2)

    stats = [
        ("60/61", "planted violations caught"),
        ("0.90", "precision"),
        ("9.3s", "median scan"),
    ]
    num_f, lab_f = font(SANS, 40, 1), font(SANS, 20, 0)
    x = pad_x
    for value, label in stats:
        d.text((x, y), value, font=num_f, fill=RED if value == "60/61" else INK)
        d.text((x, y + 52), label, font=lab_f, fill=MUTED)
        x += max(
            d.textlength(value, font=num_f), d.textlength(label, font=lab_f)
        ) + 62

    url = "github.com/tywangq/leasehound"
    uf = font(MONO, 21)
    d.text(
        (W - pad_x - d.textlength(url, font=uf), y + 52), url, font=uf, fill=MUTED
    )

    OUT.parent.mkdir(exist_ok=True)
    img.save(OUT, optimize=True)
    print(f"{OUT.relative_to(ROOT)}  {img.width}x{img.height}  {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
