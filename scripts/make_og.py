"""Generate the 1200x630 share card at docs/og.png.

Why a generated card and not the app screenshot: LinkedIn, Slack and Twitter all
crop share images to roughly 1.91:1. docs/screenshot.png is 1.42:1, so a quarter
of its height was being cut, and at card size its UI text was unreadable anyway.

The card shows one real finding rather than describing the product, because a
scanner that claims to cite its sources should be able to show one. Nothing on it
is typed here: the wording comes from app.py's social constants, the clause from
examples/sample_lease.md, and the three numbers from evaluation/. A card that
could drift from the evidence behind it would be the wrong card for this repo.

    python scripts/make_og.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "leasehound" / "app.py"
LEASE = ROOT / "examples" / "sample_lease.md"
EVAL = ROOT / "evaluation"
OUT = ROOT / "docs" / "og.png"
PAW = ROOT / "leasehound" / "assets" / "favicon.png"

W, H = 1200, 630
PAD = 72
BG, BAND = "#ffffff", "#f1f5f9"
INK, MUTED, TEAL, RED, RULE = "#0f172a", "#64748b", "#0d9488", "#dc2626", "#e2e8f0"

SANS = "/System/Library/Fonts/HelveticaNeue.ttc"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
SERIF = "/System/Library/Fonts/Supplemental/Georgia.ttf"


def font(path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, index=index)


def constant(name: str) -> str:
    """Read a module-level string out of app.py without importing it.

    Importing would pull in gradio and the model stack for four strings.
    """
    for node in ast.parse(APP.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise SystemExit(f"{name} not found in {APP.relative_to(ROOT)}")


def clause_excerpt(draw, fnt, width: int, lines: int = 2) -> list[str]:
    """The real late-charges clause from the sample lease, trimmed to fit.

    Trimmed on word boundaries and marked with an ellipsis, so what sits inside
    the quotation marks is text the lease actually contains.
    """
    heading = constant("CARD_CLAUSE_HEADING")
    body = next(
        (ln[len(heading):].strip() for ln in LEASE.read_text().splitlines()
         if ln.startswith(heading)),
        "",
    )
    if not body:
        raise SystemExit(f"{heading!r} not found in {LEASE.relative_to(ROOT)}")

    out, line = [], ""
    for word in body.split():
        trial = f"{line} {word}".strip()
        # Leave room for the closing quote and ellipsis on the final line.
        room = width - (34 if len(out) == lines - 1 else 0)
        if draw.textlength(trial, font=fnt) <= room:
            line = trial
        else:
            out.append(line)
            line = word
            if len(out) == lines:
                break
    if len(out) < lines and line:
        out.append(line)
    out = out[:lines]
    consumed = len(" ".join(out).split())
    if consumed < len(body.split()):
        out[-1] = out[-1].rstrip(",") + "…"
    out[0] = "“" + out[0]
    out[-1] = out[-1] + "”"
    return out


def evidence() -> list[tuple[str, str]]:
    """The three numbers, recomputed from the evaluation output on every build."""
    synth = json.loads((EVAL / "synthetic_results.json").read_text())["summary"]
    cost = json.loads((EVAL / "scan_cost_summary.json").read_text())["all_scans"]
    planted = synth["planted_violations"]
    caught = round(planted * synth["strict_recall"])
    return [
        (f"{caught}/{planted}", "violations caught"),
        (f"{synth['precision']:.2f}", "precision"),
        (f"{cost['p50_seconds']:g}s", "median scan"),
    ]


def main() -> None:
    name = constant("SOCIAL_TITLE").partition("—")[0].strip()
    # The description's em dash is its hinge; the card keeps only the promise.
    promise = constant("SOCIAL_DESCRIPTION").partition("—")[2].strip().capitalize()

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, W, 96], fill=BAND)
    paw = Image.open(PAW).convert("RGBA").resize((50, 50), Image.LANCZOS)
    img.paste(paw, (PAD, 24), paw)
    d.text((PAD + 64, 30), name, font=font(SANS, 38, 1), fill=INK)
    stack, sf = constant("SOCIAL_STACK"), font(SANS, 22)
    d.text((W - PAD - d.textlength(stack, font=sf), 41), stack, font=sf, fill=MUTED)

    y = 166
    d.rectangle([PAD, y, PAD + 8, y + 172], fill=RED)
    d.text((PAD + 30, y + 2), "RED FLAG", font=font(SANS, 20, 1), fill=RED)
    quote = font(SERIF, 33)
    for i, line in enumerate(clause_excerpt(d, quote, W - PAD * 2 - 30)):
        d.text((PAD + 30, y + 36 + i * 44), line, font=quote, fill=INK)
    d.text((PAD + 30, y + 134), constant("CARD_CITATION"), font=font(MONO, 22), fill=TEAL)

    y += 244
    d.line([PAD, y, W - PAD, y], fill=RULE, width=2)

    y += 40
    d.text((PAD, y), promise, font=font(SANS, 42, 1), fill=INK)

    y += 84
    num_f, lab_f = font(SANS, 34, 1), font(SANS, 19)
    x = PAD
    for value, label in evidence():
        d.text((x, y), value, font=num_f, fill=TEAL)
        d.text((x, y + 44), label, font=lab_f, fill=MUTED)
        x += max(d.textlength(value, font=num_f), d.textlength(label, font=lab_f)) + 56

    url, uf = "github.com/tywangq/leasehound", font(MONO, 20)
    d.text((W - PAD - d.textlength(url, font=uf), y + 44), url, font=uf, fill=MUTED)

    img.save(OUT, optimize=True)
    print(f"{OUT.relative_to(ROOT)}  {img.width}x{img.height}  {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
