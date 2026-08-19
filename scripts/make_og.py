"""Generate the 1200x630 share card at docs/og.png.

Why a generated card and not the app screenshot: LinkedIn, Slack and Twitter all
crop share images to roughly 1.91:1. docs/screenshot.png is 1.42:1, so a quarter
of its height was being cut, and at card size its UI text was unreadable anyway.

The card draws the product's own report surface rather than describing the
product, because a scanner whose whole claim is that it cites its sources should
be able to show one doing it. Drawn rather than screenshotted so the type stays
crisp at the ~360px LinkedIn actually renders.

Two places it deliberately differs from the panel, and both come from that 360px.
The panel dropped its red fill and its serif quote: with a whole report on screen,
a section heading in red already says which pile a finding is in, and weight says
"someone else wrote this" as well as a serif does. Neither holds at a third of the
size. The heading becomes 4px of unreadable red, so the fill carries the signal
instead; and New York's thin strokes disappear, so the quote is the app's sans at
600. Everything else — the radius, the neutral chips, the hairline, the muted
clause label — is the panel's.

Almost nothing on it is typed here: the wording comes from app.py's social
constants, the clause from examples/sample_lease.md, and the headline three from
evaluation/. The one exception is the verdict row, and the reason is instructive.
The obvious source, scan_results.json, has a `flagged_yellow` field — but it
counts planted violations that came back yellow, not cautions on ordinary
clauses, and reading it as the latter put "0 cautions" on a card whose product
reports one. The counts live in app.py with that history recorded, and the two
the artifact genuinely does cover are cross-checked here so a card and an
evaluation that disagree fail the build rather than ship.

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
PAD = 56
PAGE = "#eef2f6"
CARD, CHROME, EDGE = "#ffffff", "#f6f8fa", "#dbe2ea"
INK, BODY, MUTED = "#0f172a", "#334155", "#94a3b8"
# TEAL is the product's own colour and appears where the product speaks: the statute
# citation and the evidence row. MISSING is a verdict category and is blue — see the
# --lh-missing comment in app.py for why it stopped being teal.
TEAL_D = "#0f766e"
MISSING = "#2563eb"
RED, RED_BG, RED_EDGE = "#dc2626", "#fef2f2", "#fecaca"
AMBER, GREEN = "#d97706", "#16a34a"
# The panel's own two neutrals, for the chips and the finding that is not red-filled.
CARD_FILL, HAIR = "#f8fafc", "#e2e8f0"

SANS = "/System/Library/Fonts/HelveticaNeue.ttc"
MONO = "/System/Library/Fonts/SFNSMono.ttf"


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


def sample_scan() -> dict:
    """The stored scan of the lease whose clause this card quotes."""
    leases = json.loads((EVAL / "scan_results.json").read_text())["leases"]
    for entry in leases:
        if Path(entry["file"]).name == LEASE.name:
            return entry
    raise SystemExit(f"no stored scan of {LEASE.name} in scan_results.json")


def chips() -> list[tuple[str, str]]:
    """The report's own verdict row, in the app's colour order.

    Read from app.py rather than from scan_results.json. That artifact has
    `flagged_red` and `protections_expected`, which agree with the report, but its
    `flagged_yellow` counts something else — planted violations that came back
    yellow, not cautions on ordinary clauses — and reading it as the caution count
    put "0 cautions" on a card whose product shows one. The counts app.py carries
    are checked against the running app; see the comment beside CARD_VERDICTS.
    """
    palette = {"red": RED, "caution": AMBER, "missing": MISSING, "clear": GREEN}
    out = []
    for part in constant("CARD_VERDICTS").split("|"):
        label = part.strip()
        colour = next((c for k, c in palette.items() if k in label), MISSING)
        out.append((colour, label))
    # Cross-check the two the artifact does cover, so a card and an evaluation
    # that disagree fail the build instead of shipping.
    scan = sample_scan()
    for count, word in [(len(scan["flagged_red"]), "red flags"),
                        (len(scan["protections_expected"]), "missing protections")]:
        if f"{count} {word}" not in constant("CARD_VERDICTS"):
            raise SystemExit(
                f"CARD_VERDICTS disagrees with scan_results.json: expected "
                f"{count} {word}"
            )
    return out


def evidence() -> list[tuple[str, str]]:
    """The headline three, recomputed from the evaluation output on every build."""
    synth = json.loads((EVAL / "synthetic_results.json").read_text())["summary"]
    cost = json.loads((EVAL / "scan_cost_summary.json").read_text())["all_scans"]
    planted = synth["planted_violations"]
    return [
        (f"{round(planted * synth['strict_recall'])}/{planted}", "violations caught"),
        (f"{synth['precision']:.2f}", "precision"),
        # One decimal, not the stored precision: the resume and the portfolio both say
        # 9.3s, and a card reading 9.31s beside them looks like a different measurement
        # rather than the same one rounded. The artifact keeps the full value.
        (f"{cost['p50_seconds']:.1f}s", "median scan"),
    ]


def corpus_line() -> str:
    prov = json.loads((EVAL / "scan_cost_summary.json").read_text())["provenance"]
    return f"Washington · RCW 59.18 · corpus {prov['corpus_snapshot']}"


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
    if len(" ".join(out).split()) < len(body.split()):
        out[-1] = out[-1].rstrip(",") + "…"
    out[0] = "“" + out[0]
    out[-1] = out[-1] + "”"
    return out


def mark(img, x, y, size):
    """The hound, from the same favicon the app serves in its tab.

    This was briefly a drawn "LH" monogram, on the reasoning that an emoji renders
    differently on every platform that scrapes the card. That reasoning does not apply
    here: the card is a PNG, so the glyph is rasterised once when this script runs and
    no scraper ever receives a codepoint. What was left was taste, weighed against the
    fact that this product is named after a hound and speaks in its voice throughout.
    The dog is the mark.
    """
    paw = Image.open(PAW).convert("RGBA").resize((size, size), Image.LANCZOS)
    img.paste(paw, (x, y), paw)


def main() -> None:
    img = Image.new("RGB", (W, H), PAGE)
    d = ImageDraw.Draw(img)
    bx, by = PAD, PAD
    bw, bh = W - PAD * 2, H - PAD * 2

    d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=16, fill=CARD, outline=EDGE, width=2)
    d.rounded_rectangle([bx, by, bx + bw, by + 62], radius=16, fill=CHROME)
    d.rectangle([bx, by + 46, bx + bw, by + 62], fill=CHROME)
    d.line([bx, by + 62, bx + bw, by + 62], fill=EDGE, width=2)
    for i in range(3):
        d.ellipse([bx + 22 + i * 22, by + 24, bx + 34 + i * 22, by + 36], fill="#e5eaf0")

    mark(img, bx + 102, by + 15, 32)
    d.text((bx + 140, by + 21), constant("SOCIAL_TITLE").partition("—")[0].strip(),
           font=font(SANS, 21, 1), fill=INK)
    tag, tf = corpus_line(), font(MONO, 15)
    d.text((bx + bw - 24 - d.textlength(tag, font=tf), by + 25), tag, font=tf, fill=MUTED)

    ix, right = bx + 40, bx + bw - 40
    title_f = font(SANS, 26, 1)
    d.text((ix, by + 100), "Scan report", font=title_f, fill=INK)
    d.text((ix + d.textlength("Scan report", font=title_f) + 18, by + 108),
           LEASE.name, font=font(MONO, 17), fill=MUTED)

    cx, cf = ix, font(SANS, 17, 1)
    for colour, label in chips():
        wide = d.textlength(label, font=cf) + 46
        d.rounded_rectangle([cx, by + 152, cx + wide, by + 190], radius=19,
                            fill=CARD_FILL, outline=HAIR, width=2)
        d.rounded_rectangle([cx + 16, by + 165, cx + 27, by + 176], radius=3, fill=colour)
        d.text((cx + 35, by + 162), label, font=cf, fill=BODY)
        cx += wide + 12

    fy = by + 222
    d.rounded_rectangle([ix, fy, right, fy + 180], radius=8, fill=RED_BG,
                        outline=RED_EDGE, width=2)
    d.rounded_rectangle([ix, fy, ix + 6, fy + 180], radius=3, fill=RED)
    d.rectangle([ix + 3, fy, ix + 6, fy + 180], fill=RED)
    d.text((ix + 28, fy + 20), "CLAUSE 3 · LATE CHARGES", font=font(SANS, 15, 1), fill=RED)
    quote = font(SANS, 23, 1)
    for i, line in enumerate(clause_excerpt(d, quote, right - ix - 60)):
        d.text((ix + 28, fy + 52 + i * 32), line, font=quote, fill=INK)
    d.text((ix + 28, fy + 132), constant("CARD_CITATION"), font=font(SANS, 18, 1), fill=TEAL_D)

    y = by + bh - 72
    d.line([ix, y - 24, right, y - 24], fill=PAGE, width=2)
    x, nf, lf = ix, font(SANS, 26, 1), font(SANS, 16)
    for value, label in evidence():
        d.text((x, y), value, font=nf, fill=TEAL_D)
        d.text((x + d.textlength(value, font=nf) + 10, y + 8), label, font=lf, fill=MUTED)
        x += d.textlength(value, font=nf) + d.textlength(label, font=lf) + 52

    img.save(OUT, optimize=True)
    print(f"{OUT.relative_to(ROOT)}  {img.width}x{img.height}  {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
