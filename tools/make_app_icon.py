"""Draw the application icon, so the taskbar shows this product and not cmd.

A .bat cannot carry an icon -- the console window wears cmd.exe's -- but the
window an operator keeps on the taskbar is the Qt one, and that takes its icon
from the QApplication.  So this is drawn rather than scraped, from the product's
own tokens (``zlc_ui.fluent.style``): the tile is the accent, the word is the
ink those windows write in, and the type is the one the chrome is set in.  An
icon that shares a palette with the window it opens is recognised as the same
thing; one borrowed off the web is not.

The mark is the wordmark over a single pulse.  ZLab is what the lab is called
and the pulse is what this software streams, and at the sizes a taskbar
actually uses the two together still resolve into one recognisable tile.

Every size is DRAWN at that size rather than resampled from the largest: a
stroke computed for 256 px turns to porridge at 16.

    python tools/make_app_icon.py
    python tools/make_app_icon.py --out /tmp/zlc.ico
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

#: The product's own tokens (zlc_ui.fluent.style ACCENT / TEXT / SURFACE).
ACCENT = "#77AADD"
INK = "#323130"
SURFACE = "#FFFFFF"
#: The chrome's typeface.  Only needed to DRAW the icon, never to show it.
TYPEFACE = "C:/Windows/Fonts/segoeuib.ttf"

WORD = "ZLab"
SUPERSAMPLE = 8
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def _fitted(draw: ImageDraw.ImageDraw, side: float) -> ImageFont.FreeTypeFont:
    """The largest Segoe UI Bold that keeps the word inside ``side``."""

    size = 10
    while True:
        font = ImageFont.truetype(TYPEFACE, size)
        left, top, right, bottom = draw.textbbox((0, 0), WORD, font=font)
        if right - left > side or bottom - top > side:
            return ImageFont.truetype(TYPEFACE, max(10, size - 1))
        size += 2


def _mark(draw: ImageDraw.ImageDraw, side: int) -> None:
    radius = int(side * 0.20)
    draw.rounded_rectangle((0, 0, side - 1, side - 1), radius=radius, fill=ACCENT)

    font = _fitted(draw, side * 0.74)
    left, top, right, bottom = draw.textbbox((0, 0), WORD, font=font)
    draw.text(
        (
            (side - (right - left)) / 2 - left,
            side * 0.42 - (bottom - top) / 2 - top,
        ),
        WORD,
        font=font,
        fill=INK,
    )

    # One rising pulse under the word, in the surface colour the windows use.
    stroke = side * 0.055
    half = stroke / 2.0
    left_x, right_x = side * 0.17, side * 0.83
    rise, fall = side * 0.40, side * 0.63
    low, high = side * 0.78, side * 0.66
    draw.rectangle((left_x, low - half, rise, low + half), fill=SURFACE)
    draw.rectangle((rise - stroke, high - half, rise, low + half), fill=SURFACE)
    draw.rectangle((rise - stroke, high - half, fall, high + half), fill=SURFACE)
    draw.rectangle((fall - stroke, high - half, fall, low + half), fill=SURFACE)
    draw.rectangle((fall - stroke, low - half, right_x, low + half), fill=SURFACE)


def render(side: int) -> Image.Image:
    """One square icon, drawn large and resolved down so the edges stay clean."""

    big = side * SUPERSAMPLE
    image = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    _mark(ImageDraw.Draw(image), big)
    return image.resize((side, side), Image.LANCZOS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="packages/zlc_ui/src/zlc_ui/assets/zlc.ico",
        help="where the .ico goes",
    )
    arguments = parser.parse_args(argv)
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = [render(side) for side in SIZES]
    frames[-1].save(out, format="ICO", sizes=[(side, side) for side in SIZES])
    print("wrote %s (%s)" % (out, ", ".join(map(str, SIZES))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
