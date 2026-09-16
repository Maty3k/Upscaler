"""Font discovery and loading, shared by the panel, Steam and watermark tools.

Lives on its own so a tool that only needs to draw text doesn't have to import
:mod:`upscaler.panel`, which reaches ffmpeg and torch through the video module.
"""

from __future__ import annotations

import os
from functools import lru_cache

from PIL import ImageFont

# Candidate fonts (first that exists wins); falls back to PIL's bitmap font.
_FONT_DIRS = [
    "/System/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
    "C:/Windows/Fonts",
    "/usr/share/fonts",
    "/usr/local/share/fonts",
]
# Nice display families shown first in the picker (only those that exist are kept).
_CURATED = [
    "Arial Bold", "Arial", "Arial Black", "Helvetica", "HelveticaNeue", "Impact",
    "Futura", "Gill Sans", "Avenir Next", "Avenir", "Optima", "Trebuchet MS",
    "Verdana", "Verdana Bold", "Georgia", "Georgia Bold", "Times New Roman",
    "Baskerville", "Didot", "Palatino", "Copperplate", "American Typewriter",
    "Courier New Bold", "Courier New", "Menlo", "Monaco", "Andale Mono",
    "Chalkboard", "Chalkduster", "Marker Felt", "Noteworthy", "Bradley Hand",
    "Snell Roundhand", "Apple Chancery", "Papyrus", "Comic Sans MS",
    "arialbd", "arial", "impact", "DejaVuSans-Bold", "DejaVuSans",
]
_FONT_SKIP = ("emoji", "braille", "symbol", "wingding", "webding", "dingbat",
              "bookshelf", "opensymbol")


def _discover_fonts() -> dict[str, str]:
    """Map a display name → font file path for usable display fonts on this
    machine. Curated families come first; everything else follows so the picker
    is rich but the good options are at the top."""
    found: dict[str, str] = {}
    for d in _FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".ttf", ".ttc", ".otf")):
                    found.setdefault(os.path.splitext(f)[0], os.path.join(root, f))
    fonts: dict[str, str] = {}
    for stem in _CURATED:
        if stem in found:
            fonts[stem] = found[stem]
    for stem, path in sorted(found.items()):
        if stem in fonts:
            continue
        if any(j in stem.lower() for j in _FONT_SKIP):
            continue
        fonts[stem] = path
    return fonts or {"Default": ""}


FONTS = _discover_fonts()
FONT_NAMES = list(FONTS)
DEFAULT_FONT = next((n for n in ("Arial Bold", "Impact", "Helvetica") if n in FONTS),
                    FONT_NAMES[0])


@lru_cache(maxsize=128)
def _load_font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(8, int(size))
    path = FONTS.get(name)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    for p in FONTS.values():  # fall back to any working font
        if p:
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    try:
        # Pillow's own face, at the size asked for — so text still scales on a
        # machine with no system fonts at all, such as a bare CI container.
        return ImageFont.load_default(size=size)
    except TypeError:                         # Pillow < 10.1 takes no size
        return ImageFont.load_default()
