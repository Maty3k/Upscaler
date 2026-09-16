"""Hit a file-size budget — "make this fit under 500 KB" — losing as little as
possible on the way.

Pure Pillow (no AI). The picture is encoded into memory over and over and
measured, rather than guessed at from a formula, because how large a JPEG
comes out depends entirely on what is in it: a flat sky and a forest of leaves
at the same quality differ by an order of magnitude.

The levers are pulled in order of how much they cost you:

1. **Format.** WebP carries the same picture in roughly half a JPEG's bytes,
   so "auto" reaches for it first and only falls back to JPEG if this Pillow
   build can't write it.
2. **Metadata.** Stripped always. It is usually a rounding error in size, but
   it removes the GPS coordinates too, which is worth having by default.
3. **Quality.** Binary-searched for the *highest* setting that still fits, so
   the answer is the best of the ones that fit rather than the first.
4. **Size.** Only if quality alone can't get there. The scale is estimated
   from how far over budget the smallest-quality attempt was, rather than
   stepped down blindly, so it converges in a couple of rounds.

A PNG has no quality dial, so it gets palette reduction and then resizing
instead.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

from PIL import Image

from upscaler.convert import FORMATS, convert, extension_for

AUTO = "auto"
# Formats worth targeting a budget with; the rest of FORMATS are conversions,
# not compressors.
BUDGET_FORMATS = [AUTO] + [f for f in ("WebP", "JPEG", "AVIF", "PNG") if f in FORMATS]
LOSSY = {f for f, (_, _, lossy) in FORMATS.items() if lossy}
# Stops at 64: below that a photo posterises into bands, which looks worse
# than the same picture simply being smaller, so resizing takes over instead.
PNG_COLOR_STEPS = (256, 128, 64)
MIN_SCALE = 0.15
_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([kmg]?)b?\s*$", re.I)


@dataclass
class OptimizeParams:
    target: str = "500 KB"       # "500KB", "2 MB", "1.5mb", or plain bytes
    fmt: str = AUTO
    min_quality: int = 40        # go no lower before resizing instead
    max_quality: int = 95
    allow_resize: bool = True
    max_edge: int = 0            # cap the long edge first, 0 = leave it
    background: str = "#ffffff"  # what transparency is flattened onto, if it must be


@dataclass
class OptimizeResult:
    data: bytes
    fmt: str
    quality: "int | None"
    size: "tuple[int, int]"
    scale: float
    fits: bool
    attempts: int
    target_bytes: int
    colors: "int | None" = None

    @property
    def nbytes(self) -> int:
        return len(self.data)

    @property
    def extension(self) -> str:
        return extension_for(self.fmt)


def parse_size(text: "str | int | float") -> "int | None":
    """Read a budget: "500KB", "2 MB", "1.5mb", "750000" all work. Returns
    None for anything unreadable, so a typo can be reported rather than
    silently treated as zero."""
    if isinstance(text, (int, float)):
        return int(text) if text > 0 else None
    m = _SIZE_RE.match(str(text or ""))
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    factor = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}[unit]
    total = int(value * factor)
    return total if total > 0 else None


def human_size(n: int) -> str:
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.2f} MB"
    if n >= 1024:
        # A decimal below 100 KB, so "8.3 KB over the 8 KB budget" doesn't
        # round into the nonsense "8 KB over the 8 KB budget".
        return f"{n / 1024:.1f} KB" if n < 100 * 1024 else f"{n / 1024:.0f} KB"
    return f"{n} B"


def _hex(c: str) -> "tuple[int, int, int]":
    s = (c or "").lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except (ValueError, IndexError):
        return (255, 255, 255)


def pick_format(img: Image.Image, fmt: str = AUTO) -> str:
    """Which encoder to use. "auto" prefers WebP — same picture, about half
    the bytes of JPEG — and only drops to JPEG when this Pillow build has no
    WebP, or to PNG when transparency has to survive and WebP is missing."""
    if fmt != AUTO:
        if fmt not in FORMATS:
            raise ValueError(f"unknown format {fmt!r} — choose from {', '.join(BUDGET_FORMATS)}")
        return fmt
    has_alpha = "A" in img.getbands()
    if "WebP" in FORMATS:
        return "WebP"
    if has_alpha:
        return "PNG"
    return "JPEG" if "JPEG" in FORMATS else next(iter(FORMATS))


def _encode(img: Image.Image, fmt: str, quality: int, background) -> bytes:
    return convert(img, fmt, quality=int(quality), background=background)


def _scaled(img: Image.Image, scale: float) -> Image.Image:
    if scale >= 0.999:
        return img
    w = max(1, int(round(img.width * scale)))
    h = max(1, int(round(img.height * scale)))
    return img.resize((w, h), Image.LANCZOS)


def _best_quality(img: Image.Image, fmt: str, target: int, lo: int, hi: int,
                  background) -> "tuple[bytes, int, int]":
    """Binary-search the highest quality that fits. Returns (data, quality,
    encodes). If even ``lo`` overshoots, the ``lo`` attempt comes back so the
    caller can see how far over it was and scale from that."""
    attempts = 0
    low_data = _encode(img, fmt, lo, background)
    attempts += 1
    if len(low_data) > target:
        return low_data, lo, attempts

    best_data, best_q = low_data, lo
    a, b = lo, hi
    while a <= b:
        mid = (a + b) // 2
        data = _encode(img, fmt, mid, background)
        attempts += 1
        if len(data) <= target:
            best_data, best_q = data, mid
            a = mid + 1
        else:
            b = mid - 1
    return best_data, best_q, attempts


def _png_attempt(img: Image.Image, target: int) -> "tuple[bytes, int | None, int]":
    """PNG has no quality dial, so trade colours instead: full RGB first, then
    an adaptive palette with fewer and fewer entries."""
    attempts = 0
    data = convert(img, "PNG")
    attempts += 1
    if len(data) <= target:
        return data, None, attempts
    if "A" in img.getbands():
        # Quantising an image with transparency mangles the edges; leave the
        # colours alone and let the caller resize instead.
        return data, None, attempts
    best = data
    for colors in PNG_COLOR_STEPS:
        quantised = img.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT)
        data = convert(quantised, "PNG")
        attempts += 1
        best = data
        if len(data) <= target:
            return data, colors, attempts
    return best, PNG_COLOR_STEPS[-1], attempts


def optimize(img: Image.Image, p: OptimizeParams) -> OptimizeResult:
    """Encode ``img`` as small as it needs to be, as well as it can be.

    Raises ValueError for an unreadable budget; otherwise always returns a
    result, with ``fits=False`` when even the smallest allowed version is over.
    """
    target = parse_size(p.target)
    if target is None:
        raise ValueError(f"can't read the size {p.target!r} — try 500KB or 2MB")
    fmt = pick_format(img, p.fmt)
    background = _hex(p.background)
    lo = int(max(1, min(100, p.min_quality)))
    hi = int(max(lo, min(100, p.max_quality)))

    work = img
    scale = 1.0
    if p.max_edge and max(img.size) > p.max_edge:
        scale = p.max_edge / max(img.size)
        work = _scaled(img, scale)

    attempts = 0
    best: "OptimizeResult | None" = None
    for _round in range(6):
        if fmt == "PNG":
            data, colors, used = _png_attempt(work, target)
            quality = None
        else:
            data, quality, used = _best_quality(work, fmt, target, lo, hi, background)
            colors = None
        attempts += used
        best = OptimizeResult(data=data, fmt=fmt, quality=quality, size=work.size,
                              scale=scale, fits=len(data) <= target, attempts=attempts,
                              target_bytes=target, colors=colors)
        if best.fits or not p.allow_resize or scale <= MIN_SCALE:
            break
        # Bytes track pixel count, so the linear scale needed is about the
        # square root of how far over we are. Overshoot slightly (0.95) so a
        # stubborn image converges instead of creeping down in tiny steps.
        ratio = target / max(1, len(data))
        step = max(0.35, min(0.92, math.sqrt(ratio) * 0.95))
        scale = max(MIN_SCALE, scale * step)
        work = _scaled(img, scale)
    return best  # type: ignore[return-value]


def describe(res: OptimizeResult) -> str:
    """A short human summary of what it took."""
    bits = [f"{res.fmt} {human_size(res.nbytes)}"]
    if res.quality is not None:
        bits.append(f"quality {res.quality}")
    if res.colors:
        bits.append(f"{res.colors} colors")
    bits.append(f"{res.size[0]}×{res.size[1]}")
    if res.scale < 0.999:
        bits.append(f"scaled to {res.scale * 100:.0f}%")
    bits.append(f"{res.attempts} encodes")
    text = " · ".join(bits)
    if not res.fits:
        text += (f" · ⚠ still over the {human_size(res.target_bytes)} budget — "
                 "allow resizing, lower the minimum quality, or raise the target")
    return text
