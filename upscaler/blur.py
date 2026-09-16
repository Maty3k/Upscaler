"""Photo blur toolbox — many kinds of blur, applied to the whole picture or
through a shaped, painted or gradient mask, with adjustable strength.

Pure PIL + numpy (no AI, no OpenCV). Every blur takes its strength as a
percentage of the image's short side, so one setting looks the same on a
phone snap and a 4K frame — and the reduced-size live preview matches the
full-size export.

Kinds
    gaussian  soft, natural blur
    box       flat average (harsher, "cheap camera" look)
    motion    streaks along an angle, like camera shake / a moving subject
    spin      rotation blur around a centre point
    zoom      radial streaks out of a centre point ("warp speed")
    lens      disc-shaped bokeh, optional highlight bloom (bright discs)
    pixelate  mosaic squares — the privacy blur
    surface   edge-preserving smoothing — softens skin / noise, keeps edges

Masks
    whole, rectangle (rounded), ellipse, band (a straight strip — tilt-shift
    when combined with "outside"), painted (a brush mask). ``feather`` softens
    the edge, ``outside`` flips which side gets blurred, ``progressive`` ramps
    the strength through the feathered zone (half → full) instead of
    cross-fading one blur, which is what makes tilt-shift look graded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

KINDS = ["gaussian", "box", "motion", "spin", "zoom", "lens", "pixelate", "surface"]
SHAPES = ["whole", "rectangle", "ellipse", "band", "painted"]

MAX_RADIUS_FRAC = 0.10   # strength 100 → radius = 10% of the short side
MAX_SPIN_DEG = 40.0      # strength 100 → ±20° of spin
MAX_ZOOM = 0.40          # strength 100 → streaks out to 1.4× scale
PREVIEW_EDGE = 900       # live preview renders at most this long


@dataclass
class BlurParams:
    kind: str = "gaussian"
    strength: float = 30.0    # 0..100, relative to the short side
    angle: float = 0.0        # motion streak direction, degrees (0 = horizontal)
    center_x: float = 50.0    # spin / zoom centre, % of width
    center_y: float = 50.0    # spin / zoom centre, % of height
    highlights: float = 0.0   # lens: bokeh bloom, 0..100
    threshold: float = 25.0   # surface: how strong an edge must be to survive, 0..100


@dataclass
class MaskParams:
    shape: str = "whole"
    x: float = 50.0           # shape centre, % of width
    y: float = 50.0           # shape centre, % of height
    w: float = 50.0           # rectangle / ellipse width, % of width
    h: float = 50.0           # rectangle / ellipse height; band thickness — % of height
    angle: float = 0.0        # band tilt, degrees (0 = horizontal strip)
    roundness: float = 0.0    # rectangle corner rounding, 0..100
    feather: float = 10.0     # edge softness, % of the short side
    outside: bool = False     # blur outside the shape instead of inside it
    progressive: bool = True  # ramp half → full strength across the feather
    painted: Image.Image | None = None   # brush mask (L), any size


# ── helpers ───────────────────────────────────────────────────────────────────
def radius_px(strength: float, size: tuple[int, int]) -> float:
    """Blur radius in pixels for a strength percentage on an image of ``size``."""
    return max(0.0, min(100.0, float(strength))) / 100.0 * MAX_RADIUS_FRAC * min(size)


def _u8(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _pad_reflect(img: Image.Image, pad: int) -> Image.Image:
    arr = np.pad(np.asarray(img.convert("RGB")), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    return Image.fromarray(arr, "RGB")


def _center_crop(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = size
    left = (img.width - w) // 2
    top = (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def _box1d(a: np.ndarray, length: int, axis: int) -> np.ndarray:
    """Moving average over ``length`` samples along ``axis`` (same shape;
    edges average whatever is available). Cumulative sums → O(n)."""
    n = a.shape[axis]
    cs = np.cumsum(a, axis=axis, dtype=np.float64)
    zero = np.zeros_like(np.take(cs, [0], axis=axis))
    cs = np.concatenate([zero, cs], axis=axis)
    half = length // 2
    idx = np.arange(n)
    hi = np.clip(idx + (length - half), 0, n)
    lo = np.clip(idx - half, 0, n)
    out = np.take(cs, hi, axis=axis) - np.take(cs, lo, axis=axis)
    shape = [1] * a.ndim
    shape[axis] = n
    return (out / (hi - lo).reshape(shape)).astype(np.float32)


# ── the blurs (RGB in, RGB out) ───────────────────────────────────────────────
def _gaussian(img: Image.Image, r: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(r)) if r >= 0.3 else img


def _box(img: Image.Image, r: float) -> Image.Image:
    return img.filter(ImageFilter.BoxBlur(r)) if r >= 0.5 else img


def _motion(img: Image.Image, length: float, angle: float) -> Image.Image:
    """Streaks of ``length`` px along ``angle``: reflect-pad, rotate the streak
    direction to horizontal, 1-D box average, rotate back, crop the centre."""
    n = int(round(length))
    if n < 2:
        return img
    pad = n + 2
    big = _pad_reflect(img, pad).rotate(angle, resample=Image.BILINEAR, expand=True)
    arr = _box1d(np.asarray(big, dtype=np.float32), n, axis=1)
    back = _u8(arr).rotate(-angle, resample=Image.BILINEAR, expand=True)
    return _center_crop(back, img.size)


def _spin(img: Image.Image, spread_deg: float, cx: float, cy: float) -> Image.Image:
    """Rotation blur: the average of the image turned through ±spread/2
    around (cx, cy). Reflect-padded so corners never smear black."""
    if spread_deg < 0.3:
        return img
    w, h = img.size
    steps = int(np.clip(spread_deg * 1.5, 6, 40))
    pad = int(1.2 * max(w, h) * math.sin(math.radians(spread_deg / 2))) + 2
    base = _pad_reflect(img, pad)
    acc = np.zeros((h + 2 * pad, w + 2 * pad, 3), np.float32)
    for ang in np.linspace(-spread_deg / 2, spread_deg / 2, steps):
        acc += np.asarray(base.rotate(ang, resample=Image.BILINEAR, center=(cx + pad, cy + pad)),
                          dtype=np.float32)
    return _u8(acc / steps).crop((pad, pad, pad + w, pad + h))


def _zoom(img: Image.Image, amount: float, cx: float, cy: float) -> Image.Image:
    """Radial streaks: the average of the image scaled 1 → 1+amount about
    (cx, cy). Every scaled copy still covers the frame, so no edges show."""
    if amount < 0.003:
        return img
    w, h = img.size
    steps = int(np.clip(amount * 100, 6, 40))
    acc = np.asarray(img, dtype=np.float32).copy()
    for s in np.linspace(1.0, 1.0 + amount, steps)[1:]:
        sw, sh = max(w + 1, round(w * s)), max(h + 1, round(h * s))
        ox = min(sw - w, max(0, round(cx * (sw / w - 1))))
        oy = min(sh - h, max(0, round(cy * (sh / h - 1))))
        big = img.resize((sw, sh), Image.BILINEAR).crop((ox, oy, ox + w, oy + h))
        acc += np.asarray(big, dtype=np.float32)
    return _u8(acc / steps)


def _disc(img: Image.Image, r: float) -> np.ndarray:
    """A square box blur averaged over three rotations — a 12-gon, close
    enough to the disc a lens draws. Float RGB array out."""
    pad = int(r) + 2
    padded = _pad_reflect(img, pad)
    acc = np.zeros((img.height, img.width, 3), np.float32)
    for ang in (0.0, 30.0, 60.0):
        big = padded.rotate(ang, resample=Image.BILINEAR, expand=True).filter(ImageFilter.BoxBlur(r))
        back = big.rotate(-ang, resample=Image.BILINEAR, expand=True)
        acc += np.asarray(_center_crop(back, img.size), dtype=np.float32)
    return acc / 3.0


def _lens(img: Image.Image, r: float, highlights: float) -> Image.Image:
    """Disc-shaped bokeh. ``highlights`` adds a blurred copy of just the bright
    parts on top (an additive bloom), so points of light swell into glowing
    discs while midtones keep their exact tone."""
    if r < 0.5:
        return img
    out = _disc(img, r)
    k = max(0.0, min(100.0, highlights)) / 100.0 * 2.5
    if k > 0:
        a = np.asarray(img, dtype=np.float32)
        lum = a.mean(axis=2, keepdims=True) / 255.0
        weight = _smoothstep((lum - 0.55) / 0.45)          # only the bright stuff
        hl = _u8(a * weight)                                # 0..255, full precision
        out = out + _disc(hl, r) * k
    return _u8(out)


def _pixelate(img: Image.Image, block: float) -> Image.Image:
    b = int(round(block))
    if b < 2:
        return img
    w, h = img.size
    small = img.resize((max(1, math.ceil(w / b)), max(1, math.ceil(h / b))), Image.BOX)
    return small.resize((w, h), Image.NEAREST)


def _surface(img: Image.Image, r: float, threshold: float) -> Image.Image:
    """Edge-preserving smoothing: blend towards a gaussian blur only where
    the blur changes a pixel by less than the threshold — flat areas and
    fine noise smooth out, real edges stay put."""
    if r < 0.3:
        return img
    a = np.asarray(img, dtype=np.float32)
    b = np.asarray(img.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32)
    t = max(1.0, float(threshold) / 100.0 * 128.0)
    diff = np.abs(a - b).max(axis=2, keepdims=True)
    wgt = np.clip(1.0 - diff / t, 0.0, 1.0)
    wgt = wgt * wgt * (3.0 - 2.0 * wgt)   # smoothstep
    return _u8(a + (b - a) * wgt)


def blur_image(img: Image.Image, p: BlurParams) -> Image.Image:
    """Blur the whole image (RGB) with ``p``; strength scales with the image."""
    if p.kind not in KINDS:
        raise ValueError(f"unknown blur kind {p.kind!r} — expected one of {KINDS}")
    rgb = img.convert("RGB")
    r = radius_px(p.strength, rgb.size)
    cx = rgb.width * p.center_x / 100.0
    cy = rgb.height * p.center_y / 100.0
    if p.kind == "gaussian":
        return _gaussian(rgb, r)
    if p.kind == "box":
        return _box(rgb, r)
    if p.kind == "motion":
        return _motion(rgb, 2.0 * r, p.angle)          # streak length = the blur diameter
    if p.kind == "spin":
        return _spin(rgb, p.strength / 100.0 * MAX_SPIN_DEG, cx, cy)
    if p.kind == "zoom":
        return _zoom(rgb, p.strength / 100.0 * MAX_ZOOM, cx, cy)
    if p.kind == "lens":
        return _lens(rgb, r, p.highlights)
    if p.kind == "pixelate":
        return _pixelate(rgb, r)
    return _surface(rgb, r, p.threshold)


# ── masks (L, 255 = blur here) ────────────────────────────────────────────────
def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_mask(size: tuple[int, int], m: MaskParams) -> Image.Image:
    """The blur weight map for an image of ``size``: 255 where the blur
    applies fully, 0 where the original stays, feathered in between."""
    if m.shape not in SHAPES:
        raise ValueError(f"unknown mask shape {m.shape!r} — expected one of {SHAPES}")
    w, h = size
    short = min(w, h)
    feather_px = max(0.0, m.feather) / 100.0 * short
    cx, cy = w * m.x / 100.0, h * m.y / 100.0

    if m.shape == "whole":
        mask = Image.new("L", size, 255)
    elif m.shape in ("rectangle", "ellipse"):
        rw, rh = max(1.0, w * m.w / 100.0), max(1.0, h * m.h / 100.0)
        box = [cx - rw / 2, cy - rh / 2, cx + rw / 2, cy + rh / 2]
        mask = Image.new("L", size, 0)
        d = ImageDraw.Draw(mask)
        if m.shape == "ellipse":
            d.ellipse(box, fill=255)
        else:
            rad = max(0.0, min(100.0, m.roundness)) / 100.0 * min(rw, rh) / 2.0
            d.rounded_rectangle(box, radius=rad, fill=255)
        if feather_px > 0.5:
            mask = mask.filter(ImageFilter.GaussianBlur(feather_px / 2.0))
    elif m.shape == "band":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = math.radians(m.angle)
        dist = np.abs(-(xx - cx) * math.sin(a) + (yy - cy) * math.cos(a))
        half = max(1.0, h * m.h / 100.0) / 2.0
        if feather_px > 0.5:
            weight = 1.0 - _smoothstep((dist - half) / feather_px)
        else:
            weight = (dist <= half).astype(np.float32)
        mask = Image.fromarray((weight * 255).astype(np.uint8), "L")
    else:  # painted
        if m.painted is None:
            mask = Image.new("L", size, 0)
        else:
            mask = m.painted.convert("L").resize(size, Image.BILINEAR)
            if feather_px > 0.5:
                mask = mask.filter(ImageFilter.GaussianBlur(feather_px / 2.0))

    if m.outside:
        mask = Image.fromarray(255 - np.asarray(mask), "L")
    return mask


# ── putting it together ───────────────────────────────────────────────────────
def apply(img: Image.Image, p: BlurParams, m: MaskParams) -> Image.Image:
    """Blur ``img`` through the mask. Alpha (a cut-out's edge) is kept as is."""
    alpha = img.getchannel("A") if "A" in img.getbands() else None
    rgb = img.convert("RGB")
    full = blur_image(rgb, p)
    if m.shape == "whole" and not m.outside:
        out = full
    else:
        weight = np.asarray(build_mask(rgb.size, m), dtype=np.float32)[..., None] / 255.0
        a = np.asarray(rgb, dtype=np.float32)
        b = np.asarray(full, dtype=np.float32)
        graded = m.progressive and m.feather > 0 and p.kind != "pixelate" and p.strength > 0
        if graded:
            # Two strength levels: the feather ramps original → half → full,
            # so a tilt-shift band builds up blur instead of cross-fading.
            half = np.asarray(blur_image(rgb, replace(p, strength=p.strength / 2.0)), dtype=np.float32)
            mid = a + (half - a) * np.clip(weight * 2.0, 0.0, 1.0)
            out_arr = mid + (b - mid) * np.clip(weight * 2.0 - 1.0, 0.0, 1.0)
        else:
            out_arr = a + (b - a) * weight
        out = _u8(out_arr)
    if alpha is not None:
        out = out.convert("RGBA")
        out.putalpha(alpha)
    return out


def preview_pair(img: Image.Image, p: BlurParams, m: MaskParams,
                 max_edge: int = PREVIEW_EDGE) -> tuple[Image.Image, Image.Image]:
    """(before, after) at preview size — the same look as the export, since
    strength and every mask measure are relative to the image."""
    scale = min(1.0, max_edge / max(img.size))
    small = img if scale >= 1.0 else img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    return small, apply(small, p, m)


def describe(p: BlurParams, m: MaskParams, size: tuple[int, int]) -> str:
    r = radius_px(p.strength, size)
    if p.kind == "spin":
        amount = f"±{p.strength / 100 * MAX_SPIN_DEG / 2:.1f}°"
    elif p.kind == "zoom":
        amount = f"to {1 + p.strength / 100 * MAX_ZOOM:.2f}×"
    elif p.kind == "motion":
        amount = f"{2 * r:.0f}px streaks at {p.angle:g}°"
    elif p.kind == "pixelate":
        amount = f"{max(2, round(r))}px blocks"
    else:
        amount = f"{r:.1f}px radius"
    where = "whole image" if m.shape == "whole" else (
        f"{'outside' if m.outside else 'inside'} the {m.shape}"
        + (f", {m.feather:g}% feather" if m.feather > 0 else "")
        + (", graded" if m.progressive and m.feather > 0 else ""))
    return f"{p.kind} · strength {p.strength:g} ({amount}) · {where}"
