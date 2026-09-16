"""Color and light — exposure, contrast, tone, white balance, saturation and
black-and-white conversion, over the whole photo or through a mask.

Pure PIL + numpy (no AI). Everything runs in float, in a fixed order that
matches how photo editors work, so the sliders stay predictable no matter
which combination is used:

    exposure → levels → gamma → highlights / shadows → contrast → clarity
    → white balance → hue / vibrance / saturation → black & white + tone

Exposure and white balance work in **linear light** (sRGB decoded first), which
is why a stop of exposure looks like a stop and a warm shift doesn't muddy the
midtones. Everything else works on the displayed values, where the sliders feel
the way people expect.

Regions reuse the Blur tab's masks (``upscaler.blur.MaskParams``), so an
adjustment can be limited to a shape, a band (a graduated filter) or wherever
you paint.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
from PIL import Image, ImageFilter

from upscaler import blur

# Rec.709 luma weights — used for every "keep the brightness" calculation.
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
MAX_EV = 2.0          # exposure ±100 → ±2 stops
PREVIEW_EDGE = blur.PREVIEW_EDGE


@dataclass
class AdjustParams:
    # ── light ──
    exposure: float = 0.0        # -100..100, ±2 stops
    contrast: float = 0.0        # -100..100
    highlights: float = 0.0      # -100..100, recover (-) or lift (+) the brights
    shadows: float = 0.0         # -100..100, lift (+) or deepen (-) the darks
    black_point: float = 0.0     # 0..50, where black starts (% of range)
    white_point: float = 100.0   # 50..100, where white starts (% of range)
    gamma: float = 1.0           # 0.2..3.0, midtone brightness
    clarity: float = 0.0         # -100..100, local contrast / punch
    # ── color ──
    temperature: float = 0.0     # -100 cool … +100 warm
    tint: float = 0.0            # -100 green … +100 magenta
    hue: float = 0.0             # -180..180 degrees
    saturation: float = 0.0      # -100..100
    vibrance: float = 0.0        # -100..100, spares already-saturated colors
    # ── black & white ──
    mono: bool = False
    mono_red: float = 30.0       # channel mixer weights, normalised on use
    mono_green: float = 59.0
    mono_blue: float = 11.0
    tone_color: str = "#d8b070"  # split-tone / sepia color
    tone_strength: float = 0.0   # 0..100

    def is_identity(self) -> bool:
        """True when nothing would change the picture."""
        return self == AdjustParams(tone_color=self.tone_color,
                                    mono_red=self.mono_red,
                                    mono_green=self.mono_green,
                                    mono_blue=self.mono_blue)


# ── presets ───────────────────────────────────────────────────────────────────
PRESET_NONE = "None"
PRESETS: dict[str, AdjustParams] = {
    "Natural boost": AdjustParams(exposure=4, contrast=8, shadows=12, vibrance=18, clarity=10),
    "Punchy": AdjustParams(contrast=28, clarity=30, saturation=15, black_point=3),
    "Soft / matte": AdjustParams(contrast=-12, black_point=8, white_point=95, shadows=18,
                                 saturation=-8, clarity=-10),
    "Warm golden": AdjustParams(exposure=6, temperature=32, tint=6, highlights=-15,
                                shadows=14, vibrance=20),
    "Cool teal": AdjustParams(temperature=-30, tint=-8, contrast=14, shadows=10, vibrance=12),
    "Faded film": AdjustParams(contrast=-8, black_point=10, white_point=93, temperature=10,
                               saturation=-18, gamma=1.08),
    "High key": AdjustParams(exposure=22, contrast=-6, shadows=30, highlights=-10, saturation=-6),
    "Low key / moody": AdjustParams(exposure=-14, contrast=22, shadows=-22, clarity=18,
                                    temperature=-10),
    "Black & white": AdjustParams(mono=True, contrast=18, clarity=20, shadows=8),
    "Sepia": AdjustParams(mono=True, contrast=10, tone_strength=70, tone_color="#d8b070"),
    "Cyanotype": AdjustParams(mono=True, contrast=14, tone_strength=75, tone_color="#5b8fc9"),
}
PRESET_NAMES = [PRESET_NONE] + list(PRESETS)


def preset(name: str) -> AdjustParams:
    """The params for a preset name; unknown names and "None" reset to neutral."""
    return replace(PRESETS[name]) if name in PRESETS else AdjustParams()


# ── color helpers ────────────────────────────────────────────────────────────
def _hex(c: str) -> np.ndarray:
    s = (c or "#ffffff").lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0
    except ValueError:
        return np.ones(3, dtype=np.float32)


def _to_linear(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 0.0, 1.0)
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def _to_srgb(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 0.0, 1.0)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)


def _luma(a: np.ndarray) -> np.ndarray:
    return (a * LUMA).sum(axis=2, keepdims=True)


def _neutral_gains(gains: np.ndarray) -> np.ndarray:
    """Scale channel gains so a neutral grey keeps its brightness."""
    return gains / max(1e-6, float((gains * LUMA).sum()))


def _hue_matrix(deg: float) -> np.ndarray:
    """The SVG feColorMatrix hue-rotation matrix (rows = output channels)."""
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([
        [0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928],
        [0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283],
        [0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072],
    ], dtype=np.float32)


def _push(a: np.ndarray, amount: float, weight: np.ndarray) -> np.ndarray:
    """Move ``a`` toward white (amount > 0) or black (amount < 0) by ``amount``,
    scaled per pixel by ``weight``. Never clips: it always lands inside 0..1."""
    if amount == 0:
        return a
    k = amount / 100.0
    return a + (weight * k * ((1.0 - a) if k > 0 else a))


# ── the pipeline ──────────────────────────────────────────────────────────────
def adjust_image(img: Image.Image, p: AdjustParams) -> Image.Image:
    """Apply every adjustment to the whole image (RGB in, RGB out)."""
    rgb = img.convert("RGB")
    a = np.asarray(rgb, dtype=np.float32) / 255.0

    # 1. exposure — in linear light, so ±100 really is ±2 stops
    if p.exposure:
        a = _to_srgb(_to_linear(a) * (2.0 ** (p.exposure / 100.0 * MAX_EV)))

    # 2. levels — stretch the range between the black and white points
    lo = np.clip(p.black_point, 0.0, 99.0) / 100.0
    hi = np.clip(p.white_point, 1.0, 100.0) / 100.0
    if lo > 0 or hi < 1.0:
        a = np.clip((a - lo) / max(1e-3, hi - lo), 0.0, 1.0)

    # 3. gamma — midtone brightness
    g = float(np.clip(p.gamma, 0.2, 3.0))
    if abs(g - 1.0) > 1e-3:
        a = np.clip(a, 0.0, 1.0) ** (1.0 / g)

    # 4. highlights / shadows — weighted by how bright each pixel already is
    if p.highlights or p.shadows:
        lum = _luma(np.clip(a, 0.0, 1.0))
        hi_w = np.clip((lum - 0.5) * 2.0, 0.0, 1.0) ** 2
        sh_w = np.clip((0.5 - lum) * 2.0, 0.0, 1.0) ** 2
        a = _push(a, p.highlights, hi_w)
        a = _push(a, p.shadows, sh_w)

    # 5. contrast — a straight pull around mid grey
    if p.contrast:
        a = (a - 0.5) * (1.0 + p.contrast / 100.0) + 0.5

    # 6. clarity — local contrast: add back what a wide blur removed
    if p.clarity:
        a = np.clip(a, 0.0, 1.0)
        radius = max(2.0, min(rgb.size) * 0.03)
        base = np.asarray(
            Image.fromarray((a * 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(radius)),
            dtype=np.float32) / 255.0
        a = a + (a - base) * (p.clarity / 100.0)

    # 7. white balance — channel gains in linear light, brightness preserved
    if p.temperature or p.tint:
        t, ti = p.temperature / 100.0, p.tint / 100.0
        gains = _neutral_gains(np.array([1.0 + 0.30 * t + 0.12 * ti,
                                         1.0 - 0.22 * ti,
                                         1.0 - 0.30 * t + 0.12 * ti], dtype=np.float32))
        a = _to_srgb(_to_linear(a) * gains)

    # 8. hue, then vibrance, then saturation
    if p.hue:
        a = np.clip(a, 0.0, 1.0) @ _hue_matrix(p.hue).T
    if p.vibrance or p.saturation:
        a = np.clip(a, 0.0, 1.0)
        lum = _luma(a)
        if p.vibrance:
            # Weight by how saturated the pixel already is, the HSV way:
            # (max - min) / max. Muted colors get nearly the full boost, vivid
            # ones and skin barely move. Raw channel spread would under-boost
            # pale pastels, which are exactly what vibrance is for.
            mx = a.max(axis=2, keepdims=True)
            sat = (mx - a.min(axis=2, keepdims=True)) / np.clip(mx, 1e-3, None)
            a = lum + (a - lum) * (1.0 + p.vibrance / 100.0 * (1.0 - sat))
            lum = _luma(np.clip(a, 0.0, 1.0))
        if p.saturation:
            a = lum + (a - lum) * (1.0 + p.saturation / 100.0)

    # 9. black & white, with an optional split tone
    if p.mono:
        w = np.array([p.mono_red, p.mono_green, p.mono_blue], dtype=np.float32)
        total = float(w.sum())
        w = w / total if abs(total) > 1e-6 else LUMA.copy()
        grey = (np.clip(a, 0.0, 1.0) * w).sum(axis=2, keepdims=True)
        a = np.repeat(grey, 3, axis=2)
        if p.tone_strength:
            gains = _neutral_gains(_hex(p.tone_color))
            a = a * (1.0 + (gains - 1.0) * (p.tone_strength / 100.0))

    return Image.fromarray((np.clip(a, 0.0, 1.0) * 255).round().astype(np.uint8), "RGB")


def apply(img: Image.Image, p: AdjustParams,
          m: blur.MaskParams | None = None) -> Image.Image:
    """Adjust ``img`` through an optional mask. Alpha is carried through."""
    alpha = img.getchannel("A") if "A" in img.getbands() else None
    rgb = img.convert("RGB")
    out = adjust_image(rgb, p)
    if m is not None and (m.shape != "whole" or m.outside):
        weight = np.asarray(blur.build_mask(rgb.size, m), dtype=np.float32)[..., None] / 255.0
        base = np.asarray(rgb, dtype=np.float32)
        out = Image.fromarray(
            np.clip(base + (np.asarray(out, dtype=np.float32) - base) * weight, 0, 255)
            .astype(np.uint8), "RGB")
    if alpha is not None:
        out = out.convert("RGBA")
        out.putalpha(alpha)
    return out


# ── auto ──────────────────────────────────────────────────────────────────────
def auto_params(img: Image.Image, base: AdjustParams | None = None) -> AdjustParams:
    """Read the picture and return params that level it out: black and white
    points from its actual range, a gamma that puts the midtone where the eye
    expects it, and a grey-world white balance. Everything else in ``base`` is
    kept, including exposure — auto works through levels and gamma instead, so
    the three stay consistent with the order they're applied in."""
    p = replace(base) if base is not None else AdjustParams()
    small = img.convert("RGB")
    if max(small.size) > 512:
        small = small.resize((max(1, small.width * 512 // max(small.size)),
                              max(1, small.height * 512 // max(small.size))), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32) / 255.0

    lum = _luma(a).ravel()
    lo, hi = (float(v) for v in np.percentile(lum, [0.1, 99.9]))
    if hi - lo < 0.02:                      # a nearly flat image: leave levels alone
        lo, hi = 0.0, 1.0
    p.black_point = float(np.clip(lo * 100.0, 0.0, 25.0))
    p.white_point = float(np.clip(hi * 100.0, 60.0, 100.0))

    # gamma: move the median half-way toward 0.45 — auto should help, not
    # force every photo to the same key (a low-key shot may be dark on purpose)
    median = float(np.median(np.clip((lum - lo) / max(1e-3, hi - lo), 0.0, 1.0)))
    if 1e-3 < median < 0.999:
        want = np.log(median) / np.log(0.45)
        p.gamma = float(np.clip(1.0 + (want - 1.0) * 0.5, 0.7, 1.5))

    # white balance: grey-world, but only over pixels that should BE grey —
    # near-neutral midtones. Averaging a blue sky would "correct" the sky away.
    flat = a.reshape(-1, 3)
    mx, mn = flat.max(axis=1), flat.min(axis=1)
    sat = (mx - mn) / np.clip(mx, 1e-3, None)
    neutral = flat[(sat < 0.25) & (mx > 0.15) & (mx < 0.95)]
    if len(neutral) >= max(64, int(0.01 * len(flat))):
        means = np.clip(neutral.mean(axis=0), 1e-3, None)
        gains = float(means.mean()) / means
        # half strength: a nudge toward neutral, not a forced match
        p.temperature = float(np.clip((gains[0] - gains[2]) / 0.60 * 50.0, -40.0, 40.0))
        p.tint = float(np.clip((1.0 - gains[1]) / 0.22 * 50.0, -40.0, 40.0))
    return p


# ── preview + description ─────────────────────────────────────────────────────
def preview_pair(img: Image.Image, p: AdjustParams, m: blur.MaskParams | None = None,
                 max_edge: int = PREVIEW_EDGE) -> tuple[Image.Image, Image.Image]:
    """(before, after) at preview size — every setting is relative, so it looks
    like the full-size export."""
    scale = min(1.0, max_edge / max(img.size))
    small = img if scale >= 1.0 else img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    return small, apply(small, p, m)


_LABELS = {
    "exposure": "exposure", "contrast": "contrast", "highlights": "highlights",
    "shadows": "shadows", "clarity": "clarity", "temperature": "temperature",
    "tint": "tint", "hue": "hue", "saturation": "saturation", "vibrance": "vibrance",
}


def _num(v: float, digits: int = 1, sign: bool = False) -> str:
    """A tidy number: 6 not 6.0, -15.5 not -15.476001."""
    text = format(round(float(v), digits), "g")
    return f"+{text}" if sign and v > 0 else text


def describe(p: AdjustParams, m: blur.MaskParams | None = None) -> str:
    """A short human summary of what's actually turned on."""
    neutral = AdjustParams()
    bits = [f"{label} {_num(getattr(p, name), sign=True)}"
            for name, label in _LABELS.items() if getattr(p, name) != getattr(neutral, name)]
    if p.black_point or p.white_point != 100.0:
        bits.append(f"levels {_num(p.black_point)}–{_num(p.white_point)}")
    if abs(p.gamma - 1.0) > 1e-3:
        bits.append(f"gamma {_num(p.gamma, 2)}")
    if p.mono:
        bits.append("black & white" + (f" · {_num(p.tone_strength)}% tone" if p.tone_strength else ""))
    where = ""
    if m is not None and (m.shape != "whole" or m.outside):
        shape = m.shape
        if shape == "faces":
            n = len(m.faces or [])
            shape = f"{n} face{'s' if n != 1 else ''}"
        where = f" · {'outside' if m.outside else 'inside'} the {shape}"
    return (", ".join(bits) or "no changes") + where
