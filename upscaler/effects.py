"""Effects and film looks — grain, vignette, halation, light leaks, duotone,
posterize, dither, halftone, scanlines, glitch and chromatic aberration.

Pure PIL + numpy (no AI). Unlike the Blur tab, where you pick one kind, these
*stack*: a film look is grain plus halation plus a vignette plus a leak, so
every effect has its own amount and 0 turns it off. They run in a fixed order,
which is what keeps a combination predictable:

    duotone → posterize → dither → halftone → chromatic aberration → halation
    → light leak → scanlines → glitch → vignette → grain

Grain lands last because on a print it sits on top of everything, and the
vignette just before it so the grain isn't darkened along with the corners.

Every size is relative to the image's short side, so one setting looks the
same on a phone snap and a 4K frame, and the reduced-size live preview matches
the full-size export. Regions reuse the Blur tab's masks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageFilter

from upscaler import blur

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
PREVIEW_EDGE = blur.PREVIEW_EDGE

# A Bayer 8×8 threshold matrix, normalised to 0..1 — the ordered dither.
_BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32) / 64.0


@dataclass
class EffectParams:
    # ── film ──
    grain: float = 0.0            # 0..100
    grain_size: float = 1.0       # 1 = per pixel, higher = coarser clumps
    halation: float = 0.0         # 0..100, glow bleeding out of highlights
    halation_threshold: float = 65.0   # how bright a pixel must be to glow
    halation_radius: float = 2.0  # % of the short side
    halation_color: str = "#ff5522"
    leak: float = 0.0             # 0..100, a colored wash from one side
    leak_angle: float = 45.0      # degrees; 0 = from the left
    leak_color: str = "#ff8a3d"
    leak_softness: float = 60.0   # 0 = a hard edge, 100 = the whole frame
    # ── lens ──
    vignette: float = 0.0         # -100 (bright corners) .. 100 (dark corners)
    vignette_radius: float = 60.0   # % of the frame that stays untouched
    vignette_feather: float = 50.0  # how gradually it falls off
    aberration: float = 0.0       # 0..100, red/blue fringing toward the corners
    # ── print ──
    duotone: float = 0.0          # 0..100 blend
    duotone_dark: str = "#1b2a4a"
    duotone_light: str = "#ffd9a0"
    posterize: int = 0            # 0 = off, else the number of levels (2..32)
    dither: float = 0.0           # 0..100, ordered dithering into `dither_levels`
    dither_levels: int = 4
    halftone: float = 0.0         # 0..100 blend toward a dot screen
    halftone_cell: float = 1.0    # % of the short side
    halftone_angle: float = 45.0
    # ── screen ──
    scanlines: float = 0.0        # 0..100
    scanline_spacing: float = 3.0   # pixels between lines (at 1× preview scale)
    glitch: float = 0.0           # 0..100, displaced bands + channel shift
    glitch_seed: int = 7

    def is_identity(self) -> bool:
        """True when nothing would change the picture."""
        return not any((self.grain, self.halation, self.leak, self.vignette,
                        self.aberration, self.duotone, self.posterize, self.dither,
                        self.halftone, self.scanlines, self.glitch))


# ── looks ─────────────────────────────────────────────────────────────────────
PRESET_NONE = "None"
PRESETS: dict[str, EffectParams] = {
    "Film grain": EffectParams(grain=22, grain_size=1.4, vignette=18, vignette_radius=70),
    "Cinematic halation": EffectParams(halation=55, halation_threshold=60, halation_radius=2.5,
                                       grain=12, vignette=25),
    "Faded vintage": EffectParams(grain=30, grain_size=1.8, leak=35, leak_angle=20,
                                  vignette=30, halation=20),
    "Lomo": EffectParams(vignette=65, vignette_radius=45, vignette_feather=60,
                         aberration=25, grain=18),
    "Dreamy glow": EffectParams(halation=60, halation_threshold=40, halation_radius=5.0,
                                halation_color="#ffe8d5", vignette=-15),
    "Sun leak": EffectParams(leak=60, leak_angle=135, leak_softness=45, halation=30, grain=14),
    "Newspaper print": EffectParams(halftone=100, halftone_cell=0.8, posterize=0, grain=8),
    "Comic halftone": EffectParams(halftone=85, halftone_cell=1.4, posterize=6),
    "Duotone blue": EffectParams(duotone=100, duotone_dark="#10203f", duotone_light="#f2c76b",
                                 grain=10),
    "Retro CRT": EffectParams(scanlines=45, scanline_spacing=3, aberration=18, vignette=30,
                              halation=25),
    "VHS glitch": EffectParams(glitch=45, aberration=35, scanlines=30, grain=20,
                               vignette=20),
    "8-color dither": EffectParams(dither=100, dither_levels=2),
}
PRESET_NAMES = [PRESET_NONE] + list(PRESETS)


def preset(name: str) -> EffectParams:
    """The params for a look; unknown names and "None" reset to neutral."""
    return replace(PRESETS[name]) if name in PRESETS else EffectParams()


# ── helpers ───────────────────────────────────────────────────────────────────
def _hex(c: str, default=(1.0, 1.0, 1.0)) -> np.ndarray:
    s = (c or "").lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0
    except (ValueError, IndexError):
        return np.array(default, dtype=np.float32)


def _luma(a: np.ndarray) -> np.ndarray:
    return (a * LUMA).sum(axis=2, keepdims=True)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _blur_array(a: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian-blur a float 0..1 RGB array by going through PIL."""
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGB")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


def _norm_coords(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates scaled to -1..1 across each axis (centre = 0)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return (xx - (w - 1) / 2) / max(1.0, (w - 1) / 2), (yy - (h - 1) / 2) / max(1.0, (h - 1) / 2)


# ── the effects (float 0..1 RGB in and out) ───────────────────────────────────
def _duotone(a, p):
    dark, light = _hex(p.duotone_dark, (0, 0, 0)), _hex(p.duotone_light, (1, 1, 1))
    lum = _luma(a)
    mapped = dark + (light - dark) * lum
    return a + (mapped - a) * (p.duotone / 100.0)


def _posterize(a, p):
    levels = int(np.clip(p.posterize, 2, 32))
    return np.round(np.clip(a, 0, 1) * (levels - 1)) / (levels - 1)


def _dither(a, p):
    levels = int(np.clip(p.dither_levels, 2, 16))
    h, w = a.shape[:2]
    tile = np.tile(_BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w, None]
    q = np.clip(a, 0, 1) * (levels - 1)
    out = np.floor(q + tile) / (levels - 1)
    return a + (np.clip(out, 0, 1) - a) * (p.dither / 100.0)


def _halftone(a, p, short: int):
    """A rotated dot screen: each cell's dot grows as that area gets darker."""
    cell = max(2.0, p.halftone_cell / 100.0 * short)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = math.radians(p.halftone_angle)
    xr = xx * math.cos(ang) + yy * math.sin(ang)
    yr = -xx * math.sin(ang) + yy * math.cos(ang)
    fx = np.mod(xr, cell) / cell - 0.5
    fy = np.mod(yr, cell) / cell - 0.5
    dist = np.sqrt(fx * fx + fy * fy) * 2.0          # 0 at a dot centre, ~1.41 at a corner
    # Sample the tone at the dot's scale, so a cell gets one ink level.
    tone = _luma(_blur_array(a, cell / 2.0))[..., 0]
    # A dot of radius r covers pi*r^2/4 of its cell (dist is 1 at half a cell
    # across), so r = 1.128*sqrt(ink) makes 50% grey print as 50% ink instead
    # of running dark.
    dots = (dist < 1.128 * np.sqrt(np.clip(1.0 - tone, 0, 1)))
    screen = np.repeat((~dots).astype(np.float32)[..., None], 3, axis=2)
    return a + (screen - a) * (p.halftone / 100.0)


def _aberration(a, p):
    """Scale the red and blue channels apart around the centre, so colour
    fringes appear toward the corners the way a cheap lens does."""
    k = p.aberration / 100.0 * 0.01
    h, w = a.shape[:2]
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGB")
    chans = list(img.split())
    for idx, scale in ((0, 1.0 + k), (2, 1.0 - k)):
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = chans[idx].resize((sw, sh), Image.BILINEAR)
        if scale >= 1.0:
            left, top = (sw - w) // 2, (sh - h) // 2
            chans[idx] = resized.crop((left, top, left + w, top + h))
        else:
            canvas = Image.new("L", (w, h))
            canvas.paste(resized, ((w - sw) // 2, (h - sh) // 2))
            chans[idx] = canvas
    return np.asarray(Image.merge("RGB", chans), dtype=np.float32) / 255.0


def _halation(a, p, short: int):
    """Bloom the highlights and add them back, tinted — the glow that spills
    around bright areas on film."""
    t = np.clip(p.halation_threshold / 100.0, 0.0, 0.99)
    lum = _luma(a)
    weight = np.clip((lum - t) / max(1e-3, 1.0 - t), 0.0, 1.0)
    glow = _blur_array(a * weight, max(1.0, p.halation_radius / 100.0 * short))
    # The tint is a colour, not a filter: normalise it so a deep orange glows
    # as brightly as a white one instead of only surviving in the red channel.
    tint = _hex(p.halation_color)
    tint = tint / max(0.35, float(tint.max()))
    return a + glow * tint * (p.halation / 100.0 * 2.0)


def _leak(a, p):
    """A coloured wash across the frame, screen-blended so it lightens rather
    than covers — a light leak down one side of the film."""
    h, w = a.shape[:2]
    u, v = _norm_coords(h, w)
    ang = math.radians(p.leak_angle)
    t = (u * math.cos(ang) + v * math.sin(ang) + 1.0) / 2.0     # 0..1 across the frame
    soft = np.clip(p.leak_softness, 1.0, 100.0) / 100.0
    weight = _smoothstep((t - (1.0 - soft)) / soft)[..., None]
    leak = _hex(p.leak_color) * weight * (p.leak / 100.0)
    return 1.0 - (1.0 - np.clip(a, 0, 1)) * (1.0 - np.clip(leak, 0, 1))


def _scanlines(a, p, short: int):
    h = a.shape[0]
    spacing = max(2.0, p.scanline_spacing / 1000.0 * short + 1.0)
    rows = (np.arange(h, dtype=np.float32) % spacing) < (spacing / 2.0)
    factor = np.where(rows, 1.0, 1.0 - p.scanlines / 100.0)[:, None, None]
    return a * factor


def _glitch(a, p):
    """Displace random horizontal bands and pull the colour channels apart —
    the look of a damaged tape."""
    h, w = a.shape[:2]
    rng = np.random.default_rng(int(p.glitch_seed))
    amount = p.glitch / 100.0
    out = a.copy()
    bands = max(1, int(amount * 14))
    for _ in range(bands):
        top = int(rng.integers(0, h))
        height = max(1, int(rng.integers(2, max(3, int(h * 0.06)))))
        shift = int(rng.integers(-1, 2) * rng.integers(1, max(2, int(w * 0.06 * amount + 2))))
        bottom = min(h, top + height)
        out[top:bottom] = np.roll(out[top:bottom], shift, axis=1)
        if rng.random() < 0.5:      # tear one channel a little further
            ch = int(rng.integers(0, 3))
            out[top:bottom, :, ch] = np.roll(out[top:bottom, :, ch], shift * 2, axis=1)
    px = max(1, int(amount * w * 0.01))
    out[..., 0] = np.roll(out[..., 0], px, axis=1)
    out[..., 2] = np.roll(out[..., 2], -px, axis=1)
    return out


def _vignette(a, p):
    h, w = a.shape[:2]
    u, v = _norm_coords(h, w)
    r = np.sqrt(u * u + v * v) / math.sqrt(2.0)          # 0 centre, 1 corner
    start = np.clip(p.vignette_radius, 0.0, 99.0) / 100.0
    feather = max(0.02, np.clip(p.vignette_feather, 1.0, 100.0) / 100.0)
    fall = _smoothstep((r - start) / feather)[..., None]
    k = p.vignette / 100.0
    if k > 0:
        return a * (1.0 - fall * k)
    return a + (1.0 - a) * fall * (-k)


def _grain(a, p, short: int):
    """Monochrome film grain, strongest in the midtones (film's shoulder and
    toe hold less grain) and coarsened by `grain_size`."""
    h, w = a.shape[:2]
    rng = np.random.default_rng(1234)
    size = max(1.0, float(p.grain_size))
    if size > 1.0:
        sh, sw = max(1, int(h / size)), max(1, int(w / size))
        small = rng.standard_normal((sh, sw)).astype(np.float32)
        noise = np.asarray(Image.fromarray(
            np.clip(small * 40 + 128, 0, 255).astype(np.uint8), "L"
        ).resize((w, h), Image.BILINEAR), dtype=np.float32)
        noise = (noise - 128.0) / 40.0
    else:
        noise = rng.standard_normal((h, w)).astype(np.float32)
    lum = _luma(np.clip(a, 0, 1))[..., 0]
    weight = 1.0 - (2.0 * lum - 1.0) ** 2               # a hump over the midtones
    return a + (noise * weight * (p.grain / 100.0) * 0.22)[..., None]


def effect_image(img: Image.Image, p: EffectParams) -> Image.Image:
    """Run the whole stack over the image (RGB in, RGB out)."""
    rgb = img.convert("RGB")
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    short = min(rgb.size)

    if p.duotone:
        a = _duotone(a, p)
    if p.posterize:
        a = _posterize(a, p)
    if p.dither:
        a = _dither(a, p)
    if p.halftone:
        a = _halftone(a, p, short)
    if p.aberration:
        a = _aberration(a, p)
    if p.halation:
        a = _halation(a, p, short)
    if p.leak:
        a = _leak(a, p)
    if p.scanlines:
        a = _scanlines(a, p, short)
    if p.glitch:
        a = _glitch(a, p)
    if p.vignette:
        a = _vignette(a, p)
    if p.grain:
        a = _grain(a, p, short)

    return Image.fromarray((np.clip(a, 0.0, 1.0) * 255).round().astype(np.uint8), "RGB")


def apply(img: Image.Image, p: EffectParams,
          m: blur.MaskParams | None = None) -> Image.Image:
    """Apply the stack through an optional mask. Alpha is carried through."""
    alpha = img.getchannel("A") if "A" in img.getbands() else None
    rgb = img.convert("RGB")
    out = effect_image(rgb, p)
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


def preview_pair(img: Image.Image, p: EffectParams, m: blur.MaskParams | None = None,
                 max_edge: int = PREVIEW_EDGE) -> tuple[Image.Image, Image.Image]:
    """(before, after) at preview size — every size is relative, so it looks
    like the full-size export."""
    scale = min(1.0, max_edge / max(img.size))
    small = img if scale >= 1.0 else img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    return small, apply(small, p, m)


_LABELS = [
    ("grain", "grain"), ("halation", "halation"), ("leak", "light leak"),
    ("vignette", "vignette"), ("aberration", "aberration"), ("duotone", "duotone"),
    ("dither", "dither"), ("halftone", "halftone"), ("scanlines", "scanlines"),
    ("glitch", "glitch"),
]


def describe(p: EffectParams, m: blur.MaskParams | None = None) -> str:
    """A short human summary of what's actually turned on."""
    bits = [f"{label} {getattr(p, name):g}" for name, label in _LABELS if getattr(p, name)]
    if p.posterize:
        bits.append(f"posterize {int(p.posterize)} levels")
    where = ""
    if m is not None and (m.shape != "whole" or m.outside):
        shape = m.shape
        if shape == "faces":
            n = len(m.faces or [])
            shape = f"{n} face{'s' if n != 1 else ''}"
        where = f" · {'outside' if m.outside else 'inside'} the {shape}"
    return (", ".join(bits) or "no effects") + where
