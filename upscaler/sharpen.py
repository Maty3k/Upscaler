"""Sharpening stages.

Phase 1 ships the classical *unsharp mask* — zero extra weights, instant, and
good for mild softness. A model-based deblur stage (e.g. NAFNet) is planned for
Phase 2 for genuinely blurry input; it will slot in alongside this with the same
(image, strength) -> image signature.
"""

from __future__ import annotations

from PIL import Image, ImageFilter


def unsharp_mask(
    image: Image.Image,
    strength: float = 1.0,
    radius: float = 2.0,
    threshold: int = 3,
) -> Image.Image:
    """Sharpen via unsharp masking.

    ``strength`` is a convenience multiplier on the standard ``percent`` amount
    (1.0 -> ~150%). ``radius`` controls the scale of detail enhanced; ``threshold``
    avoids amplifying flat-area noise.
    """
    if strength <= 0:
        return image
    percent = int(150 * strength)
    return image.convert("RGB").filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )


# ── The sharpening toolbox ────────────────────────────────────────────────────
# Four kinds, one detail signal. Everything below works on a "detail layer" —
# the picture minus a blurred copy of itself — and differs only in how that
# layer is shaped and added back:
#
#   unsharp    add it straight (the classic)
#   high-pass  overlay-blend it, which lifts edges without flattening tone
#   smart      gate it by an edge mask, so flat sky and skin keep their noise
#              instead of having it amplified
#   texture    two scales at once: fine grit plus a little structure
#
# Unlike the other tabs, the radius here is in PIXELS, not a percentage of the
# image. Sharpening is about the scale of real detail — a photo's edges are a
# pixel or two wide whatever the frame size — and a relative radius turns into
# a 6px smear on a 4K file, which is structure enhancement, not sharpening.
# That means the live preview cannot be a downscaled whole image: shrinking a
# photo changes its detail scale, so a downscaled preview would lie. It shows a
# 1:1 centre crop instead, which is how sharpening has to be judged anyway.

from dataclasses import dataclass, replace  # noqa: E402

import numpy as np  # noqa: E402

from upscaler import blur  # noqa: E402

KINDS = ["unsharp", "high-pass", "smart", "texture"]
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
PREVIEW_EDGE = blur.PREVIEW_EDGE
MIN_RADIUS, MAX_RADIUS = 0.3, 10.0     # pixels


@dataclass
class SharpenParams:
    kind: str = "unsharp"
    amount: float = 80.0          # 0..300 (100 ≈ a classic 100% unsharp mask)
    radius: float = 1.0           # pixels — the width of the edges you're lifting
    threshold: float = 4.0        # 0..100, leave flat areas alone
    halo: float = 35.0            # 0..100, cap the bright/dark rim at an edge
    luminance_only: bool = True   # sharpen brightness, not color
    protect_shadows: float = 0.0     # 0..100
    protect_highlights: float = 0.0  # 0..100
    edge_sensitivity: float = 50.0   # "smart": how strictly it follows edges
    detail_balance: float = 50.0     # "texture": fine (0) … structure (100)

    def is_identity(self) -> bool:
        return self.amount <= 0


PRESET_NONE = "None"
PRESETS: dict[str, SharpenParams] = {
    "Subtle": SharpenParams(amount=50, radius=0.8, threshold=5, halo=25),
    "Standard": SharpenParams(amount=90, radius=1.0, threshold=4, halo=35),
    "Punchy": SharpenParams(amount=160, radius=1.2, threshold=3, halo=55),
    "Fine detail": SharpenParams(kind="texture", amount=120, radius=0.7, threshold=3,
                                 halo=30, detail_balance=25),
    "Portrait (skin-safe)": SharpenParams(kind="smart", amount=110, radius=1.0, threshold=12,
                                          halo=25, edge_sensitivity=70,
                                          protect_highlights=25),
    "Landscape": SharpenParams(kind="texture", amount=130, radius=1.1, threshold=4,
                               halo=45, detail_balance=60),
    "After upscaling": SharpenParams(kind="smart", amount=90, radius=1.8, threshold=6,
                                     halo=30, edge_sensitivity=45),
    "Rescue a soft shot": SharpenParams(kind="high-pass", amount=180, radius=2.2,
                                        threshold=2, halo=70),
}
PRESET_NAMES = [PRESET_NONE] + list(PRESETS)


def preset(name: str) -> SharpenParams:
    """The params for a preset; unknown names and "None" reset to neutral."""
    return replace(PRESETS[name]) if name in PRESETS else SharpenParams(amount=0.0)


def radius_px(radius: float) -> float:
    """The blur radius in pixels, clamped to a sane range."""
    return float(np.clip(radius, MIN_RADIUS, MAX_RADIUS))


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _gauss(a: np.ndarray, r: float) -> np.ndarray:
    """Gaussian-blur a float 0..1 array with 1 or 3 channels, via PIL."""
    mode = "L" if a.shape[2] == 1 else "RGB"
    arr = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr[..., 0] if mode == "L" else arr, mode)
    out = np.asarray(img.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32) / 255.0
    return out[..., None] if mode == "L" else out


def _detail(a: np.ndarray, r: float) -> np.ndarray:
    return a - _gauss(a, r)


def _edge_weight(a: np.ndarray, r: float, sensitivity: float) -> np.ndarray:
    """1 on an edge, 0 on a flat area — the gate the "smart" kind uses."""
    lum = (a * LUMA).sum(axis=2, keepdims=True) if a.shape[2] == 3 else a
    gy, gx = np.gradient(lum[..., 0])
    mag = np.sqrt(gx * gx + gy * gy)[..., None]
    mag = _gauss(np.clip(mag * 4.0, 0, 1), max(1.0, r))
    # Higher sensitivity = stricter: the gradient has to be stronger before
    # the gate opens, so more of the flat area (skin, sky, noise) is spared.
    knee = 0.02 + np.clip(sensitivity, 0, 100) / 100.0 * 0.28
    return _smoothstep(mag / max(1e-3, knee))


def sharpen_image(img: Image.Image, p: SharpenParams) -> Image.Image:
    """Sharpen the whole image (RGB in, RGB out)."""
    if p.kind not in KINDS:
        raise ValueError(f"unknown sharpen kind {p.kind!r} — expected one of {KINDS}")
    rgb = img.convert("RGB")
    if p.amount <= 0:
        return rgb
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    r = radius_px(p.radius)
    amount = max(0.0, float(p.amount)) / 100.0

    # The detail layer: on luma alone by default, so edges gain definition
    # without the colour fringing that per-channel sharpening leaves behind.
    base = (a * LUMA).sum(axis=2, keepdims=True) if p.luminance_only else a
    if p.kind == "texture":
        fine = _detail(base, r)
        coarse = _detail(base, r * 3.0)
        mix = np.clip(p.detail_balance, 0, 100) / 100.0
        d = fine * (1.0 - mix) + coarse * mix
    else:
        d = _detail(base, r)

    # Threshold: fade out detail smaller than this, so grain and noise in flat
    # areas are left alone instead of being amplified.
    t = np.clip(p.threshold, 0.0, 100.0) / 100.0 * 0.25
    if t > 0:
        d = d * _smoothstep((np.abs(d) - t * 0.5) / max(1e-4, t * 0.5))

    if p.kind == "smart":
        d = d * _edge_weight(a, r, p.edge_sensitivity)


    # Leave the deepest shadows and brightest highlights alone if asked.
    if p.protect_shadows or p.protect_highlights:
        lum = (a * LUMA).sum(axis=2, keepdims=True)
        if p.protect_shadows:
            k = np.clip(p.protect_shadows, 0, 100) / 100.0
            d = d * (1.0 - k * (1.0 - _smoothstep(lum / 0.35)))
        if p.protect_highlights:
            k = np.clip(p.protect_highlights, 0, 100) / 100.0
            d = d * (1.0 - k * _smoothstep((lum - 0.65) / 0.35))

    # Halo control: cap how far an edge may overshoot into a bright or dark
    # rim — this is what separates sharpening from an outlined look. It is
    # applied *after* the amount, so the number means what it says: halo 20
    # allows an edge to swing at most 20% of the full range.
    d = d * amount
    cap = np.clip(p.halo, 0.0, 100.0) / 100.0 * 0.5
    if cap > 0:
        d = np.clip(d, -cap, cap)

    if p.kind == "high-pass":
        # Overlay-blend a 50% grey lifted by the detail layer: edges gain
        # contrast while the overall tone of the picture stays put.
        hp = np.clip(0.5 + d, 0.0, 1.0)
        low = a * hp * 2.0
        high = 1.0 - 2.0 * (1.0 - a) * (1.0 - hp)
        out = np.where(a <= 0.5, low, high)
    else:
        out = a + d

    return Image.fromarray((np.clip(out, 0.0, 1.0) * 255).round().astype(np.uint8), "RGB")


def apply(img: Image.Image, p: SharpenParams,
          m: "blur.MaskParams | None" = None) -> Image.Image:
    """Sharpen through an optional region mask. Alpha is carried through."""
    alpha = img.getchannel("A") if "A" in img.getbands() else None
    rgb = img.convert("RGB")
    out = sharpen_image(rgb, p)
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


def _crop_origin(src_size: "tuple[int, int]", crop_size: "tuple[int, int]",
                 center: "tuple[float, float]") -> "tuple[int, int]":
    """Top-left of a ``crop_size`` window centred on ``center`` inside ``src_size``."""
    (sw, sh), (cw, ch) = src_size, crop_size
    return (int(round(np.clip(sw * center[0] - cw / 2, 0, sw - cw))),
            int(round(np.clip(sh * center[1] - ch / 2, 0, sh - ch))))


def preview_crop(img: Image.Image, max_edge: int = PREVIEW_EDGE,
                 center: "tuple[float, float]" = (0.5, 0.5)) -> Image.Image:
    """A 1:1 window around ``center`` (fractions of the image), at most
    ``max_edge`` across. Returns the image itself when it already fits."""
    if max(img.size) <= max_edge:
        return img
    cw, ch = min(img.width, max_edge), min(img.height, max_edge)
    left, top = _crop_origin(img.size, (cw, ch), center)
    return img.crop((left, top, left + cw, top + ch))


def preview_pair(img: Image.Image, p: SharpenParams, m: "blur.MaskParams | None" = None,
                 max_edge: int = PREVIEW_EDGE,
                 center: "tuple[float, float]" = (0.5, 0.5)) -> "tuple[Image.Image, Image.Image]":
    """(before, after) at 1:1 — a centre crop of a big photo, never a
    downscale, because shrinking a picture changes the very detail scale the
    radius is measured in. What you see is exactly what the export does.

    A region mask is measured against the whole image, so it is built at full
    size and the same window cut out of it; that keeps the region lined up
    without paying to render the whole frame on every slider move.
    """
    crop = preview_crop(img, max_edge, center)
    if crop is img or m is None or (m.shape == "whole" and not m.outside):
        return crop, apply(crop, p, m)
    left, top = _crop_origin(img.size, crop.size, center)
    window = blur.build_mask(img.size, m).crop(
        (left, top, left + crop.width, top + crop.height))
    return crop, apply(crop, p, blur.MaskParams(shape="painted", painted=window, feather=0))


def describe(p: SharpenParams, m: "blur.MaskParams | None" = None,
             size: "tuple[int, int] | None" = None) -> str:
    """A short human summary of the settings."""
    if p.amount <= 0:
        return "no sharpening"
    bits = [f"{p.kind} · amount {p.amount:g}", f"radius {radius_px(p.radius):g}px"]
    if p.threshold:
        bits.append(f"threshold {p.threshold:g}")
    bits.append(f"halo {p.halo:g}")
    if p.luminance_only:
        bits.append("luminance only")
    if p.protect_shadows or p.protect_highlights:
        bits.append(f"protect {p.protect_shadows:g}/{p.protect_highlights:g}")
    where = ""
    if m is not None and (m.shape != "whole" or m.outside):
        shape = m.shape
        if shape == "faces":
            n = len(m.faces or [])
            shape = f"{n} face{'s' if n != 1 else ''}"
        where = f" · {'outside' if m.outside else 'inside'} the {shape}"
    return " · ".join(bits) + where
