"""Crop, straighten and frame — the geometry pass.

Pure PIL + numpy (no AI). Everything happens in one fixed order, because each
step changes what the next one sees:

    orient (90° turns, mirror) → perspective → straighten → crop to aspect
    → exact size → rounded corners → border → shadow

Straighten comes before the aspect crop so the crop is taken from the
*straightened* frame, and it trims to the largest rectangle that fits inside
the rotated picture, so a tilt correction never leaves empty triangles in the
corners. The aspect crop reuses :mod:`upscaler.fit`, which already knows how to
find the biggest box of a given ratio and slide it along the free axis.

The border can be a flat colour or a blurred, zoomed copy of the photo itself —
the fill people use to put a landscape photo in a square post without cropping
it. Sizes are percentages of the short side, so one setting looks the same on a
phone snap and a 4K frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from upscaler import fit

# name → (w, h) ratio, or None for "leave the shape alone".
ASPECTS: dict[str, "tuple[int, int] | None"] = {
    "Original": None,
    "Square · 1:1": (1, 1),
    "Portrait · 4:5": (4, 5),
    "Portrait · 2:3": (2, 3),
    "Story · 9:16": (9, 16),
    "Landscape · 3:2": (3, 2),
    "Landscape · 4:3": (4, 3),
    "Widescreen · 16:9": (16, 9),
    "Ultrawide · 21:9": (21, 9),
    "Custom…": None,
}
DEFAULT_ASPECT = "Original"
CUSTOM_ASPECT = "Custom…"
BORDER_STYLES = ["solid", "blurred photo"]
# "fill" crops the photo to the chosen shape; "fit" keeps the whole photo and
# fills the leftover margin instead — the square post that doesn't cut anything.
CROP_MODES = ["fill", "fit"]
ROTATIONS = [0, 90, 180, 270]
MAX_STRAIGHTEN = 20.0        # degrees either way
MAX_ZOOM = 4.0
PREVIEW_EDGE = 1000


@dataclass
class FrameParams:
    # ── orientation ──
    rotate: int = 0              # 0 / 90 / 180 / 270, clockwise
    flip_h: bool = False
    flip_v: bool = False
    # ── perspective ──
    keystone_h: float = 0.0      # -100..100, lean the left/right edges
    keystone_v: float = 0.0      # -100..100, lean the top/bottom edges
    # ── straighten ──
    straighten: float = 0.0      # degrees, ±MAX_STRAIGHTEN
    # ── crop ──
    aspect: str = DEFAULT_ASPECT
    custom_aspect: str = ""      # "16:10" or "1200x800" when aspect is Custom…
    crop_mode: str = "fill"      # crop to the shape, or fit the whole photo in
    position_x: float = 50.0     # which part survives, % across the free axis
    position_y: float = 50.0
    zoom: float = 1.0            # 1 = the largest box that fits; higher crops in
    # ── output size ──
    out_size: str = ""           # "1920x1080" to land on an exact size
    # ── frame ──
    border: float = 0.0          # % of the short side
    border_style: str = "solid"
    border_color: str = "#ffffff"
    border_blur: float = 60.0    # "blurred photo" style: how soft the fill is
    corner_radius: float = 0.0   # % of the short side
    shadow: float = 0.0          # 0..100, falls on the border

    def is_identity(self) -> bool:
        """True when nothing would change the picture."""
        return self == FrameParams(border_color=self.border_color,
                                   custom_aspect=self.custom_aspect,
                                   out_size=self.out_size,
                                   border_style=self.border_style,
                                   border_blur=self.border_blur,
                                   crop_mode=self.crop_mode)


PRESET_NONE = "None"
PRESETS: dict[str, FrameParams] = {
    "Instagram square": FrameParams(aspect="Square · 1:1"),
    "Instagram portrait": FrameParams(aspect="Portrait · 4:5"),
    "Story / Reel": FrameParams(aspect="Story · 9:16"),
    "Square with blurred fill": FrameParams(aspect="Square · 1:1", crop_mode="fit",
                                            border_style="blurred photo"),
    "Story with blurred fill": FrameParams(aspect="Story · 9:16", crop_mode="fit",
                                           border_style="blurred photo"),
    "Polaroid": FrameParams(aspect="Square · 1:1", border=9, border_color="#fdfdf8",
                            shadow=35),
    "White mat": FrameParams(border=6, border_color="#ffffff", shadow=25),
    "Black mat": FrameParams(border=6, border_color="#111111", shadow=0),
    "Rounded card": FrameParams(corner_radius=6, border=4, border_color="#ffffff",
                                shadow=40),
    "Desktop wallpaper": FrameParams(aspect="Widescreen · 16:9", out_size="2560x1440"),
    "Phone wallpaper": FrameParams(aspect="Story · 9:16", out_size="1080x2400"),
}
PRESET_NAMES = [PRESET_NONE] + list(PRESETS)


def preset(name: str) -> FrameParams:
    """The params for a preset; unknown names and "None" reset to neutral."""
    return replace(PRESETS[name]) if name in PRESETS else FrameParams()


# ── helpers ───────────────────────────────────────────────────────────────────
def _hex(c: str, default=(255, 255, 255)) -> tuple[int, int, int]:
    s = (c or "").lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except (ValueError, IndexError):
        return default


def parse_aspect(text: str) -> "tuple[int, int] | None":
    """Read a custom ratio: "16:10", "16/10", "1200x800" all mean the same
    shape. Returns None for anything unparseable, so a typo falls back to the
    original shape rather than raising at the user."""
    if not text:
        return None
    cleaned = text.strip().lower().replace("×", "x").replace("*", "x")
    cleaned = cleaned.replace(" ", "").replace(",", "")
    for sep in (":", "/", "x"):
        if sep in cleaned:
            parts = cleaned.split(sep)
            if len(parts) != 2:
                return None
            try:
                w, h = float(parts[0]), float(parts[1])
            except ValueError:
                return None
            if w <= 0 or h <= 0:
                return None
            return int(round(w * 1000)), int(round(h * 1000))
    return None


def aspect_ratio(p: FrameParams) -> "tuple[int, int] | None":
    """The (w, h) ratio the crop should use, or None to keep the shape.

    An ``aspect`` that isn't a known name is an error rather than a shrug: a
    typo in a saved recipe would otherwise produce a file that looks finished
    and was never cropped.
    """
    if p.aspect == CUSTOM_ASPECT:
        return parse_aspect(p.custom_aspect)
    if p.aspect not in ASPECTS:
        raise ValueError(f"unknown shape {p.aspect!r} — expected one of "
                         f"{', '.join(ASPECTS)}")
    return ASPECTS[p.aspect]


def perspective_coeffs(dst: list, src: list) -> list:
    """The 8 PIL PERSPECTIVE coefficients mapping output ``dst`` back to
    input ``src`` (each 4 (x, y) corners: TL, TR, BR, BL)."""
    m = []
    for (dx, dy), (sx, sy) in zip(dst, src):
        m.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        m.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    A = np.array(m, dtype=np.float64)
    b = np.array(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(A, b).tolist()


def inscribed_rect(w: float, h: float, angle_deg: float) -> "tuple[int, int]":
    """The largest axis-aligned rectangle that fits inside a ``w``×``h``
    picture rotated by ``angle_deg`` — i.e. how much survives a straighten
    without pulling in the empty corners."""
    if w <= 0 or h <= 0:
        return 0, 0
    angle = math.radians(abs(angle_deg) % 180.0)
    if angle > math.pi / 2:
        angle = math.pi - angle
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    longer_is_w = w >= h
    side_long, side_short = (w, h) if longer_is_w else (h, w)
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half-constrained: the rectangle touches the midpoint of the long side
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if longer_is_w else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return max(1, int(wr)), max(1, int(hr))


# ── the steps ─────────────────────────────────────────────────────────────────
def orient(img: Image.Image, p: FrameParams) -> Image.Image:
    out = img
    if p.rotate % 360:
        out = out.rotate(-(p.rotate % 360), expand=True)   # positive = clockwise
    if p.flip_h:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if p.flip_v:
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
    return out


def keystone(img: Image.Image, p: FrameParams) -> Image.Image:
    """Lean the frame to correct converging verticals (a building shot from
    below) or horizontals.

    The lean pulls one edge in, which would otherwise leave empty wedges along
    it, so the result is trimmed back to the largest rectangle made entirely of
    real pixels — the same bargain straighten makes.
    """
    if not p.keystone_h and not p.keystone_v:
        return img
    w, h = img.size
    kh = float(np.clip(p.keystone_h, -100, 100)) / 100.0 * 0.35
    kv = float(np.clip(p.keystone_v, -100, 100)) / 100.0 * 0.35
    # Each control pulls in exactly one edge: the sign says which.
    top_dx = w * kv / 2 if kv > 0 else 0.0
    bot_dx = w * -kv / 2 if kv < 0 else 0.0
    left_dy = h * kh / 2 if kh > 0 else 0.0
    right_dy = h * -kh / 2 if kh < 0 else 0.0

    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(top_dx, left_dy), (w - top_dx, right_dy),
           (w - bot_dx, h - right_dy), (bot_dx, h - left_dy)]
    out = img.transform((w, h), Image.PERSPECTIVE, perspective_coeffs(dst, src),
                        resample=Image.BICUBIC)
    # Trim by the deepest inset on each axis, so nothing empty survives.
    cut_x, cut_y = int(math.ceil(max(top_dx, bot_dx))), int(math.ceil(max(left_dy, right_dy)))
    if cut_x or cut_y:
        out = out.crop((cut_x, cut_y, max(cut_x + 1, w - cut_x), max(cut_y + 1, h - cut_y)))
    return out


def straighten(img: Image.Image, p: FrameParams) -> Image.Image:
    """Rotate by a small angle and trim to the largest rectangle that stays
    inside the picture, so the corners never come back empty."""
    angle = float(np.clip(p.straighten, -MAX_STRAIGHTEN, MAX_STRAIGHTEN))
    if abs(angle) < 0.01:
        return img
    w, h = img.size
    turned = img.rotate(angle, resample=Image.BICUBIC, expand=True)
    keep_w, keep_h = inscribed_rect(w, h, angle)
    left = (turned.width - keep_w) // 2
    top = (turned.height - keep_h) // 2
    return turned.crop((left, top, left + keep_w, top + keep_h))


def crop_to_aspect(img: Image.Image, p: FrameParams) -> Image.Image:
    """Crop to the chosen ratio, positioned along the free axis, then zoom in
    further if asked. Zoom 1 keeps the largest box that fits."""
    ratio = aspect_ratio(p)
    out = img
    if ratio:
        # Which axis is free depends on whether the source is wider or taller
        # than the target ratio, so both sliders are offered and the one that
        # can't move is simply ignored by fit.crop_box_for_aspect.
        wider = img.width * ratio[1] > ratio[0] * img.height
        pos = (p.position_x if wider else p.position_y) / 100.0
        box = fit.crop_box_for_aspect(img.width, img.height, ratio[0], ratio[1],
                                      position=float(np.clip(pos, 0.0, 1.0)))
        out = img.crop(box)
    zoom = float(np.clip(p.zoom, 1.0, MAX_ZOOM))
    if zoom > 1.0:
        cw, ch = max(1, round(out.width / zoom)), max(1, round(out.height / zoom))
        left = int(round((out.width - cw) * np.clip(p.position_x / 100.0, 0, 1)))
        top = int(round((out.height - ch) * np.clip(p.position_y / 100.0, 0, 1)))
        out = out.crop((left, top, left + cw, top + ch))
    return out


def _background(size: "tuple[int, int]", photo: Image.Image, p: FrameParams) -> Image.Image:
    """What fills the space around the photo: a flat colour, or a zoomed and
    blurred copy of the photo itself."""
    w, h = size
    if p.border_style != "blurred photo":
        return Image.new("RGB", size, _hex(p.border_color))
    scale = max(w / photo.width, h / photo.height)
    big = photo.convert("RGB").resize(
        (max(1, round(photo.width * scale)), max(1, round(photo.height * scale))),
        Image.LANCZOS)
    left, top = (big.width - w) // 2, (big.height - h) // 2
    return big.crop((left, top, left + w, top + h)).filter(
        ImageFilter.GaussianBlur(max(1.0, p.border_blur / 100.0 * 0.06 * min(w, h))))


def fit_to_aspect(img: Image.Image, p: FrameParams) -> Image.Image:
    """Letterbox the whole photo into the chosen shape, filling the margin —
    nothing is cropped away."""
    ratio = aspect_ratio(p)
    if not ratio:
        return img
    rw, rh = ratio
    # The smallest box of this ratio that still contains the whole photo.
    if img.width * rh > rw * img.height:
        w, h = img.width, max(1, round(img.width * rh / rw))
    else:
        w, h = max(1, round(img.height * rw / rh)), img.height
    canvas = _background((w, h), img, p)
    photo = img.convert("RGBA") if "A" in img.getbands() else img
    canvas = canvas.convert("RGBA")
    canvas.paste(photo, ((w - img.width) // 2, (h - img.height) // 2),
                 photo if photo.mode == "RGBA" else None)
    return canvas.convert("RGB")


def _rounded_mask(size: "tuple[int, int]", radius_pct: float) -> "Image.Image | None":
    r = float(np.clip(radius_pct, 0.0, 50.0)) / 100.0 * min(size)
    if r < 0.5:
        return None
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1],
                                           radius=r, fill=255)
    return mask


def add_frame(img: Image.Image, p: FrameParams) -> Image.Image:
    """Rounded corners, then a border, then a shadow on it."""
    photo = img
    mask = _rounded_mask(photo.size, p.corner_radius)
    if mask is not None:
        photo = photo.convert("RGBA")
        photo.putalpha(mask)

    pad = int(round(float(np.clip(p.border, 0.0, 40.0)) / 100.0 * min(img.size)))
    if pad <= 0:
        return photo

    w, h = img.width + 2 * pad, img.height + 2 * pad
    canvas = _background((w, h), img, p)

    shadow = float(np.clip(p.shadow, 0.0, 100.0))
    if shadow > 0:
        blur_r = max(1.0, pad * 0.35)
        offset = max(1, int(pad * 0.18))
        shade = Image.new("L", (w, h), 0)
        shape = mask if mask is not None else Image.new("L", img.size, 255)
        shade.paste(shape, (pad, pad + offset))
        shade = shade.filter(ImageFilter.GaussianBlur(blur_r))
        dark = Image.new("RGB", (w, h), (0, 0, 0))
        alpha = np.asarray(shade, dtype=np.float32) * (shadow / 100.0 * 0.75) / 255.0
        canvas = Image.composite(dark, canvas.convert("RGB"),
                                 Image.fromarray((alpha * 255).astype(np.uint8), "L"))

    canvas = canvas.convert("RGBA")
    canvas.paste(photo, (pad, pad), photo if photo.mode == "RGBA" else None)
    return canvas.convert("RGB") if mask is None else canvas


def apply(img: Image.Image, p: FrameParams) -> Image.Image:
    """Run the whole geometry pass."""
    if p.aspect == CUSTOM_ASPECT and p.custom_aspect and parse_aspect(p.custom_aspect) is None:
        raise ValueError(f"can't read the custom ratio {p.custom_aspect!r} — "
                         "try something like 16:10 or 1200x800")
    out = orient(img, p)
    out = keystone(out, p)
    out = straighten(out, p)
    out = fit_to_aspect(out, p) if p.crop_mode == "fit" else crop_to_aspect(out, p)
    target = fit.parse_target(p.out_size) if p.out_size else None
    if target:
        # The crop already matches the requested shape when an aspect was
        # chosen; otherwise this lands on the exact pixels regardless.
        out = fit.resize_exact(fit.crop(out, *target), *target)
    return add_frame(out, p)


def preview(img: Image.Image, p: FrameParams, max_edge: int = PREVIEW_EDGE) -> Image.Image:
    """The result at preview size. Every measure is relative, so it matches
    the full-size export."""
    scale = min(1.0, max_edge / max(img.size))
    small = img if scale >= 1.0 else img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    return apply(small, p)


def result_size(src_size: "tuple[int, int]", p: FrameParams) -> "tuple[int, int]":
    """What ``apply`` would produce, without doing the work."""
    probe = Image.new("RGB", src_size)
    return apply(probe, p).size


def describe(p: FrameParams, src_size: "tuple[int, int] | None" = None) -> str:
    """A short human summary of the geometry."""
    bits = []
    if p.rotate % 360:
        bits.append(f"rotate {p.rotate % 360}°")
    if p.flip_h or p.flip_v:
        bits.append("mirror " + "+".join(x for x, on in (("h", p.flip_h), ("v", p.flip_v)) if on))
    if p.keystone_h or p.keystone_v:
        bits.append(f"keystone {p.keystone_h:g}/{p.keystone_v:g}")
    if abs(p.straighten) >= 0.01:
        bits.append(f"straighten {p.straighten:g}°")
    ratio = aspect_ratio(p)
    if ratio:
        label = p.custom_aspect if p.aspect == CUSTOM_ASPECT else p.aspect
        bits.append(("fit into " if p.crop_mode == "fit" else "crop to ") + label)
    if p.zoom > 1.0:
        bits.append(f"zoom {p.zoom:g}×")
    if p.out_size and fit.parse_target(p.out_size):
        bits.append(f"exact {p.out_size}")
    if p.corner_radius:
        bits.append(f"rounded {p.corner_radius:g}%")
    if p.border:
        bits.append(f"{p.border_style} border {p.border:g}%")
    if p.shadow:
        bits.append(f"shadow {p.shadow:g}")
    text = ", ".join(bits) or "no changes"
    if src_size:
        w, h = result_size(src_size, p)
        text += f" · {src_size[0]}×{src_size[1]} → {w}×{h}"
    return text
