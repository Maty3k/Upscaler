"""Screenshot beautifier — put the shot on something worth looking at.

A raw screen capture is a rectangle of UI with hard edges, usually ending in a
band of white. Every place you would want to show one — a README, a landing
page, a slide, a post — looks better with the shot floating on a colour field:
padding around it, rounded corners, a soft shadow underneath, and optionally a
window or browser frame so it reads as an application rather than a crop.

The pass is fixed, because each step changes what the next one sees:

    chrome → rounded corners → 3D tilt → spin → shadow → background → canvas

Chrome comes first so the title bar is part of the window and gets rounded and
tilted with it. The shadow is taken from the finished silhouette, so it follows
a tilted, spun shape rather than the rectangle we started with.

Sizes are percentages of the shot's short side, so one preset suits a phone
capture and a 5K display grab alike. Pure PIL + numpy — no AI, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from upscaler import fit, frame
from upscaler.fonts import FONTS, _load_font

# What sits behind the shot.
SOLID, GRADIENT, MESH, BLURRED, TRANSPARENT = (
    "solid", "gradient", "mesh", "blurred", "transparent")
BACKGROUNDS = [GRADIENT, MESH, SOLID, BLURRED, TRANSPARENT]
# Values stay one CLI-friendly word; these are what a person should read.
BACKGROUND_LABELS = {
    GRADIENT: "Gradient", MESH: "Mesh", SOLID: "Solid colour",
    BLURRED: "Blurred copy of the shot", TRANSPARENT: "Transparent",
}

# The frame drawn around the shot, if any.
NO_CHROME = "none"
CHROMES = [NO_CHROME, "window", "window-dark", "browser", "browser-dark"]
CHROME_LABELS = {
    NO_CHROME: "None", "window": "App window", "window-dark": "App window (dark)",
    "browser": "Browser", "browser-dark": "Browser (dark)",
}

# Canvas shapes. "Auto" means the canvas is just the shot plus its padding.
AUTO_ASPECT = "Auto"
CUSTOM_ASPECT = "Custom…"
ASPECTS: "dict[str, tuple[int, int] | None]" = {
    AUTO_ASPECT: None,
    "Square · 1:1": (1, 1),
    "Landscape · 4:3": (4, 3),
    "Widescreen · 16:9": (16, 9),
    "Wide card · 2:1": (2, 1),
    "Portrait · 4:5": (4, 5),
    "Story · 9:16": (9, 16),
    CUSTOM_ASPECT: None,
}
ASPECT_NAMES = list(ASPECTS)

MAX_TILT = 60.0          # degrees of lean, either way
MAX_SPIN = 30.0          # in-plane rotation, either way
PREVIEW_EDGE = 1100


@dataclass
class ShotParams:
    # ── the background ──
    background: str = GRADIENT
    color: str = "#6366f1"       # the one colour for solid; the start of a gradient
    color2: str = "#a855f7"      # the other end
    angle: float = 135.0         # gradient direction in degrees
    # ── the shot ──
    padding: float = 9.0         # % of the shot's short side
    corner_radius: float = 2.0   # % of the shot's short side
    rim: float = 0.0             # 0..100, a hairline highlight around the edge
    shadow: float = 55.0         # 0..100, how dark the shadow under the shot is
    shadow_softness: float = 50.0    # 0..100, a tight drop or a wide ambient pool
    # ── the frame around it ──
    chrome: str = NO_CHROME
    title: str = ""              # what the browser's address bar reads
    # ── in space ──
    tilt_x: float = 0.0          # ±, the top edge leans away from you
    tilt_y: float = 0.0          # ±, the right edge leans away from you
    spin: float = 0.0            # ±, rotation in the plane of the page
    # ── the canvas ──
    aspect: str = AUTO_ASPECT
    custom_aspect: str = ""      # "16:10" or "1200x800" when aspect is Custom…
    out_size: str = ""           # "1600x900" to land on exact pixels


PRESET_NONE = "None"
PRESETS: "dict[str, ShotParams]" = {
    "Indigo mesh": ShotParams(background=MESH, color="#6366f1", color2="#a855f7",
                              padding=10, corner_radius=2.2, shadow=60),
    "Sunset": ShotParams(color="#ff8a3d", color2="#c02b6b", angle=135,
                         padding=10, corner_radius=2.2, shadow=55),
    "Ocean": ShotParams(color="#22d3ee", color2="#1d4ed8", angle=120,
                        padding=10, corner_radius=2.2, shadow=55),
    "Clean white": ShotParams(background=SOLID, color="#f4f4f6", padding=8,
                              corner_radius=1.6, shadow=32, shadow_softness=62),
    "Slate dark": ShotParams(background=SOLID, color="#14141a", padding=8,
                             corner_radius=1.6, rim=34, shadow=70),
    "App window": ShotParams(background=MESH, color="#38bdf8", color2="#6366f1",
                             chrome="window", padding=10, corner_radius=1.8, shadow=60),
    "Dark app window": ShotParams(background=SOLID, color="#101014", chrome="window-dark",
                                  padding=9, corner_radius=1.8, shadow=75),
    "Browser": ShotParams(chrome="browser", title="example.com", color="#818cf8",
                          color2="#e879f9", padding=10, corner_radius=1.8, shadow=58),
    "Tilted 3D": ShotParams(background=MESH, color="#f472b6", color2="#6366f1",
                            tilt_y=24, spin=-3, padding=12, corner_radius=2.2, shadow=65),
    "Blurred backdrop": ShotParams(background=BLURRED, padding=12, corner_radius=2.2,
                                   rim=22, shadow=62),
    "README hero": ShotParams(background=SOLID, color="#ffffff", aspect="Widescreen · 16:9",
                              padding=5, corner_radius=1.4, shadow=30,
                              shadow_softness=60, out_size="1600x900"),
    "Social card": ShotParams(aspect="Wide card · 2:1", color="#4f46e5", color2="#06b6d4",
                              padding=7, corner_radius=1.8, shadow=55,
                              out_size="1600x800"),
    "No background": ShotParams(background=TRANSPARENT, padding=6, corner_radius=2.2,
                                shadow=45),
}
PRESET_NAMES = [PRESET_NONE] + list(PRESETS)


def preset(name: str) -> ShotParams:
    """The params for a preset; unknown names and "None" reset to neutral."""
    return replace(PRESETS[name]) if name in PRESETS else ShotParams()


# ── small helpers ─────────────────────────────────────────────────────────────
def _hex(c: str, default=(0, 0, 0)) -> "tuple[int, int, int]":
    s = (c or "").lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except (ValueError, IndexError):
        return default


def _ui_font(size: int):
    """A plain interface face for the address bar — not the display fonts the
    watermark picker offers."""
    for name in ("Helvetica", "Arial", "Segoe UI", "DejaVu Sans", "Verdana"):
        if name in FONTS:
            return _load_font(name, size)
    return _load_font("", size)


def aspect_ratio(p: ShotParams) -> "tuple[int, int] | None":
    """The canvas ratio, or None to follow the shot."""
    if p.aspect == CUSTOM_ASPECT:
        return frame.parse_aspect(p.custom_aspect)
    return ASPECTS.get(p.aspect)


# ── the window frame ──────────────────────────────────────────────────────────
#   bar, border, address pill, pill text — light and dark.
_CHROME_COLORS = {
    "light": ((233, 233, 238), (214, 214, 220), (252, 252, 253), (122, 122, 130)),
    "dark": ((45, 45, 51), (60, 60, 68), (30, 30, 35), (150, 150, 160)),
}
_DOTS = ((255, 95, 87), (254, 188, 46), (40, 200, 64))


def add_chrome(img: Image.Image, p: ShotParams) -> Image.Image:
    """Wrap the shot in a window or browser frame. Returns it unchanged when
    the chrome is "none"."""
    if p.chrome == NO_CHROME or p.chrome not in CHROMES:
        return img
    dark = p.chrome.endswith("-dark")
    browser = p.chrome.startswith("browser")
    bar_c, edge_c, pill_c, text_c = _CHROME_COLORS["dark" if dark else "light"]

    w, h = img.size
    # A real title bar is a few percent of the window's width, but on a tall
    # narrow capture that would swallow the picture — so cap it by height too.
    bh = max(12, int(round(min(w * 0.055, h * 0.16))))
    edge = max(1, int(round(w * 0.002)))

    cw, ch = w + 2 * edge, h + bh + edge
    canvas = Image.new("RGB", (cw, ch), edge_c)
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, cw - 1, bh - 1], fill=bar_c)

    r = bh * 0.155
    cx, cy = bh * 0.62, bh / 2.0
    for colour in _DOTS:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)
        cx += bh * 0.52

    if browser:
        x0 = cx + bh * 0.30
        x1 = cw - bh * 0.55
        ph = bh * 0.54
        if x1 - x0 > ph:
            d.rounded_rectangle([x0, cy - ph / 2, x1, cy + ph / 2],
                                radius=ph / 2, fill=pill_c)
            if p.title:
                font = _ui_font(max(8, int(ph * 0.60)))
                d.text((x0 + ph * 0.55, cy), p.title, font=font, fill=text_c,
                       anchor="lm")

    canvas.paste(img.convert("RGB"), (edge, bh))
    return canvas


# ── shaping ───────────────────────────────────────────────────────────────────
def _rounded(img: Image.Image, radius_px: float) -> Image.Image:
    """The shot with rounded corners, as RGBA."""
    out = img.convert("RGBA")
    if radius_px < 0.5:
        return out
    mask = Image.new("L", out.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, out.width - 1, out.height - 1],
                                           radius=radius_px, fill=255)
    # Keep any transparency the shot already had rather than overwriting it.
    existing = out.getchannel("A")
    out.putalpha(Image.fromarray(
        (np.asarray(existing, np.uint16) * np.asarray(mask, np.uint16) // 255
         ).astype(np.uint8), "L"))
    return out


def _rim(layer: Image.Image, radius_px: float, strength: float) -> Image.Image:
    """A hairline highlight just inside the edge.

    A dark screenshot on a dark background has no shadow to speak of — black on
    near-black shows nothing — so the only thing that separates the two is a
    lit edge. That is what a real window has, and what this draws.
    """
    strength = float(np.clip(strength, 0.0, 100.0))
    if strength <= 0:
        return layer
    out = layer.convert("RGBA")
    lw = max(1, int(round(min(out.size) * 0.0022)))
    inset = lw / 2.0
    over = Image.new("RGBA", out.size, (0, 0, 0, 0))
    ImageDraw.Draw(over).rounded_rectangle(
        [inset, inset, out.width - 1 - inset, out.height - 1 - inset],
        radius=max(0.0, radius_px - inset), outline=(255, 255, 255,
                                                     int(strength / 100.0 * 235)),
        width=lw)
    keep = out.getchannel("A")
    out = Image.alpha_composite(out, over)
    out.putalpha(keep)            # the stroke must not spill past the rounded corner
    return out


def tilt(layer: Image.Image, p: ShotParams) -> Image.Image:
    """Lean the shot away from the viewer. The output keeps the same box; the
    shot becomes a trapezoid inside it, with transparency around."""
    tx = float(np.clip(p.tilt_x, -MAX_TILT, MAX_TILT)) / MAX_TILT
    ty = float(np.clip(p.tilt_y, -MAX_TILT, MAX_TILT)) / MAX_TILT
    if abs(tx) < 1e-3 and abs(ty) < 1e-3:
        return layer
    w, h = layer.size
    # How much the receding edge shrinks. 0.62 at full lean is steep enough to
    # read as depth without folding the shot away to nothing.
    kx, ky = abs(tx) * 0.62, abs(ty) * 0.62
    left = ky if ty < 0 else 0.0        # a negative yaw pushes the LEFT edge back
    right = ky if ty > 0 else 0.0
    top = kx if tx > 0 else 0.0         # a positive pitch pushes the TOP back
    bottom = kx if tx < 0 else 0.0
    dst = [
        (w * top / 2, h * left / 2),
        (w * (1 - top / 2), h * right / 2),
        (w * (1 - bottom / 2), h * (1 - right / 2)),
        (w * bottom / 2, h * (1 - left / 2)),
    ]
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    return layer.convert("RGBA").transform(
        (w, h), Image.PERSPECTIVE, frame.perspective_coeffs(dst, src),
        resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))


def _spin(layer: Image.Image, p: ShotParams) -> Image.Image:
    angle = float(np.clip(p.spin, -MAX_SPIN, MAX_SPIN))
    if abs(angle) < 0.05:
        return layer
    return layer.convert("RGBA").rotate(-angle, resample=Image.BICUBIC, expand=True)


# ── the background ────────────────────────────────────────────────────────────
def _linear(w: int, h: int, c1, c2, angle: float) -> np.ndarray:
    ang = np.deg2rad(angle)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = xx * np.cos(ang) + yy * np.sin(ang)
    lo, hi = float(proj.min()), float(proj.max())
    t = (proj - lo) / (hi - lo) if hi > lo else np.zeros_like(proj)
    return (np.asarray(c1, np.float32)[None, None, :] * (1 - t)[..., None]
            + np.asarray(c2, np.float32)[None, None, :] * t[..., None])


def _mesh(w: int, h: int, c1, c2, angle: float) -> np.ndarray:
    """The soft multi-blob wash every screenshot tool ships: a base gradient
    with a few wide radial pools of related colour dropped on top.

    The blob positions are fixed rather than random, so the same photo and the
    same colours always give the same picture — a preview you can trust.
    """
    a1, a2 = np.asarray(c1, np.float32), np.asarray(c2, np.float32)
    arr = _linear(w, h, a1, a2, angle)
    mix = (a1 + a2) / 2.0
    blobs = (
        (0.16, 0.20, np.clip(a1 * 1.35 + 30, 0, 255), 0.85),
        (0.84, 0.16, np.clip(a2 * 1.15 + 15, 0, 255), 0.75),
        (0.26, 0.86, np.clip(mix * 0.55, 0, 255), 0.70),
        (0.88, 0.80, np.clip(a2 * 0.60, 0, 255), 0.75),
    )
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    reach = 0.62 * float(np.hypot(w, h))
    for fx, fy, colour, strength in blobs:
        d = np.hypot(xx - fx * w, yy - fy * h) / reach
        weight = (np.clip(1.0 - d, 0.0, 1.0) ** 2) * strength
        arr = arr * (1 - weight[..., None]) + colour[None, None, :] * weight[..., None]
    return arr


def background(size: "tuple[int, int]", shot: Image.Image, p: ShotParams) -> Image.Image:
    """What fills the canvas behind the shot, as RGBA."""
    w, h = size
    kind = p.background if p.background in BACKGROUNDS else GRADIENT
    if kind == TRANSPARENT:
        return Image.new("RGBA", size, (0, 0, 0, 0))
    if kind == SOLID:
        return Image.new("RGBA", size, _hex(p.color) + (255,))
    if kind == BLURRED:
        # The shot itself, zoomed to cover and blurred to a wash, then dimmed so
        # the sharp copy in front still separates from it.
        photo = shot.convert("RGB")
        scale = max(w / photo.width, h / photo.height) * 1.15
        big = photo.resize((max(1, round(photo.width * scale)),
                            max(1, round(photo.height * scale))), Image.LANCZOS)
        left, top = (big.width - w) // 2, (big.height - h) // 2
        wash = big.crop((left, top, left + w, top + h)).filter(
            ImageFilter.GaussianBlur(max(4.0, 0.05 * min(w, h))))
        return Image.eval(wash, lambda v: int(v * 0.82)).convert("RGBA")
    c1, c2 = _hex(p.color), _hex(p.color2)
    arr = _mesh(w, h, c1, c2, p.angle) if kind == MESH else _linear(w, h, c1, c2, p.angle)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB").convert("RGBA")


# ── the whole pass ────────────────────────────────────────────────────────────
def _canvas_size(art: "tuple[int, int]", pad: int,
                 ratio: "tuple[int, int] | None") -> "tuple[int, int]":
    """The padded box, grown (never cropped) to the requested shape."""
    w, h = art[0] + 2 * pad, art[1] + 2 * pad
    if not ratio:
        return w, h
    rw, rh = ratio
    if w * rh > rw * h:          # too wide for the shape: add height
        return w, max(1, round(w * rh / rw))
    return max(1, round(h * rw / rh)), h


def apply(img: Image.Image, p: ShotParams) -> Image.Image:
    """Run the whole pass and return the finished picture."""
    if p.aspect == CUSTOM_ASPECT and p.custom_aspect and \
            frame.parse_aspect(p.custom_aspect) is None:
        raise ValueError(f"can't read the custom ratio {p.custom_aspect!r} — "
                         "try something like 16:10 or 1200x800")
    shot = img
    short = min(shot.size)

    art = add_chrome(shot, p)
    radius = float(np.clip(p.corner_radius, 0.0, 50.0)) / 100.0 * min(art.size)
    art = _rounded(art, radius)
    art = _rim(art, radius, p.rim)
    art = tilt(art, p)
    art = _spin(art, p)

    pad = int(round(float(np.clip(p.padding, 0.0, 60.0)) / 100.0 * short))
    cw, ch = _canvas_size(art.size, pad, aspect_ratio(p))
    canvas = background((cw, ch), shot, p)
    x, y = (cw - art.width) // 2, (ch - art.height) // 2

    strength = float(np.clip(p.shadow, 0.0, 100.0))
    if strength > 0:
        soft = float(np.clip(p.shadow_softness, 0.0, 100.0))
        blur_r = max(1.0, (0.012 + soft / 100.0 * 0.075) * short)
        drop = int(round((0.010 + soft / 100.0 * 0.020) * short))
        shade = Image.new("L", (cw, ch), 0)
        shade.paste(art.getchannel("A"), (x, y + drop))
        shade = shade.filter(ImageFilter.GaussianBlur(blur_r))
        alpha = (np.asarray(shade, np.float32) * (strength / 100.0 * 0.62)).astype(np.uint8)
        dark = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        dark.putalpha(Image.fromarray(alpha, "L"))
        canvas = Image.alpha_composite(canvas, dark)

    canvas.paste(art, (x, y), art)

    if p.out_size:
        target = fit.parse_target(p.out_size)
        if target:
            canvas = fit.resize_exact(fit.crop(canvas, *target), *target)
    return canvas if p.background == TRANSPARENT else canvas.convert("RGB")


def preview(img: Image.Image, p: ShotParams, max_edge: int = PREVIEW_EDGE) -> Image.Image:
    """The result at preview size. Every measure is relative, so it matches the
    full-size export."""
    scale = min(1.0, max_edge / max(img.size))
    small = img if scale >= 1.0 else img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
        Image.LANCZOS)
    return apply(small, p)


def result_size(src_size: "tuple[int, int]", p: ShotParams) -> "tuple[int, int]":
    """What ``apply`` would produce, without doing the expensive work.

    Only the chrome, the spin, the padding and the canvas shape move the edges.
    The background, the corner radius, the rim, the shadow and the 3D lean all
    paint inside a box they don't resize — so the probe drops them, which is
    what makes this cheap enough to run on every slider move.
    """
    probe = replace(p, background=SOLID, corner_radius=0.0, rim=0.0, shadow=0.0,
                    tilt_x=0.0, tilt_y=0.0)
    return apply(Image.new("RGB", src_size), probe).size


def describe(p: ShotParams) -> str:
    """A short human summary of the treatment."""
    bits = []
    if p.chrome != NO_CHROME:
        label = CHROME_LABELS[p.chrome].lower()
        bits.append(f"in {'an' if label[0] in 'aeiou' else 'a'} {label} frame")
    kind = p.background if p.background in BACKGROUNDS else GRADIENT
    if kind in (GRADIENT, MESH):
        bits.append(f"on a {p.color} → {p.color2} {kind}")
    elif kind == SOLID:
        bits.append(f"on {p.color}")
    elif kind == BLURRED:
        bits.append("on a blurred copy of itself")
    else:
        bits.append("on transparency")
    if p.padding:
        bits.append(f"{p.padding:g}% padding")
    if p.corner_radius:
        bits.append(f"{p.corner_radius:g}% corners")
    if p.rim:
        bits.append(f"edge {p.rim:g}")
    if p.shadow:
        bits.append(f"shadow {p.shadow:g}")
    lean = [f"{name} {value:g}°" for name, value in
            (("tilt", p.tilt_y), ("pitch", p.tilt_x), ("spin", p.spin)) if value]
    if lean:
        bits.append(", ".join(lean))
    if p.aspect != AUTO_ASPECT:
        bits.append(p.custom_aspect if p.aspect == CUSTOM_ASPECT else p.aspect)
    if p.out_size:
        bits.append(f"at {p.out_size}")
    return "Screenshot " + " · ".join(bits)
