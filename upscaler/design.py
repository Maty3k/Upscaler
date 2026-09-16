"""Design templates — a finished graphic from a photo and a few words.

Every other tool here changes a picture you already have. This one starts from
a blank canvas of a known size — a YouTube thumbnail, an Instagram post, a
poster — and lays your photo and your words onto it.

A template is an ordered list of **layers**: a photo, a block of colour, a line
of type, a logo. Every measurement is a percentage of the canvas, so the same
template renders at any size, and the preview is the export at a smaller scale
rather than an approximation of it.

Two kinds of indirection make a template worth having rather than a one-off.
Layers name **palette roles** — "accent", "ink" — instead of hex, so one
dropdown restyles the whole design; and they name **font roles** — "display",
"serif" — instead of a font that only exists on one operating system. A layer
carrying a **slot** name is the part you fill in: the headline, the photo, the
date. Everything else is the design.

Text is fitted rather than placed: it wraps to its box and shrinks until it
fits, so a long headline never runs off the edge of the picture.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from upscaler.fonts import DEFAULT_FONT, FONTS, _load_font
from upscaler.screenshot import linear_wash, mesh_wash

SCHEMA = 1


class DesignError(ValueError):
    """A template that can't be rendered as written."""


# ── canvases ──────────────────────────────────────────────────────────────────
CANVASES: "dict[str, tuple[int, int]]" = {
    "YouTube thumbnail · 1280×720": (1280, 720),
    "Social card · 1600×900": (1600, 900),
    "Instagram post · 1080×1080": (1080, 1080),
    "Instagram portrait · 1080×1350": (1080, 1350),
    "Story / Reel · 1080×1920": (1080, 1920),
    "Pinterest pin · 1000×1500": (1000, 1500),
    "LinkedIn banner · 1584×396": (1584, 396),
    "Presentation · 1920×1080": (1920, 1080),
    "Cover square · 1400×1400": (1400, 1400),
    "Poster A4 · 1240×1754": (1240, 1754),
}
CANVAS_NAMES = list(CANVASES)


def canvas_size(name: str) -> "tuple[int, int]":
    """The pixel size of a named canvas, or of a literal "1200x800"."""
    if name in CANVASES:
        return CANVASES[name]
    text = (name or "").lower().replace("×", "x").strip()
    if "x" in text:
        a, _, b = text.partition("x")
        try:
            w, h = int(a.strip()), int(b.strip())
        except ValueError:
            pass
        else:
            if w > 0 and h > 0:
                return w, h
    raise DesignError(f"can't read the canvas {name!r} — use a name from the list "
                      "or a size like 1200x800")


# ── palettes ──────────────────────────────────────────────────────────────────
@dataclass
class Palette:
    """The four colours a template draws with.

    ``primary`` is the ground, ``ink`` is what reads on it, ``accent`` is the
    one loud colour, and ``secondary`` is the second stop of any gradient.
    """
    primary: str = "#14141a"
    secondary: str = "#30304a"
    accent: str = "#ffd166"
    ink: str = "#ffffff"


PALETTES: "dict[str, Palette]" = {
    "Midnight": Palette("#14141a", "#30304a", "#ffd166", "#ffffff"),
    "Sunset": Palette("#3d1338", "#b3325b", "#ffb703", "#fff5e6"),
    "Ocean": Palette("#06263f", "#1b6ca8", "#4dd6c1", "#f2fbff"),
    "Forest": Palette("#132a1e", "#2f6b43", "#d8f36b", "#f4fff6"),
    "Mono": Palette("#101014", "#3a3a42", "#f5f5f5", "#ffffff"),
    "Paper": Palette("#f6f3ec", "#e3ddd0", "#c0392b", "#1a1a1a"),
    "Bubblegum": Palette("#2a1036", "#7b2d8e", "#ff5d8f", "#fff0f6"),
    "Electric": Palette("#0b0b1f", "#2b2bff", "#00f0b5", "#ffffff"),
}
PALETTE_NAMES = list(PALETTES)
ROLES = ("primary", "secondary", "accent", "ink")


def palette(name: str) -> Palette:
    """A copy of a named palette, so editing it can't affect the original."""
    return replace(PALETTES.get(name, PALETTES["Midnight"]))


# ── font roles ────────────────────────────────────────────────────────────────
# A template names a role; the first face that exists on this machine wins, so
# the same design renders on macOS, Windows and a bare Linux container.
FONT_ROLES: "dict[str, list[str]]" = {
    "display": ["Arial Black", "Impact", "DIN Condensed Bold", "Verdana Bold",
                "Arial Bold", "DejaVuSans-Bold", "arialbd", "Tahoma Bold"],
    "sans": ["Helvetica", "Arial", "Avenir Next", "Segoe UI", "DejaVuSans", "arial"],
    "sans bold": ["Arial Bold", "Verdana Bold", "Tahoma Bold", "DejaVuSans-Bold",
                  "arialbd", "Arial Black"],
    "serif": ["Georgia", "Baskerville", "Charter", "PTSerif", "Times New Roman",
              "DejaVuSerif", "times"],
    "serif bold": ["Georgia Bold", "Times New Roman Bold", "SuperClarendon",
                   "Baskerville", "DejaVuSerif-Bold", "timesbd"],
    "condensed": ["DIN Condensed Bold", "Arial Narrow Bold", "Avenir Next Condensed",
                  "Arial Narrow", "Oswald", "DejaVuSansCondensed-Bold"],
    "mono": ["Menlo", "Andale Mono", "PTMono", "Courier New", "Consolas",
             "DejaVuSansMono", "cour"],
}
ROLE_NAMES = list(FONT_ROLES)


def resolve_font(name: str) -> str:
    """A real font name for a role, a font name, or an empty string."""
    for candidate in FONT_ROLES.get(name, [name]):
        if candidate in FONTS:
            return candidate
    return DEFAULT_FONT


# ── layers ────────────────────────────────────────────────────────────────────
PHOTO, SUBJECT, TEXT, BAND, LOGO = "photo", "subject", "text", "band", "logo"
KINDS = [PHOTO, SUBJECT, TEXT, BAND, LOGO]
FITS = ["cover", "contain"]
ALIGNS = ["left", "center", "right"]
VALIGNS = ["top", "middle", "bottom"]


@dataclass
class Layer:
    kind: str = TEXT
    slot: str = ""               # named → the user fills this one in
    # the box, as percentages of the canvas; x and y are its centre
    x: float = 50.0
    y: float = 50.0
    w: float = 84.0
    h: float = 0.0               # 0 = natural height (text block, photo aspect)
    rotation: float = 0.0
    opacity: float = 100.0
    # ── text ──
    text: str = ""               # the default copy, shown until the slot is filled
    font: str = "sans"           # a role from FONT_ROLES, or a font name
    size: float = 9.0            # % of the canvas height — a ceiling; it shrinks to fit
    color: str = "ink"           # a palette role, #rgb, #rrggbb or #rrggbbaa
    align: str = "center"
    valign: str = "middle"
    line_height: float = 1.14
    tracking: float = 0.0        # letter-spacing, % of the text size
    upper: bool = False
    outline: str = ""
    outline_width: float = 0.0   # % of the text size
    shadow: float = 0.0          # 0..100
    # ── photo, subject, logo ──
    fit: str = "cover"
    radius: float = 0.0          # % of the box's short side
    # ── band ──
    color2: str = ""             # second stop; empty = flat
    angle: float = 90.0


@dataclass
class Template:
    name: str = "Untitled"
    canvas: str = CANVAS_NAMES[0]
    background: str = "solid"    # solid | gradient | mesh | photo | transparent
    palette: Palette = field(default_factory=Palette)
    layers: "list[Layer]" = field(default_factory=list)
    note: str = ""               # one line on what the template is for

    def slots(self) -> "list[tuple[str, str]]":
        """The (slot, kind) pairs a person fills in, in layer order."""
        seen, out = set(), []
        for layer in self.layers:
            if layer.slot and layer.slot not in seen:
                seen.add(layer.slot)
                out.append((layer.slot, layer.kind))
        return out

    def text_slots(self) -> "list[str]":
        return [s for s, kind in self.slots() if kind == TEXT]

    def wants_photo(self) -> bool:
        # A template can use the photo as its whole ground rather than as a
        # layer, and it still can't be finished without one.
        return (self.background == "photo"
                or any(layer.kind in (PHOTO, SUBJECT) for layer in self.layers))

    def wants_subject(self) -> bool:
        return any(layer.kind == SUBJECT for layer in self.layers)

    def wants_logo(self) -> bool:
        return any(layer.kind == LOGO for layer in self.layers)


# ── colour ────────────────────────────────────────────────────────────────────
def _rgba(value: str, pal: Palette, default=(255, 255, 255, 255)) -> "tuple[int, ...]":
    """A colour from a palette role or a hex string, always RGBA.

    A trailing ``@N`` sets the opacity as a percentage — ``"primary@85"`` is
    the palette's ground at 85%, which is how a scrim stays themeable instead
    of freezing one hex into the template.
    """
    text = (value or "").strip()
    fade = 1.0
    if "@" in text:
        text, _, share = text.partition("@")
        text = text.strip()
        try:
            fade = float(np.clip(float(share), 0.0, 100.0)) / 100.0
        except ValueError:
            fade = 1.0
    if text in ROLES:
        text = getattr(pal, text)
    if text == "transparent":
        return (0, 0, 0, 0)
    s = text.lstrip("#")
    if len(s) in (3, 4):
        s = "".join(ch * 2 for ch in s)
    try:
        parts = tuple(int(s[i:i + 2], 16) for i in range(0, len(s), 2))
    except (ValueError, IndexError):
        return default
    if len(parts) == 3:
        parts = parts + (255,)
    elif len(parts) != 4:
        return default
    return parts[:3] + (int(round(parts[3] * fade)),)


# ── text ──────────────────────────────────────────────────────────────────────
def _line_width(text: str, font, tracking_px: float) -> float:
    return float(font.getlength(text)) + tracking_px * max(0, len(text) - 1)


def _wrap(text: str, font, max_w: float, tracking_px: float) -> "list[str]":
    """Greedy word wrap. A word longer than the box gets its own line rather
    than being broken — the fitting loop will shrink the type instead."""
    lines: "list[str]" = []
    for para in text.split("\n"):
        words = para.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if _line_width(trial, font, tracking_px) <= max_w:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def fit_text(text: str, font_name: str, box_w: float, box_h: float, max_px: int,
             line_height: float, tracking_pct: float, min_px: int = 7):
    """The largest size at or below ``max_px`` whose wrapped text fits the box.

    Returns ``(font, lines, px, tracking_px)``. Fitting is a binary search
    because "does it fit" only ever goes from true to false as the type grows.
    """
    def attempt(px: int):
        font = _load_font(font_name, px)
        tracking_px = tracking_pct / 100.0 * px
        lines = _wrap(text, font, box_w, tracking_px)
        widest = max((_line_width(line, font, tracking_px) for line in lines),
                     default=0.0)
        tall = len(lines) * px * line_height
        fits = widest <= box_w and (box_h <= 0 or tall <= box_h)
        return fits, (font, lines, px, tracking_px)

    lo, hi = min_px, max(min_px, int(max_px))
    ok, best = attempt(lo)
    fits_hi, at_hi = attempt(hi)
    if fits_hi:
        return at_hi
    while lo < hi - 1:
        mid = (lo + hi) // 2
        fits, got = attempt(mid)
        if fits:
            lo, best = mid, got
        else:
            hi = mid
    return best


def _draw_line(draw, xy, text: str, font, tracking_px: float, fill,
               stroke_width: int = 0, stroke_fill=None) -> None:
    """One line, honouring letter-spacing. Without tracking this is a single
    call; with it, each character is placed by hand."""
    x, y = xy
    if tracking_px <= 0:
        draw.text((x, y), text, font=font, fill=fill, anchor="la",
                  stroke_width=stroke_width, stroke_fill=stroke_fill)
        return
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill, anchor="la",
                  stroke_width=stroke_width, stroke_fill=stroke_fill)
        x += font.getlength(ch) + tracking_px


def _text_layer(layer: Layer, box: "tuple[int, int]", pal: Palette,
                content: str, canvas_h: int) -> "Image.Image | None":
    """A block of type, fitted to its box, on its own transparent layer."""
    text = (content or "").strip()
    if not text:
        return None
    if layer.upper:
        text = text.upper()
    box_w, box_h = box
    # Type is a share of the canvas height, not of the box it sits in, so two
    # headlines set at the same size match even in boxes of different shapes.
    max_px = max(7, int(round(layer.size / 100.0 * canvas_h)))
    font, lines, px, tracking_px = fit_text(
        text, resolve_font(layer.font), box_w, box_h, max_px,
        max(0.7, layer.line_height), layer.tracking)

    line_px = px * max(0.7, layer.line_height)
    total_h = max(1, int(round(len(lines) * line_px)))
    stroke = int(round(max(0.0, layer.outline_width) / 100.0 * px))
    pad = stroke * 2 + max(4, int(px * 0.45))
    width = max(1, int(round(box_w))) + pad * 2
    height = total_h + pad * 2
    out = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    fill = _rgba(layer.color, pal)
    stroke_fill = _rgba(layer.outline, pal, (0, 0, 0, 255)) if stroke else None

    def paint(target, colour, stroke_colour, dx=0.0, dy=0.0):
        draw = ImageDraw.Draw(target)
        for i, line in enumerate(lines):
            if not line:
                continue
            w = _line_width(line, font, tracking_px)
            if layer.align == "right":
                x = pad + box_w - w
            elif layer.align == "left":
                x = pad
            else:
                x = pad + (box_w - w) / 2.0
            _draw_line(draw, (x + dx, pad + i * line_px + dy), line, font,
                       tracking_px, colour, stroke, stroke_colour)

    if layer.shadow > 0:
        alpha = int(255 * min(1.0, layer.shadow / 100.0))
        shade = Image.new("RGBA", out.size, (0, 0, 0, 0))
        paint(shade, (0, 0, 0, alpha), (0, 0, 0, alpha) if stroke else None,
              px * 0.05, px * 0.06)
        out = Image.alpha_composite(
            out, shade.filter(ImageFilter.GaussianBlur(max(1.0, px * 0.045))))
    paint(out, fill, stroke_fill)
    return out


# ── other layers ──────────────────────────────────────────────────────────────
def _rounded(img: Image.Image, radius_pct: float) -> Image.Image:
    out = img.convert("RGBA")
    r = float(np.clip(radius_pct, 0.0, 50.0)) / 100.0 * min(out.size)
    if r < 0.5:
        return out
    mask = Image.new("L", out.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, out.width - 1, out.height - 1],
                                           radius=r, fill=255)
    keep = np.asarray(out.getchannel("A"), np.uint16) * np.asarray(mask, np.uint16)
    out.putalpha(Image.fromarray((keep // 255).astype(np.uint8), "L"))
    return out


def _fit_box(img: Image.Image, box: "tuple[int, int]", how: str) -> Image.Image:
    """The picture at the box's size, either filling it or fitting inside it."""
    bw, bh = max(1, box[0]), max(1, box[1])
    scale = (max(bw / img.width, bh / img.height) if how != "contain"
             else min(bw / img.width, bh / img.height))
    sized = img.resize((max(1, round(img.width * scale)),
                        max(1, round(img.height * scale))), Image.LANCZOS)
    if how == "contain":
        canvas = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        canvas.paste(sized, ((bw - sized.width) // 2, (bh - sized.height) // 2),
                     sized if sized.mode == "RGBA" else None)
        return canvas
    left, top = (sized.width - bw) // 2, (sized.height - bh) // 2
    return sized.crop((left, top, left + bw, top + bh)).convert("RGBA")


def _band_layer(layer: Layer, box: "tuple[int, int]", pal: Palette) -> Image.Image:
    """A block of colour: flat, or a gradient between two stops. Either stop
    may be transparent, which is how a scrim under a headline is made."""
    bw, bh = max(1, box[0]), max(1, box[1])
    c1 = _rgba(layer.color, pal)
    if not layer.color2:
        return Image.new("RGBA", (bw, bh), c1)
    c2 = _rgba(layer.color2, pal)
    ang = np.deg2rad(layer.angle)
    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
    proj = xx * np.cos(ang) + yy * np.sin(ang)
    lo, hi = float(proj.min()), float(proj.max())
    t = (proj - lo) / (hi - lo) if hi > lo else np.zeros_like(proj)
    arr = (np.asarray(c1, np.float32)[None, None, :] * (1 - t)[..., None]
           + np.asarray(c2, np.float32)[None, None, :] * t[..., None])
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA")


def subject_cutout(photo: Image.Image, model: str = "") -> Image.Image:
    """The photo with its background removed, for a ``subject`` layer."""
    from upscaler import background as bg

    kw = {"model": model} if model else {}
    return bg.remove_background(photo.convert("RGB"), feather=1, **kw)


# ── the background ────────────────────────────────────────────────────────────
def _background(size: "tuple[int, int]", tpl: Template, pal: Palette,
                photo: "Image.Image | None") -> Image.Image:
    w, h = size
    kind = tpl.background
    if kind == "transparent":
        return Image.new("RGBA", size, (0, 0, 0, 0))
    if kind == "photo":
        if photo is None:
            return Image.new("RGBA", size, _rgba("primary", pal))
        return _fit_box(photo.convert("RGB"), size, "cover")
    c1 = _rgba("primary", pal)[:3]
    if kind == "solid":
        return Image.new("RGBA", size, tuple(c1) + (255,))
    c2 = _rgba("secondary", pal)[:3]
    arr = (mesh_wash(w, h, c1, c2, 135.0) if kind == "mesh"
           else linear_wash(w, h, c1, c2, 135.0))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB").convert("RGBA")


# ── rendering ─────────────────────────────────────────────────────────────────
def render(tpl: Template, photo: "Image.Image | None" = None,
           texts: "dict[str, str] | None" = None,
           logo: "Image.Image | None" = None, scale: float = 1.0,
           pal: "Palette | None" = None,
           cutout: "Image.Image | None" = None) -> Image.Image:
    """The finished graphic.

    ``scale`` renders the same design smaller — a preview is the export at a
    fraction of its size, not a different arrangement of it.
    """
    pal = pal or tpl.palette
    texts = texts or {}
    cw, ch = canvas_size(tpl.canvas)
    scale = float(np.clip(scale, 0.02, 4.0))
    cw, ch = max(1, round(cw * scale)), max(1, round(ch * scale))

    out = _background((cw, ch), tpl, pal, photo)
    for layer in tpl.layers:
        art = _layer_image(layer, (cw, ch), tpl, pal, photo, texts, logo, cutout)
        if art is None:
            continue
        if layer.rotation % 360:
            art = art.rotate(-layer.rotation, resample=Image.BICUBIC, expand=True)
        opacity = float(np.clip(layer.opacity, 0.0, 100.0)) / 100.0
        if opacity < 1.0:
            alpha = np.asarray(art.getchannel("A"), np.float32) * opacity
            art.putalpha(Image.fromarray(alpha.round().astype(np.uint8), "L"))
        x = int(round(layer.x / 100.0 * cw - art.width / 2))
        y = int(round(layer.y / 100.0 * ch - art.height / 2))
        out = _place(out, art, (x, y))
    return out if tpl.background == "transparent" else out.convert("RGB")


def _place(canvas: Image.Image, art: Image.Image,
           at: "tuple[int, int]") -> Image.Image:
    """Composite ``art`` onto ``canvas``, letting it hang off the edge.

    ``alpha_composite`` refuses a negative offset, so anything that overhangs
    goes through a full-canvas scratch — ``paste`` crops rather than shifting,
    which is what a design that bleeds off the page needs.
    """
    x, y = at
    if x >= 0 and y >= 0 and x + art.width <= canvas.width \
            and y + art.height <= canvas.height:
        canvas.alpha_composite(art, (x, y))
        return canvas
    scratch = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    scratch.paste(art, (x, y))
    return Image.alpha_composite(canvas, scratch)


def _layer_image(layer: Layer, canvas: "tuple[int, int]", tpl: Template,
                 pal: Palette, photo, texts, logo, cutout):
    cw, ch = canvas
    box_w = max(1, int(round(layer.w / 100.0 * cw)))
    box_h = int(round(layer.h / 100.0 * ch))

    if layer.kind == TEXT:
        content = texts.get(layer.slot) if layer.slot else None
        if content is None:
            content = layer.text
        art = _text_layer(layer, (box_w, box_h), pal, content, ch)
        if art is None or box_h <= 0 or layer.valign == "middle":
            return art
        # Grow the block back to the box so a bottom-aligned headline keeps its
        # baseline as it wraps to a second line.
        holder = Image.new("RGBA", (art.width, max(box_h, art.height)), (0, 0, 0, 0))
        top = 0 if layer.valign == "top" else holder.height - art.height
        holder.alpha_composite(art, (0, top))
        return holder

    if layer.kind == BAND:
        art = _band_layer(layer, (box_w, box_h or box_w), pal)
        return _rounded(art, layer.radius)

    source = logo if layer.kind == LOGO else photo
    if layer.kind == SUBJECT:
        if cutout is not None:
            source = cutout
        elif photo is not None:
            source = subject_cutout(photo)
    if source is None:
        return None
    source = source.convert("RGBA") if "A" in source.getbands() else source.convert("RGB")
    natural = max(1, round(box_w * source.height / source.width))
    art = _fit_box(source, (box_w, box_h or natural),
                   layer.fit if layer.kind == PHOTO else "contain")
    return _rounded(art, layer.radius)


def preview(tpl: Template, photo=None, texts=None, logo=None, max_edge: int = 900,
            pal=None, cutout=None) -> Image.Image:
    """The design at preview size — the export, rendered smaller."""
    cw, ch = canvas_size(tpl.canvas)
    return render(tpl, photo, texts, logo, scale=min(1.0, max_edge / max(cw, ch)),
                  pal=pal, cutout=cutout)


def missing(tpl: Template, photo=None, texts=None, logo=None) -> "list[str]":
    """What the design is still waiting for, so the GUI can say so."""
    texts = texts or {}
    gaps = []
    if tpl.wants_photo() and photo is None:
        gaps.append("a photo")
    if tpl.wants_logo() and logo is None:
        gaps.append("a logo")
    for slot, kind in tpl.slots():
        if kind == TEXT and not (texts.get(slot) or "").strip():
            gaps.append(slot)
    return gaps


def describe(tpl: Template) -> str:
    cw, ch = canvas_size(tpl.canvas)
    bits = [f"{cw}×{ch}", tpl.background]
    kinds = [layer.kind for layer in tpl.layers]
    for kind in KINDS:
        n = kinds.count(kind)
        if n:
            bits.append(f"{n} {kind}" + ("s" if n > 1 else ""))
    return f"{tpl.name}: " + " · ".join(bits)


# ── saving and loading ────────────────────────────────────────────────────────
def to_dict(tpl: Template) -> dict:
    data = asdict(tpl)
    data["schema"] = SCHEMA
    return data


def to_json(tpl: Template) -> str:
    return json.dumps(to_dict(tpl), indent=2)


def _build(cls, values: dict):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in values.items() if k in known})


def from_dict(data: dict) -> Template:
    """A template from parsed JSON, tolerating settings it doesn't know."""
    if not isinstance(data, dict):
        raise DesignError("a template must be a JSON object")
    layers = []
    for raw in data.get("layers") or []:
        if not isinstance(raw, dict):
            raise DesignError(f"each layer must be an object, got {type(raw).__name__}")
        kind = str(raw.get("kind", TEXT))
        if kind not in KINDS:
            raise DesignError(f"layer names an unknown kind {kind!r} — "
                              f"expected one of {', '.join(KINDS)}")
        layers.append(_build(Layer, raw))
    pal = data.get("palette")
    return Template(
        name=str(data.get("name") or "Untitled"),
        canvas=str(data.get("canvas") or CANVAS_NAMES[0]),
        background=str(data.get("background") or "solid"),
        palette=_build(Palette, pal) if isinstance(pal, dict) else Palette(),
        layers=layers,
        note=str(data.get("note") or ""),
    )


def from_json(text: str) -> Template:
    try:
        return from_dict(json.loads(text))
    except json.JSONDecodeError as e:
        raise DesignError(f"that isn't valid JSON: {e}") from e


def load(path: "str | Path") -> Template:
    return from_json(Path(path).read_text())


def save(tpl: Template, path: "str | Path") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_json(tpl))
    return p


# ── the templates that ship ───────────────────────────────────────────────────
def _L(**kw) -> Layer:
    return Layer(**kw)


BUILT_IN: "dict[str, Template]" = {
    "YouTube thumbnail": Template(
        name="YouTube thumbnail", canvas="YouTube thumbnail · 1280×720",
        background="photo", palette=palette("Electric"),
        note="Your frame, a dark scrim and a headline big enough to read at "
             "thumbnail size.",
        layers=[
            _L(kind=BAND, x=50, y=76, w=100, h=56, color="transparent",
               color2="primary@92", angle=90),
            _L(kind=BAND, x=12, y=11, w=19, h=8.5, color="accent", radius=26),
            _L(kind=TEXT, slot="label", text="NEW", x=12, y=11, w=17, h=8.5,
               size=4.0, font="sans bold", color="primary", upper=True, tracking=10),
            _L(kind=TEXT, slot="headline", text="Say it in four words",
               x=50, y=77, w=90, h=30, size=15, font="display", color="ink",
               upper=True, valign="bottom", shadow=45, line_height=1.02),
        ],
    ),
    "Quote card": Template(
        name="Quote card", canvas="Instagram post · 1080×1080",
        background="gradient", palette=palette("Midnight"),
        note="A line worth repeating, set large, with the credit under a rule.",
        layers=[
            _L(kind=TEXT, slot="quote", text="The best time to plant a tree was "
               "twenty years ago. The second best time is now.",
               x=50, y=45, w=76, h=48, size=8.5, font="serif", color="ink",
               line_height=1.32),
            _L(kind=BAND, x=50, y=74, w=11, h=0.65, color="accent"),
            _L(kind=TEXT, slot="author", text="Chinese proverb", x=50, y=81, w=64,
               size=3.2, font="sans", color="accent", upper=True, tracking=18),
        ],
    ),
    "Photo post": Template(
        name="Photo post", canvas="Instagram post · 1080×1080",
        background="solid", palette=palette("Paper"),
        note="Photo on top, words underneath — the caption layout that reads "
             "cleanly in a feed.",
        layers=[
            _L(kind=PHOTO, slot="photo", x=50, y=33, w=100, h=66, fit="cover"),
            _L(kind=TEXT, slot="headline", text="A good day out", x=50, y=78,
               w=84, h=15, size=7.5, font="serif bold", color="ink"),
            _L(kind=TEXT, slot="subhead", text="Somewhere worth going", x=50, y=90,
               w=80, size=3.2, font="sans", color="accent", upper=True, tracking=16),
        ],
    ),
    "Story headline": Template(
        name="Story headline", canvas="Story / Reel · 1080×1920",
        background="photo", palette=palette("Sunset"),
        note="Full-bleed photo with the words in the lower third, where a "
             "thumb won't cover them.",
        layers=[
            _L(kind=BAND, x=50, y=72, w=100, h=60, color="transparent",
               color2="primary@94", angle=90),
            _L(kind=TEXT, slot="kicker", text="THIS WEEK", x=50, y=60, w=70,
               size=2.9, font="sans bold", color="accent", upper=True, tracking=22),
            _L(kind=TEXT, slot="headline", text="Somewhere new", x=50, y=73, w=84,
               h=20, size=9.5, font="display", color="ink", line_height=1.06),
            _L(kind=TEXT, slot="footer", text="swipe up", x=50, y=92, w=70,
               size=2.4, font="sans", color="ink@70", upper=True, tracking=14),
        ],
    ),
    "Big announcement": Template(
        name="Big announcement", canvas="Instagram post · 1080×1080",
        background="mesh", palette=palette("Bubblegum"),
        note="One number, one line, one date — a sale or a launch.",
        layers=[
            _L(kind=BAND, x=50, y=44, w=132, h=24, color="accent", rotation=-7),
            _L(kind=TEXT, slot="headline", text="50% OFF", x=50, y=44, w=84, h=17,
               size=15, font="display", color="primary", upper=True),
            _L(kind=TEXT, slot="subhead", text="Everything in the shop", x=50, y=66,
               w=72, size=4.2, font="sans bold", color="ink", upper=True, tracking=12),
            _L(kind=TEXT, slot="footer", text="until Sunday", x=50, y=84, w=70,
               size=3.0, font="sans", color="ink@80"),
        ],
    ),
    "Event poster": Template(
        name="Event poster", canvas="Poster A4 · 1240×1754",
        background="solid", palette=palette("Forest"),
        note="Photo, title, when and where — printable at A4.",
        layers=[
            _L(kind=PHOTO, slot="photo", x=50, y=27, w=100, h=54, fit="cover"),
            _L(kind=BAND, x=50, y=60, w=14, h=0.7, color="accent"),
            _L(kind=TEXT, slot="title", text="Summer Session", x=50, y=69, w=84,
               h=17, size=8.5, font="display", color="ink", upper=True,
               line_height=1.04),
            _L(kind=TEXT, slot="date", text="Saturday 12 July · 7pm", x=50, y=83,
               w=76, size=3.6, font="sans bold", color="accent", upper=True,
               tracking=10),
            _L(kind=TEXT, slot="place", text="The Old Warehouse, Bristol", x=50,
               y=89, w=76, size=2.8, font="sans", color="ink@80"),
        ],
    ),
    "Presentation title": Template(
        name="Presentation title", canvas="Presentation · 1920×1080",
        background="gradient", palette=palette("Ocean"),
        note="An opening slide: title, subtitle, and your name along the bottom.",
        layers=[
            _L(kind=BAND, x=7, y=48, w=0.9, h=42, color="accent"),
            _L(kind=TEXT, slot="title", text="What we learned this quarter",
               x=53, y=42, w=76, h=26, size=10, font="sans bold", color="ink",
               align="left", valign="bottom", line_height=1.08),
            _L(kind=TEXT, slot="subtitle", text="And what we're doing about it",
               x=53, y=63, w=76, size=3.8, font="sans", color="ink@75", align="left"),
            _L(kind=TEXT, slot="footer", text="your name · the date", x=53, y=88,
               w=76, size=2.4, font="sans", color="accent", upper=True,
               tracking=18, align="left"),
        ],
    ),
    "Profile banner": Template(
        name="Profile banner", canvas="LinkedIn banner · 1584×396",
        background="mesh", palette=palette("Mono"),
        note="The wide strip at the top of a profile: who you are and what "
             "you do.",
        layers=[
            _L(kind=TEXT, slot="name", text="Your Name", x=36, y=38, w=56, h=26,
               size=21, font="sans bold", color="ink", align="left"),
            _L(kind=TEXT, slot="role", text="What you do, in six words",
               x=36, y=70, w=56, size=7.5, font="sans", color="accent",
               align="left", upper=True, tracking=8),
        ],
    ),
    "Cover art": Template(
        name="Cover art", canvas="Cover square · 1400×1400",
        background="photo", palette=palette("Midnight"),
        note="A square cover for a podcast, a playlist or an album.",
        layers=[
            _L(kind=BAND, x=50, y=50, w=100, h=100, color="primary@15",
               color2="primary@93", angle=90),
            _L(kind=TEXT, slot="title", text="The Long Way Round", x=50, y=63,
               w=82, h=26, size=12, font="display", color="ink", upper=True,
               valign="bottom", line_height=1.0),
            _L(kind=TEXT, slot="subtitle", text="Episode 12", x=50, y=81, w=70,
               size=3.4, font="sans", color="accent", upper=True, tracking=20),
        ],
    ),
    "Subject spotlight": Template(
        name="Subject spotlight", canvas="Instagram portrait · 1080×1350",
        background="gradient", palette=palette("Sunset"),
        note="The word goes behind the person: the subject is cut out of the "
             "photo and put back on top of the type.",
        layers=[
            _L(kind=TEXT, slot="headline", text="SUMMER", x=50, y=40, w=96,
               size=24, font="display", color="accent", upper=True,
               line_height=0.95),
            _L(kind=SUBJECT, slot="photo", x=50, y=57, w=94, h=74, fit="contain"),
            _L(kind=TEXT, slot="caption", text="your name", x=50, y=92, w=70,
               size=4.0, font="sans bold", color="ink", upper=True, tracking=16),
        ],
    ),
}
BUILT_IN_NAMES = list(BUILT_IN)


def built_in(name: str) -> Template:
    """A copy of a template that ships, so editing it can't affect the original."""
    if name not in BUILT_IN:
        raise DesignError(f"unknown template {name!r} — "
                          f"expected one of {', '.join(BUILT_IN_NAMES)}")
    return from_dict(to_dict(BUILT_IN[name]))
