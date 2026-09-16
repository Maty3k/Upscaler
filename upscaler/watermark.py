"""Watermarks — a signature, a caption or a logo, placed once or tiled.

Pure PIL + numpy (no AI, and no torch: the font machinery lives in
:mod:`upscaler.fonts` precisely so this module stays light enough to run over a
folder of a thousand photos).

The mark is drawn once onto its own transparent layer and then composited, so
opacity, rotation and tiling all behave the same whether the mark is text or a
logo. Every size is a percentage of the photo's short side, so one setting
looks the same on a phone snap and a 4K frame — which is what makes batching a
mixed folder work at all.

Two habits worth knowing:

* a plain white mark disappears on a white sky, so text carries an optional
  outline and shadow, and both are on by default;
* "tiled" repeats the mark across the whole frame at an angle, which is the
  proof-copy watermark that survives being cropped out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw

from upscaler.fonts import DEFAULT_FONT, FONT_NAMES, FONTS, _load_font  # noqa: F401

KINDS = ["text", "logo"]
POSITIONS = [
    "top left", "top center", "top right",
    "middle left", "center", "middle right",
    "bottom left", "bottom center", "bottom right",
    "tiled",
]
DEFAULT_POSITION = "bottom right"
TILED = "tiled"


@dataclass
class WatermarkParams:
    kind: str = "text"
    # ── text ──
    text: str = "© Your Name"
    font: str = DEFAULT_FONT
    size: float = 4.0            # % of the short side
    color: str = "#ffffff"
    outline: str = "#000000"
    outline_width: float = 8.0   # % of the text size
    shadow: float = 45.0         # 0..100
    # ── logo ──
    logo_scale: float = 18.0     # % of the photo's width
    # ── placement ──
    position: str = DEFAULT_POSITION
    margin: float = 3.0          # % of the short side
    rotation: float = 0.0        # degrees
    opacity: float = 70.0        # 0..100
    tile_gap: float = 14.0       # % of the short side, between repeats
    tile_angle: float = 30.0     # degrees, for the tiled pattern

    def is_identity(self) -> bool:
        """True when nothing would be drawn."""
        if self.opacity <= 0:
            return True
        return self.kind == "text" and not (self.text or "").strip()


PRESET_NONE = "None"
PRESETS: dict[str, WatermarkParams] = {
    "Corner signature": WatermarkParams(size=3.5, opacity=75, position="bottom right"),
    "Bold corner": WatermarkParams(size=6, opacity=90, position="bottom left",
                                   outline_width=12, shadow=60),
    "Subtle caption": WatermarkParams(size=2.5, opacity=45, position="bottom center",
                                      shadow=25, outline_width=5),
    "Centre stamp": WatermarkParams(size=12, opacity=30, position="center",
                                    rotation=-20, outline_width=4, shadow=0),
    "Proof (tiled)": WatermarkParams(text="PROOF", size=7, opacity=22, position=TILED,
                                     tile_angle=30, tile_gap=12, outline_width=0,
                                     shadow=0),
    "Do not copy (tiled)": WatermarkParams(text="DO NOT COPY", size=5, opacity=28,
                                           position=TILED, tile_angle=35, tile_gap=10,
                                           outline_width=0, shadow=0),
    "Logo corner": WatermarkParams(kind="logo", logo_scale=16, opacity=80,
                                   position="bottom right"),
    "Logo tiled": WatermarkParams(kind="logo", logo_scale=12, opacity=18,
                                  position=TILED, tile_angle=25, tile_gap=16),
}
PRESET_NAMES = [PRESET_NONE] + list(PRESETS)


def preset(name: str) -> WatermarkParams:
    """The params for a preset; unknown names and "None" mean no watermark."""
    if name in PRESETS:
        return replace(PRESETS[name])
    return WatermarkParams(opacity=0.0)


# ── helpers ───────────────────────────────────────────────────────────────────
def _hex(c: str, default=(255, 255, 255)) -> "tuple[int, int, int]":
    s = (c or "").lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except (ValueError, IndexError):
        return default


def _text_layer(p: WatermarkParams, short: int) -> Image.Image:
    """The text on its own transparent layer, with outline and shadow, sized
    to the photo rather than to a fixed point size."""
    px = max(8, int(round(max(0.2, p.size) / 100.0 * short)))
    font = _load_font(p.font, px)
    stroke = int(round(max(0.0, p.outline_width) / 100.0 * px))
    text = p.text or ""

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    box = probe.multiline_textbbox((0, 0), text, font=font, stroke_width=stroke)
    pad = max(stroke * 2, int(px * 0.5)) + 4          # room for the shadow's blur
    w = max(1, box[2] - box[0]) + pad * 2
    h = max(1, box[3] - box[1]) + pad * 2
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    origin = (pad - box[0], pad - box[1])

    if p.shadow > 0:
        from PIL import ImageFilter

        shade = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(shade).multiline_text(
            (origin[0] + px * 0.06, origin[1] + px * 0.06), text, font=font,
            fill=(0, 0, 0, int(255 * min(1.0, p.shadow / 100.0))),
            stroke_width=stroke, stroke_fill=(0, 0, 0, int(255 * min(1.0, p.shadow / 100.0))))
        layer = Image.alpha_composite(
            layer, shade.filter(ImageFilter.GaussianBlur(max(1.0, px * 0.05))))
        draw = ImageDraw.Draw(layer)

    draw.multiline_text(origin, text, font=font, fill=_hex(p.color) + (255,),
                        stroke_width=stroke,
                        stroke_fill=_hex(p.outline, (0, 0, 0)) + (255,) if stroke else None)
    return layer


def _logo_layer(logo: Image.Image, p: WatermarkParams, size: "tuple[int, int]") -> Image.Image:
    """The logo scaled to a share of the photo's width, alpha intact."""
    target_w = max(1, int(round(max(0.5, p.logo_scale) / 100.0 * size[0])))
    scale = target_w / max(1, logo.width)
    return logo.convert("RGBA").resize(
        (target_w, max(1, int(round(logo.height * scale)))), Image.LANCZOS)


def build_mark(p: WatermarkParams, size: "tuple[int, int]",
               logo: "Image.Image | None" = None) -> "Image.Image | None":
    """The mark itself on a transparent layer, rotated and faded, or None when
    there is nothing to draw."""
    short = min(size)
    if p.kind == "logo":
        if logo is None:
            return None
        layer = _logo_layer(logo, p, size)
    else:
        if not (p.text or "").strip():
            return None
        layer = _text_layer(p, short)

    if p.rotation % 360:
        layer = layer.rotate(p.rotation, resample=Image.BICUBIC, expand=True)
    opacity = float(np.clip(p.opacity, 0.0, 100.0)) / 100.0
    if opacity < 1.0:
        alpha = np.asarray(layer.getchannel("A"), dtype=np.float32) * opacity
        layer.putalpha(Image.fromarray(alpha.round().astype(np.uint8), "L"))
    return layer


def _anchor_xy(position: str, mark: "tuple[int, int]", size: "tuple[int, int]",
               margin_px: int) -> "tuple[int, int]":
    """Top-left corner for one of the nine positions."""
    w, h = size
    mw, mh = mark
    row, _, col = position.partition(" ")
    if position == "center":
        row = col = "center"
    x = {"left": margin_px, "center": (w - mw) // 2, "right": w - mw - margin_px}.get(
        col, (w - mw) // 2)
    y = {"top": margin_px, "middle": (h - mh) // 2, "center": (h - mh) // 2,
         "bottom": h - mh - margin_px}.get(row, h - mh - margin_px)
    return x, y


def _tiled_layer(mark: Image.Image, p: WatermarkParams,
                 size: "tuple[int, int]") -> Image.Image:
    """The mark repeated across the whole frame at an angle.

    The grid is laid out on a canvas big enough to cover the photo's diagonal,
    then turned and centre-cropped, so the pattern reaches the corners however
    it is rotated.
    """
    w, h = size
    gap = max(2, int(round(max(0.0, p.tile_gap) / 100.0 * min(size))))
    step_x, step_y = mark.width + gap, mark.height + gap
    diag = int(math.hypot(w, h)) + max(step_x, step_y) * 2
    canvas = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
    for row, y in enumerate(range(0, diag, step_y)):
        # Offset every other row so the repeats don't line up in columns.
        x0 = -step_x + (step_x // 2 if row % 2 else 0)
        for x in range(x0, diag, step_x):
            canvas.alpha_composite(mark, (x, y))
    if p.tile_angle % 360:
        canvas = canvas.rotate(p.tile_angle, resample=Image.BICUBIC)
    left, top = (canvas.width - w) // 2, (canvas.height - h) // 2
    return canvas.crop((left, top, left + w, top + h))


def apply(img: Image.Image, p: WatermarkParams,
          logo: "Image.Image | None" = None) -> Image.Image:
    """Stamp the watermark onto the photo. Alpha is carried through."""
    if p.position not in POSITIONS:
        raise ValueError(f"unknown position {p.position!r} — expected one of {POSITIONS}")
    if p.kind not in KINDS:
        raise ValueError(f"unknown watermark kind {p.kind!r} — expected one of {KINDS}")
    if p.kind == "logo" and logo is None:
        raise ValueError("pick a logo image, or switch the watermark to text")

    mark = build_mark(p, img.size, logo)
    if mark is None:
        return img

    had_alpha = "A" in img.getbands()
    base = img.convert("RGBA")
    if p.position == TILED:
        layer = _tiled_layer(mark, p, base.size)
    else:
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        margin_px = int(round(max(0.0, p.margin) / 100.0 * min(base.size)))
        layer.alpha_composite(mark, _anchor_xy(p.position, mark.size, base.size, margin_px))
    out = Image.alpha_composite(base, layer)
    return out if had_alpha else out.convert("RGB")


def preview(img: Image.Image, p: WatermarkParams, logo: "Image.Image | None" = None,
            max_edge: int = 1000) -> Image.Image:
    """The result at preview size. Every measure is relative, so it matches
    the full-size export."""
    scale = min(1.0, max_edge / max(img.size))
    small = img if scale >= 1.0 else img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    try:
        return apply(small, p, logo)
    except ValueError:
        return small


def describe(p: WatermarkParams) -> str:
    """A short human summary of the watermark."""
    if p.is_identity():
        return "no watermark"
    what = f'text "{p.text}"' if p.kind == "text" else f"logo at {p.logo_scale:g}% width"
    bits = [what]
    if p.kind == "text":
        bits.append(f"{p.size:g}% size")
    bits.append("tiled" if p.position == TILED else p.position)
    bits.append(f"{p.opacity:g}% opacity")
    if p.rotation % 360:
        bits.append(f"rotated {p.rotation:g}°")
    return " · ".join(bits)
