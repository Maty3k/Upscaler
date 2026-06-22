"""Lian Li 8.8" Universal Screen — panel media builder.

Produces media sized *exactly* for the Lian Li 8.8" screen so L-Connect 3 never
resamples it: 1920×480 (landscape) or 480×1920 (portrait), 4:1.

Everything is composited server-side with PIL onto a fixed-resolution canvas
(the single source of truth), and animated exports are encoded with native
ffmpeg (H.264/yuv420p for MP4, palette-optimised GIF). The same `compose_frame`
is used for the live preview and for every exported frame, so what you preview
is what you get.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from upscaler.video import _ffmpeg

# ── Canvas spec (non-negotiable) ─────────────────────────────────────────────
ORIENTATIONS: dict[str, tuple[int, int]] = {
    "Landscape · 1920×480": (1920, 480),
    "Portrait · 480×1920": (480, 1920),
}
FITS = ["cover", "contain", "stretch", "manual"]
MAX_DURATION_SEC = 180  # 3 minutes
MAX_FPS = 60

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
STILL_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

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


# An overlay is a plain dict so it round-trips through Gradio state easily:
#   text:    {type:'text', content, font, size, color, align, x, y, rotation,
#             stroke, stroke_w, motion, speed, cps}
#   sticker: {type:'sticker', image(PIL), scale, x, y, rotation, opacity}
#   clock:   {type:'clock', content(<strftime template>), font, size, color,
#             align, x, y, rotation, stroke, stroke_w, motion, speed, cps}
# motion (text/clock) is one of: none | scroll-left | scroll-right | scroll-up |
#   scroll-down | fade | typewriter. It only animates across an exported clip;
#   a still preview shows the starting frame. All motion fields are optional and
#   default to static, so older/loaded layouts render unchanged.
Overlay = dict


@dataclass
class PanelParams:
    orientation: str = "Landscape · 1920×480"
    fit: str = "cover"
    zoom: float = 1.0
    off_x: float = 0.0  # pan, percent of canvas width  (-100..100)
    off_y: float = 0.0  # pan, percent of canvas height (-100..100)
    bg_type: str = "solid"  # solid | gradient
    bg_color: str = "#000000"
    bg_color2: str = "#333333"
    bg_angle: float = 90.0
    overlays: list = field(default_factory=list)


def canvas_size(orientation: str) -> tuple[int, int]:
    return ORIENTATIONS.get(orientation, (1920, 480))


# ── Media classification ─────────────────────────────────────────────────────
def media_kind(path: str | None) -> str | None:
    """Return 'image', 'animated', or None for an uploaded file."""
    if not path:
        return None
    ext = os.path.splitext(str(path))[1].lower()
    if ext in VIDEO_EXTS:
        return "animated"
    if ext == ".gif":
        try:
            with Image.open(path) as im:
                return "animated" if getattr(im, "is_animated", False) else "image"
        except Exception:
            return "image"
    if ext in STILL_EXTS:
        return "image"
    # Unknown extension: probe with PIL.
    try:
        with Image.open(path) as im:
            return "animated" if getattr(im, "is_animated", False) else "image"
    except Exception:
        return "animated"  # assume a video container


def media_duration(path: str | None) -> float:
    """Clip length in seconds (0 if unknown / still image)."""
    if not path:
        return 0.0
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".gif":
        try:
            with Image.open(path) as im:
                total = 0
                for i in range(getattr(im, "n_frames", 1)):
                    im.seek(i)
                    total += im.info.get("duration", 100)
                return round(total / 1000.0, 2)
        except Exception:
            return 0.0
    fp = shutil.which("ffprobe")
    if not fp:
        return 0.0
    out = subprocess.run(
        [fp, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return round(float(out), 2)
    except ValueError:
        return 0.0


# ── Colour + font helpers ─────────────────────────────────────────────────────
def _hex(c: str) -> tuple[int, int, int]:
    c = (c or "#000000").lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return (0, 0, 0)


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
    return ImageFont.load_default()


def _background(cw: int, ch: int, p: PanelParams) -> Image.Image:
    if p.bg_type != "gradient":
        return Image.new("RGB", (cw, ch), _hex(p.bg_color))
    c1 = np.array(_hex(p.bg_color), dtype=np.float32)
    c2 = np.array(_hex(p.bg_color2), dtype=np.float32)
    ang = np.deg2rad(p.bg_angle)
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    proj = xx * np.cos(ang) + yy * np.sin(ang)
    lo, hi = proj.min(), proj.max()
    t = (proj - lo) / (hi - lo) if hi > lo else np.zeros_like(proj)
    grad = c1[None, None, :] * (1 - t)[..., None] + c2[None, None, :] * t[..., None]
    return Image.fromarray(grad.clip(0, 255).astype(np.uint8), "RGB")


# ── Geometry ──────────────────────────────────────────────────────────────────
def _drawn_size(sw: int, sh: int, cw: int, ch: int, fit: str, zoom: float) -> tuple[int, int]:
    if fit == "stretch":
        return cw, ch
    if fit == "contain":
        s = min(cw / sw, ch / sh)
    else:  # cover, manual
        s = max(cw / sw, ch / sh)
        if fit == "manual":
            s *= max(0.05, zoom)
    return max(1, round(sw * s)), max(1, round(sh * s))


def _paste_pos(dw: int, dh: int, cw: int, ch: int, p: PanelParams) -> tuple[int, int]:
    # Panning (which 4:1 band survives) applies to cover + manual.
    ox = (p.off_x / 100.0) * cw if p.fit in ("cover", "manual") else 0.0
    oy = (p.off_y / 100.0) * ch if p.fit in ("cover", "manual") else 0.0
    return round((cw - dw) / 2 + ox), round((ch - dh) / 2 + oy)


# ── Compositing ───────────────────────────────────────────────────────────────
def compose_frame(
    src: Image.Image,
    p: PanelParams,
    fast: bool = False,
    *,
    frame_index: int = 0,
    elapsed: float = 0.0,
    now: datetime | None = None,
    total_frames: int = 1,
    fps: int = 30,
) -> Image.Image:
    """Composite a single source frame onto the exact panel canvas.

    `fast=True` uses bilinear resampling for the live preview (cheaper, still
    crisp on screen); exports leave it False for export-grade LANCZOS.

    The keyword-only frame context drives animation: ``frame_index`` /
    ``total_frames`` / ``fps`` position scrolling/fading/typewriter text, and
    ``elapsed`` / ``now`` set the time a clock overlay shows. All default so the
    preview and still exports render a static start frame unchanged.
    """
    cw, ch = canvas_size(p.orientation)
    canvas = _background(cw, ch, p).convert("RGB")
    resample = Image.BILINEAR if fast else Image.LANCZOS

    if src is not None:
        src = src.convert("RGBA")
        dw, dh = _drawn_size(src.width, src.height, cw, ch, p.fit, p.zoom)
        resized = src.resize((dw, dh), resample)
        px, py = _paste_pos(dw, dh, cw, ch, p)
        canvas.paste(resized, (px, py), resized)

    ctx = {"frame_index": frame_index, "total_frames": max(1, total_frames),
           "fps": max(1, fps)}
    for ov in p.overlays:
        kind = ov.get("type")
        if kind == "text":
            _render_text_layer(canvas, ov.get("content", ""), ov, ctx)
        elif kind == "sticker":
            _draw_sticker(canvas, ov, resample)
        elif kind == "clock":
            _draw_clock_overlay(canvas, ov, elapsed=elapsed, now=now, ctx=ctx)
    return canvas


def _text_layer(content: str, ov: Overlay) -> Image.Image:
    """Render `content` in ov's style onto its own RGBA layer (no positioning).
    Kept separate so text and clock overlays share one glyph/stroke/rotation
    path."""
    font = _load_font(ov.get("font", DEFAULT_FONT), int(ov.get("size", 180)))
    align = ov.get("align", "center")
    sw = int(ov.get("stroke_w", 0) or 0)

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = probe.multiline_textbbox((0, 0), content, font=font, align=align,
                                    stroke_width=sw)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = sw + 6
    layer = Image.new("RGBA", (max(1, tw + 2 * pad), max(1, th + 2 * pad)), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    kwargs = dict(font=font, fill=_hex(ov.get("color", "#ffffff")), align=align)
    if sw > 0:
        kwargs["stroke_width"] = sw
        kwargs["stroke_fill"] = _hex(ov.get("stroke", "#000000"))
    ld.multiline_text((pad - bbox[0], pad - bbox[1]), content, **kwargs)

    rot = float(ov.get("rotation", 0) or 0)
    if rot:
        layer = layer.rotate(-rot, expand=True, resample=Image.BICUBIC)
    return layer


def _render_text_layer(canvas: Image.Image, content: str, ov: Overlay, ctx=None) -> None:
    """Draw `content` (text or a clock string) onto `canvas`, honouring ov's
    optional motion. `ctx` carries frame_index/total_frames/fps; None (or no
    motion) renders the static start frame, byte-identical to the old path."""
    content = (content or "").strip("\n")
    if not content.strip():
        return
    cw, ch = canvas.size
    motion = (ov.get("motion") or "none") if ctx else "none"
    fi = ctx["frame_index"] if ctx else 0
    fps = max(1, ctx["fps"]) if ctx else 30
    total = max(1, ctx["total_frames"]) if ctx else 1
    t = fi / fps  # seconds into the clip

    if motion == "typewriter":  # reveal characters over time, then hold full
        cps = float(ov.get("cps", 10) or 10)
        content = content[: max(0, int(t * cps))]
        if not content:
            return

    layer = _text_layer(content, ov)

    if motion == "fade":  # 0→1→0 over the clip so a looped clip has no seam pop
        mul = max(0.0, math.sin(math.pi * (fi / total)))
        layer.putalpha(layer.split()[3].point(lambda v: int(v * mul)))

    x = cw / 2 + (float(ov.get("x", 0)) / 100.0) * cw
    y = ch / 2 + (float(ov.get("y", 0)) / 100.0) * ch

    if motion in ("scroll-left", "scroll-right", "scroll-up", "scroll-down"):
        _paste_scrolling(canvas, layer, motion, x, y,
                         speed=float(ov.get("speed", 120) or 120), t=t, total=total, fps=fps)
    else:
        canvas.paste(layer, (int(x - layer.width / 2), int(y - layer.height / 2)), layer)


def _paste_scrolling(canvas, layer, motion, cx, cy, *, speed, t, total, fps) -> None:
    """Tile `layer` along the scroll axis and paste it at a time-wrapped offset,
    so as one copy exits an edge the next enters. The offset completes a whole
    number of cycles over the clip, so the loop is seamless (offset(0)==offset(end))."""
    cw, ch = canvas.size
    lw, lh = layer.size
    horizontal = motion in ("scroll-left", "scroll-right")
    span = lw if horizontal else lh
    canvas_span = cw if horizontal else ch
    period = span + max(int(canvas_span * 0.25), 40)  # text length + a gap
    # snap to whole cycles per clip → seamless loop; ~honours the requested speed
    cycles = max(1, round(speed * (total / fps) / period))
    offset = ((fi_frac := (t * fps) / total) * cycles * period) % period
    sign = -1 if motion in ("scroll-left", "scroll-up") else 1

    if horizontal:
        anchor = cx - lw / 2 + sign * offset
        k0 = int(math.floor((0 - anchor) / period)) - 1
        k1 = int(math.ceil((cw - anchor) / period)) + 1
        for k in range(k0, k1 + 1):
            canvas.paste(layer, (int(anchor + k * period), int(cy - lh / 2)), layer)
    else:
        anchor = cy - lh / 2 + sign * offset
        k0 = int(math.floor((0 - anchor) / period)) - 1
        k1 = int(math.ceil((ch - anchor) / period)) + 1
        for k in range(k0, k1 + 1):
            canvas.paste(layer, (int(cx - lw / 2), int(anchor + k * period)), layer)


def _draw_clock_overlay(canvas: Image.Image, ov: Overlay, *, elapsed=0.0,
                        now: datetime | None = None, ctx=None) -> None:
    base = now or datetime.now()
    ts = base + timedelta(seconds=elapsed)
    template = ov.get("content") or "%H:%M:%S"
    try:
        text = ts.strftime(template)
    except (ValueError, TypeError):
        text = template  # malformed strftime → render the literal template
    _render_text_layer(canvas, text, ov, ctx)


def _draw_sticker(canvas: Image.Image, ov: Overlay, resample) -> None:
    img = ov.get("image")
    if img is None:
        return
    cw, ch = canvas.size
    img = img.convert("RGBA")
    target_h = max(1, int(ch * float(ov.get("scale", 40)) / 100.0))
    ratio = target_h / img.height
    s = img.resize((max(1, int(img.width * ratio)), target_h), resample)

    rot = float(ov.get("rotation", 0) or 0)
    if rot:
        s = s.rotate(-rot, expand=True, resample=Image.BICUBIC)

    op = float(ov.get("opacity", 1.0))
    if op < 1.0:
        alpha = s.split()[3].point(lambda v: int(v * op))
        s.putalpha(alpha)

    x = cw / 2 + (float(ov.get("x", 0)) / 100.0) * cw
    y = ch / 2 + (float(ov.get("y", 0)) / 100.0) * ch
    canvas.paste(s, (int(x - s.width / 2), int(y - s.height / 2)), s)


# ── Crop-dimming preview ──────────────────────────────────────────────────────
def preview(src_path: str | None, p: PanelParams, frame: Image.Image | None = None) -> Image.Image:
    """Render the editor preview: the exact 4:1 frame bright, the cropped-out
    region of the source dimmed around it, with a frame border + thirds."""
    cw, ch = canvas_size(p.orientation)
    src = frame if frame is not None else _first_image(src_path)

    # Display geometry: frame scaled to a sane size, with margin around it.
    disp_fw = 1000 if cw >= ch else 320
    ds = disp_fw / cw
    disp_fh = round(ch * ds)
    pad_x = round(disp_fw * 0.16)
    pad_y = round(disp_fh * 0.16) if cw >= ch else round(disp_fw * 0.16)
    pw, ph = disp_fw + 2 * pad_x, disp_fh + 2 * pad_y

    # Composite the bright frame directly at display resolution (fast resample,
    # no full-1920 intermediate) — the preview never needs export resolution.
    comp = compose_frame(src, p, fast=True).resize((disp_fw, disp_fh), Image.BILINEAR)

    prev = Image.new("RGB", (pw, ph), (22, 22, 26))

    if src is not None:
        dw, dh = _drawn_size(src.width, src.height, cw, ch, p.fit, p.zoom)
        px, py = _paste_pos(dw, dh, cw, ch, p)
        rs = src.convert("RGBA").resize((max(1, round(dw * ds)), max(1, round(dh * ds))), Image.BILINEAR)
        prev.paste(rs, (round(pad_x + px * ds), round(pad_y + py * ds)), rs)
        # dim everything, then punch the bright frame back in
        prev = Image.blend(prev, Image.new("RGB", (pw, ph), (22, 22, 26)), 0.5)

    prev.paste(comp, (pad_x, pad_y))

    d = ImageDraw.Draw(prev)
    d.rectangle([pad_x, pad_y, pad_x + disp_fw - 1, pad_y + disp_fh - 1], outline=(255, 255, 255), width=2)
    for i in (1, 2):
        gx = pad_x + disp_fw * i // 3
        gy = pad_y + disp_fh * i // 3
        d.line([(gx, pad_y), (gx, pad_y + disp_fh)], fill=(255, 255, 255, 60), width=1)
        d.line([(pad_x, gy), (pad_x + disp_fw, gy)], fill=(255, 255, 255, 60), width=1)
    return prev


# ── 3D-style product mockup ───────────────────────────────────────────────────
def _perspective_coeffs(dst: list, src: list) -> list:
    """8 PIL PERSPECTIVE coefficients mapping the output `dst` quad back to the
    input `src` quad (each a list of 4 (x, y) corners: TL, TR, BR, BL)."""
    m = []
    for (dx, dy), (sx, sy) in zip(dst, src):
        m.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        m.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    A = np.array(m, dtype=np.float64)
    b = np.array(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(A, b).tolist()


def mockup(src_path: str | None, p: PanelParams, width: int = 1400) -> Image.Image:
    """Render the composed panel as a 3D-style product shot of the physical
    Lian Li screen: a bezel-framed display tilted in perspective, with a screen
    glow and a faded floor reflection on a dark studio background.

    Server-side (PIL) so it integrates with the reactive UI and adds no deps.
    Returns an RGB image.
    """
    comp = compose_frame(_first_image(src_path), p).convert("RGB")
    cw, ch = comp.size
    portrait = ch > cw

    # 1. Scale the screen to a sensible size within the canvas.
    if portrait:
        sh = max(1, int(width * 0.46)); sw = max(1, round(sh * cw / ch))
    else:
        sw = max(1, int(width * 0.80)); sh = max(1, round(sw * ch / cw))
    screen = comp.resize((sw, sh), Image.LANCZOS)

    # 2. Bezel: a dark rounded frame around the screen with a faint rim.
    b = max(8, round(min(sw, sh) * 0.05))
    fw, fh = sw + 2 * b, sh + 2 * b
    frame = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    d = ImageDraw.Draw(frame)
    rad = max(10, b)
    d.rounded_rectangle([0, 0, fw - 1, fh - 1], radius=rad, fill=(18, 18, 20, 255))
    d.rounded_rectangle([1, 1, fw - 2, fh - 2], radius=rad, outline=(72, 72, 80, 255), width=2)
    frame.paste(screen, (b, b))

    # 3. Yaw perspective — the right edge recedes for a 3/4 view.
    k = 0.22
    src_q = [(0, 0), (fw, 0), (fw, fh), (0, fh)]
    dst_q = [(0, 0), (fw, fh * k * 0.5), (fw, fh * (1 - k * 0.5)), (0, fh)]
    tilted = frame.transform(
        (fw, fh), Image.PERSPECTIVE, _perspective_coeffs(dst_q, src_q),
        resample=Image.BICUBIC,
    )

    # 4. Dark studio background (vertical gradient).
    height = max(1, round(width * (1.0 if portrait else 0.64)))
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)
    top = np.array([32, 32, 37], np.float32)
    bot = np.array([11, 11, 13], np.float32)
    col = top[None, :] * (1 - yy)[:, None] + bot[None, :] * yy[:, None]
    bg_arr = np.repeat(col[:, None, :], width, axis=1).clip(0, 255).astype(np.uint8)
    bg = Image.fromarray(bg_arr, "RGB")

    sx = (width - fw) // 2
    sy = round(height * (0.13 if not portrait else 0.06))

    # 5. Screen glow — a large soft, low-opacity colour spill onto the studio
    #    (radial alpha falloff so it reads as ambient light, not a rectangle).
    glow = screen.resize((round(fw * 1.35), round(fh * 1.6)), Image.LANCZOS)
    glow = glow.filter(ImageFilter.GaussianBlur(max(22, b * 3)))
    gw, gh = glow.size
    ax = 1 - np.abs(np.linspace(-1, 1, gw, dtype=np.float32))
    ay = 1 - np.abs(np.linspace(-1, 1, gh, dtype=np.float32))
    gmask = (np.outer(ay, ax) * 90).clip(0, 255).astype(np.uint8)
    glow.putalpha(Image.fromarray(gmask, "L"))
    bg.paste(glow, (sx - (gw - fw) // 2, sy - (gh - fh) // 2), glow)

    # 6. Faded floor reflection beneath the screen.
    refl = tilted.transpose(Image.FLIP_TOP_BOTTOM)
    fade = np.repeat(
        np.linspace(80, 0, refl.height, dtype=np.uint8)[:, None], refl.width, axis=1
    )
    ralpha = (np.array(refl.split()[-1], np.uint16) * fade // 255).astype(np.uint8)
    refl.putalpha(Image.fromarray(ralpha, "L"))
    bg.paste(refl, (sx, sy + fh + max(3, b // 2)), refl)

    # 7. The screen itself on top.
    bg.paste(tilted, (sx, sy), tilted)
    return bg


@lru_cache(maxsize=8)
def _decode_first_frame(path: str, _mtime: float) -> Image.Image | None:
    """Decode the first frame of a source file. Cached by (path, mtime) so the
    live preview doesn't re-read/decode the file (or re-run ffmpeg) on every
    control change — that decode was the lag you saw while editing."""
    kind = media_kind(path)
    if kind == "image":
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    if kind == "animated":
        ext = os.path.splitext(str(path))[1].lower()
        if ext == ".gif":
            try:
                with Image.open(path) as im:  # context-manage so the fp closes
                    im.seek(0)
                    return im.convert("RGB")
            except Exception:
                return None
        # video: pull the first frame with ffmpeg (only once per file, cached)
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            subprocess.run(
                [_ffmpeg(), "-y", "-i", str(path), "-frames:v", "1", tmp],
                capture_output=True,
            )
            return Image.open(tmp).convert("RGB")
        except Exception:
            return None
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return None


def _first_image(path: str | None) -> Image.Image | None:
    """First frame of whatever was uploaded, as a PIL image (cached)."""
    if not path:
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return _decode_first_frame(path, mtime)


# ── Export ────────────────────────────────────────────────────────────────────
def export_still(src_path: str | None, p: PanelParams, fmt: str) -> str:
    src = _first_image(src_path)
    comp = compose_frame(src, p)
    suffix = ".jpg" if fmt == "jpeg" else f".{fmt}"
    fd, out = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    if fmt == "jpeg":
        comp.save(out, "JPEG", quality=95)
    else:
        comp.save(out, "PNG")
    return out


def _ffmpeg_run(cmd: list) -> None:
    """Run an ffmpeg command, raising RuntimeError (with a stderr tail) when it
    fails — so a broken extract/encode surfaces as a real error instead of
    silently leaving a 0-byte or truncated output that we'd report as success."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-6:])
        raise RuntimeError(f"ffmpeg failed:\n{tail}")


def _extract_frames(src_path: str, fps: float, start: float, dur: float, into: str) -> int:
    """Extract frames into `into` as frame_%05d.png; return the frame count."""
    cmd = [_ffmpeg(), "-y"]
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(src_path)]
    if dur > 0:
        cmd += ["-t", str(dur)]
    cmd += ["-vf", f"fps={fps}", "-vsync", "0", os.path.join(into, "frame_%05d.png")]
    _ffmpeg_run(cmd)
    return len([f for f in os.listdir(into) if f.startswith("frame_")])


LOOP_STYLES = ["normal", "boomerang", "crossfade"]


def export_animated(
    src_path: str | None,
    p: PanelParams,
    fmt: str,  # "gif" | "mp4"
    fps: int,
    loop: bool,
    gif_colors: int,
    trim_start: float,
    trim_end: float,
    loop_mode: str = "normal",
    progress=None,
) -> str:
    if not src_path:
        raise ValueError("Upload media first.")
    fps = max(1, min(MAX_FPS, int(fps)))
    cw, ch = canvas_size(p.orientation)

    dur_total = media_duration(src_path)
    start = max(0.0, float(trim_start or 0))
    has_end = bool(trim_end and float(trim_end) > start)
    if has_end:
        dur = min(float(trim_end) - start, MAX_DURATION_SEC)
    elif dur_total > start:
        dur = min(dur_total - start, MAX_DURATION_SEC)
    else:
        # Duration unknown (ffprobe returned 0 / N/A) and no explicit end — don't
        # fabricate a 1-second clip; let ffmpeg read to EOF, capped at the max.
        dur = MAX_DURATION_SEC

    work = tempfile.mkdtemp(prefix="panel_")
    raw = os.path.join(work, "raw")
    comp = os.path.join(work, "comp")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(comp, exist_ok=True)
    try:
        n = _extract_frames(src_path, fps, start, dur, raw)
        if n == 0:
            # No video frames decoded: treat as a still and hold it for a short
            # clip — but if even the first frame won't decode, fail loudly rather
            # than emitting a 0-byte "export".
            base = _first_image(src_path)
            if base is None:
                raise ValueError(
                    "Couldn't decode the uploaded media — it may be corrupt or an "
                    "unsupported format."
                )
            hold = dur if has_end else min(dur, 3.0)
            n = max(1, int(round(hold * fps)))
            for i in range(1, n + 1):
                base.save(os.path.join(raw, f"frame_{i:05d}.png"))

        files = sorted(f for f in os.listdir(raw) if f.startswith("frame_"))
        if not files:
            raise ValueError("No frames could be extracted from the source.")
        n_files = len(files)
        for i, fn in enumerate(files, 1):
            with Image.open(os.path.join(raw, fn)) as fr:
                out = compose_frame(
                    fr.convert("RGB"), p,
                    frame_index=i - 1, elapsed=(i - 1) / fps,
                    total_frames=n_files, fps=fps,
                )
            out.save(os.path.join(comp, f"c_{i:05d}.png"))
            if progress:
                progress(0.1 + 0.6 * i / len(files), desc=f"Compositing frame {i}/{len(files)}")

        pattern = _apply_loop(comp, len(files), loop_mode, fps)
        if fmt == "mp4":
            return _encode_mp4(pattern, fps, progress)
        return _encode_gif(pattern, fps, loop, gif_colors, progress)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _apply_loop(comp_dir: str, n: int, mode: str, fps: int) -> str:
    """Reorder / blend the composited frames for a seamless loop and return the
    ffmpeg input pattern. `boomerang` plays forward then back; `crossfade`
    overlaps the tail onto the head so the loop join falls between two original
    consecutive frames (no visible seam)."""
    default = os.path.join(comp_dir, "c_%05d.png")
    if mode not in ("boomerang", "crossfade") or n < 4:
        return default

    seq = os.path.join(comp_dir, "seq")
    os.makedirs(seq, exist_ok=True)
    cpath = lambda i: os.path.join(comp_dir, f"c_{i:05d}.png")  # noqa: E731
    spath = lambda i: os.path.join(seq, f"s_{i:05d}.png")  # noqa: E731

    def link(src: str, dst: str) -> None:
        try:
            os.link(src, dst)  # hardlink — no copy cost on the same fs
        except OSError:
            shutil.copyfile(src, dst)

    if mode == "boomerang":
        order = list(range(1, n + 1)) + list(range(n - 1, 1, -1))  # 1..n, n-1..2
        for k, idx in enumerate(order, 1):
            link(cpath(idx), spath(k))
    else:  # crossfade
        k = max(1, min(int(round(fps * 0.5)), n // 3))  # ~0.5 s crossfade
        length = n - k
        for i in range(1, length + 1):
            if i <= k:
                a = i / (k + 1)  # 0→1: tail fades into head over the first k frames
                tail = Image.open(cpath(n - k + i)).convert("RGB")
                head = Image.open(cpath(i)).convert("RGB")
                Image.blend(tail, head, a).save(spath(i))
            else:
                link(cpath(i), spath(i))
    return os.path.join(seq, "s_%05d.png")


def _encode_mp4(in_pattern: str, fps: int, progress=None) -> str:
    fd, out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    if progress:
        progress(0.8, desc="Encoding H.264…")
    _ffmpeg_run(
        [_ffmpeg(), "-y", "-framerate", str(fps), "-i", in_pattern,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "main",
         "-preset", "medium", "-movflags", "+faststart", "-r", str(fps), out]
    )
    if not os.path.getsize(out):
        raise RuntimeError("ffmpeg produced an empty MP4.")
    if progress:
        progress(1.0)
    return out


def _encode_gif(in_pattern: str, fps: int, loop: bool, colors: int, progress=None) -> str:
    fd, out = tempfile.mkstemp(suffix=".gif")
    os.close(fd)
    if progress:
        progress(0.8, desc="Encoding GIF…")
    colors = max(2, min(256, int(colors)))
    loop_arg = "0" if loop else "-1"
    vf = (
        f"fps={fps},split[s0][s1];[s0]palettegen=max_colors={colors}[p];"
        f"[s1][p]paletteuse=dither=bayer"
    )
    _ffmpeg_run(
        [_ffmpeg(), "-y", "-framerate", str(fps), "-i", in_pattern,
         "-vf", vf, "-loop", loop_arg, out]
    )
    if not os.path.getsize(out):
        raise RuntimeError("ffmpeg produced an empty GIF.")
    if progress:
        progress(1.0)
    return out
