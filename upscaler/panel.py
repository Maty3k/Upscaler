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

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


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
    text: str = ""
    text_size: int = 180
    text_color: str = "#ffffff"
    text_off_x: float = 0.0  # percent of canvas width
    text_off_y: float = 0.0  # percent of canvas height
    text_stroke: str = "#000000"
    text_stroke_w: int = 0


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


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(8, int(size))
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
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
def compose_frame(src: Image.Image, p: PanelParams, fast: bool = False) -> Image.Image:
    """Composite a single source frame onto the exact panel canvas.

    `fast=True` uses bilinear resampling for the live preview (cheaper, still
    crisp on screen); exports leave it False for export-grade LANCZOS.
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

    if p.text.strip():
        _draw_text(canvas, p)
    return canvas


def _draw_text(canvas: Image.Image, p: PanelParams) -> None:
    cw, ch = canvas.size
    draw = ImageDraw.Draw(canvas)
    font = _load_font(p.text_size)
    x = cw / 2 + (p.text_off_x / 100.0) * cw
    y = ch / 2 + (p.text_off_y / 100.0) * ch
    kwargs = dict(font=font, fill=_hex(p.text_color), anchor="mm", align="center")
    if p.text_stroke_w and p.text_stroke_w > 0:
        kwargs["stroke_width"] = int(p.text_stroke_w)
        kwargs["stroke_fill"] = _hex(p.text_stroke)
    draw.multiline_text((x, y), p.text, **kwargs)


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
                im = Image.open(path)
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


def _extract_frames(src_path: str, fps: float, start: float, dur: float, into: str) -> int:
    """Extract frames into `into` as frame_%05d.png; return the frame count."""
    cmd = [_ffmpeg(), "-y"]
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(src_path)]
    if dur > 0:
        cmd += ["-t", str(dur)]
    cmd += ["-vf", f"fps={fps}", "-vsync", "0", os.path.join(into, "frame_%05d.png")]
    subprocess.run(cmd, capture_output=True)
    return len([f for f in os.listdir(into) if f.startswith("frame_")])


def export_animated(
    src_path: str | None,
    p: PanelParams,
    fmt: str,  # "gif" | "mp4"
    fps: int,
    loop: bool,
    gif_colors: int,
    trim_start: float,
    trim_end: float,
    progress=None,
) -> str:
    if not src_path:
        raise ValueError("Upload media first.")
    fps = max(1, min(MAX_FPS, int(fps)))
    cw, ch = canvas_size(p.orientation)

    dur_total = media_duration(src_path)
    start = max(0.0, float(trim_start or 0))
    end = float(trim_end) if trim_end and trim_end > 0 else dur_total
    if end <= start:
        end = dur_total if dur_total > start else start + 1
    dur = min(end - start, MAX_DURATION_SEC)

    work = tempfile.mkdtemp(prefix="panel_")
    raw = os.path.join(work, "raw")
    comp = os.path.join(work, "comp")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(comp, exist_ok=True)
    try:
        n = _extract_frames(src_path, fps, start, dur, raw)
        if n == 0:
            # Static source treated as animated: synthesise a held frame.
            base = _first_image(src_path)
            n = max(1, int(round(dur * fps)))
            for i in range(1, n + 1):
                base.save(os.path.join(raw, f"frame_{i:05d}.png")) if base else None

        files = sorted(f for f in os.listdir(raw) if f.startswith("frame_"))
        for i, fn in enumerate(files, 1):
            with Image.open(os.path.join(raw, fn)) as fr:
                out = compose_frame(fr.convert("RGB"), p)
            out.save(os.path.join(comp, f"c_{i:05d}.png"))
            if progress:
                progress(0.1 + 0.6 * i / len(files), desc=f"Compositing frame {i}/{len(files)}")

        if fmt == "mp4":
            return _encode_mp4(comp, fps, progress)
        return _encode_gif(comp, fps, loop, gif_colors, progress)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _encode_mp4(comp_dir: str, fps: int, progress=None) -> str:
    fd, out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    if progress:
        progress(0.8, desc="Encoding H.264…")
    subprocess.run(
        [_ffmpeg(), "-y", "-framerate", str(fps), "-i", os.path.join(comp_dir, "c_%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "main",
         "-preset", "medium", "-movflags", "+faststart", "-r", str(fps), out],
        capture_output=True,
    )
    if progress:
        progress(1.0)
    return out


def _encode_gif(comp_dir: str, fps: int, loop: bool, colors: int, progress=None) -> str:
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
    subprocess.run(
        [_ffmpeg(), "-y", "-framerate", str(fps), "-i", os.path.join(comp_dir, "c_%05d.png"),
         "-vf", vf, "-loop", loop_arg, out],
        capture_output=True,
    )
    if progress:
        progress(1.0)
    return out
