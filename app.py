"""Local drag-and-drop GUI for Upscaler. Run: `python app.py` then open the URL.

Everything runs on your machine — Gradio just serves a local web UI. No data
leaves your computer.

Two tools on one page:
  • File Converter — fast, lossless-where-possible format conversion (no models).
  • Upscale & Enhance — Real-ESRGAN upscaling, optional NAFNet deblur + sharpen.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import zipfile
from datetime import datetime

import gradio as gr
from PIL import Image

from upscaler import background, panel
from upscaler.convert import FORMATS, convert, extension_for
from upscaler.document import images_to_pdf, pdf_to_images
from upscaler.deblur import Deblurrer
from upscaler.engine import Upscaler, resolve_device
from upscaler.models.registry import DEBLUR_MODELS, MODELS
from upscaler.sharpen import unsharp_mask

# Cache loaded models so switching images doesn't reload weights every run.
# Keyed by (model, device, onnx) so torch and ONNX engines are cached separately.
_UP_CACHE: dict[tuple, object] = {}
_DB_CACHE: dict[tuple, object] = {}


def _onnx_engines():
    try:
        from upscaler.onnx_engine import OnnxDeblurrer, OnnxUpscaler
    except ImportError as e:
        raise gr.Error(
            'ONNX backend needs optional deps — install with: pip install -e ".[onnx]"'
        ) from e
    return OnnxUpscaler, OnnxDeblurrer


def _get_upscaler(model: str, device: str, tile: int, onnx: bool):
    key = (model, device, onnx)
    up = _UP_CACHE.get(key)
    if up is None or getattr(up, "tile", None) != tile:
        if onnx:
            OnnxUpscaler, _ = _onnx_engines()
            up = OnnxUpscaler(model=model, device=device, tile=tile)
        else:
            up = Upscaler(model=model, device=device, tile=tile)
        _UP_CACHE[key] = up
    return up


def _get_deblurrer(model: str, device: str, onnx: bool):
    key = (model, device, onnx)
    db = _DB_CACHE.get(key)
    if db is None:
        if onnx:
            _, OnnxDeblurrer = _onnx_engines()
            db = OnnxDeblurrer(model=model, device=device)
        else:
            db = Deblurrer(model=model, device=device)
        _DB_CACHE[key] = db
    return db


# -- File converter ----------------------------------------------------------

def convert_image(image, fmt, quality, lossless):
    if image is None:
        raise gr.Error("Upload an image to convert first.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    data = convert(img, fmt, quality=int(quality), lossless=bool(lossless))

    fd, path = tempfile.mkstemp(suffix=f".{extension_for(fmt)}")
    with os.fdopen(fd, "wb") as f:
        f.write(data)

    kb = len(data) / 1024
    note = "lossless" if (lossless and fmt == "WebP") or not FORMATS[fmt][2] else f"q{int(quality)}"
    return path, f"✅ Converted to **{fmt}** ({note}) · {kb:,.1f} KB · {img.width}×{img.height}px"


# -- Image <-> PDF -----------------------------------------------------------

def build_pdf(files):
    if not files:
        raise gr.Error("Add at least one image.")
    images = [Image.open(f) for f in files]
    data = images_to_pdf(images)
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path, f"✅ {len(images)} image(s) → PDF · {len(data) / 1024:,.1f} KB"


def extract_pdf(pdf_file, dpi):
    if pdf_file is None:
        raise gr.Error("Upload a PDF first.")
    try:
        pages = pdf_to_images(pdf_file, dpi=int(dpi))
    except ImportError as e:
        raise gr.Error(str(e)) from e

    fd, zpath = tempfile.mkstemp(suffix=".zip")
    with os.fdopen(fd, "wb") as fh, zipfile.ZipFile(fh, "w") as z:
        for i, im in enumerate(pages, 1):
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            z.writestr(f"page_{i:03d}.png", buf.getvalue())
    return zpath, pages, f"✅ {len(pages)} page(s) → PNG · {int(dpi)} dpi (zip)"


# -- Background removal -------------------------------------------------------

_BG_CHOICES = [(f"{s.name} — {s.notes}", s.name) for s in background.BG_MODELS.values()]


def remove_bg_ui(image, model, feather, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload an image first.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    progress(0.2, desc="Loading model…")
    try:
        cut = background.remove_background(img.convert("RGB"), model=model,
                                           feather=int(feather))
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise gr.Error(str(e)) from e
    progress(0.9, desc="Saving transparent PNG…")
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cut.save(path, "PNG")  # PNG keeps the alpha channel
    preview = background.on_checkerboard(cut)
    return preview, path, (
        f"✅ Background removed — {cut.width}×{cut.height}px transparent PNG. "
        "Drop it into the Lian Li tab as a sticker."
    )


# -- Upscale & enhance -------------------------------------------------------

def enhance(image, model, device, deblur, deblur_model, sharpen, tile, onnx, out_size):
    if image is None:
        raise gr.Error("Upload an image to enhance first.")
    original = image if isinstance(image, Image.Image) else Image.fromarray(image)
    src = original
    stages = []
    if deblur:
        src = _get_deblurrer(deblur_model, device, onnx).deblur(src)
        stages.append(f"deblur `{deblur_model}`")
    up = _get_upscaler(model, device, int(tile), onnx)
    result = up.upscale(src)
    stages.append(f"upscale ×{up.scale}")
    if sharpen > 0:
        result = unsharp_mask(result, strength=float(sharpen))
        stages.append(f"sharpen {sharpen:g}")

    # Fit to a target resolution preset (longest edge), if chosen.
    target = _SIZE_PRESETS.get(out_size)
    if target:
        w, h = result.size
        longest = max(w, h)
        if longest != target:
            r = target / longest
            result = result.resize(
                (max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS
            )
            stages.append(f"→ {target}px")

    backend = "onnx" if onnx else getattr(up, "device", None) and up.device.type
    info = (
        "✅ " + " → ".join(stages)
        + f" · backend `{backend}` · {result.width}×{result.height}px"
    )
    # (before, after) for the comparison slider
    return (original, result), info


def restore_only(image, deblur_model, sharpen, device, onnx):
    """Run just the NAFNet restoration pass (deblur or denoise) — no upscaling."""
    if image is None:
        raise gr.Error("Upload an image to restore first.")
    original = image if isinstance(image, Image.Image) else Image.fromarray(image)
    result = _get_deblurrer(deblur_model, device, onnx).deblur(original)
    stages = [f"restore `{deblur_model}`"]
    if sharpen > 0:
        result = unsharp_mask(result, strength=float(sharpen))
        stages.append(f"sharpen {sharpen:g}")
    info = (
        "✅ " + " → ".join(stages)
        + f" · {result.width}×{result.height}px (no upscale)"
    )
    return (original, result), info


# -- Video (frame-by-frame) --------------------------------------------------

def _video_duration(path):
    """Clip length in seconds (0 if unknown)."""
    import shutil
    import subprocess

    fp = shutil.which("ffprobe")
    if not fp or not path:
        return 0
    out = subprocess.run(
        [fp, "-v", "error", "-show_entries", "format=duration", "-of",
         "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return round(float(out), 1)
    except ValueError:
        return 0


def _on_video_change(path):
    """When a clip is loaded, reset Start to 0 and End to the full duration so
    'trim a section' just means lowering End / raising Start."""
    dur = _video_duration(path)
    return gr.update(value=0), gr.update(value=dur, maximum=dur or None)


def _first_frame(video_path):
    """Grab the first frame of a video as a PIL image (for the comparison)."""
    import subprocess

    from upscaler.video import _ffmpeg

    fd, p = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    subprocess.run(
        [_ffmpeg(), "-y", "-i", str(video_path), "-frames:v", "1", p],
        capture_output=True,
    )
    return Image.open(p)


def upscale_video_ui(video_path, model, out_size, sharpen, smooth, trim_start,
                     trim_end, device, tile, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a video first.")
    from upscaler.video import upscale_video

    fd, out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    fps = None if smooth in (None, "Off") else int(smooth)
    target = _SIZE_PRESETS.get(out_size)
    start = float(trim_start) if trim_start and trim_start > 0 else None
    end = float(trim_end) if trim_end and trim_end > 0 else None

    def cb(i, n):
        progress(i / n, desc=f"Upscaling frame {i}/{n}")

    try:
        upscale_video(
            video_path, out, model=model, device=device, tile=int(tile),
            sharpen=float(sharpen), interpolate_fps=fps, target_long_edge=target,
            trim_start=start, trim_end=end, progress_cb=cb,
        )
    except (RuntimeError, FileNotFoundError) as e:
        raise gr.Error(str(e)) from e

    try:
        compare = (_first_frame(video_path), _first_frame(out))
    except Exception:
        compare = None
    extra = (f" · {target}px" if target else "") + (f" · {fps} fps" if fps else "")
    if start or end:
        extra += f" · trim {start or 0:g}–{end if end else 'end'}s"
    return out, compare, f"✅ Done — preview and download below.{extra}"


_CONVERT_METHODS = ["Change image format", "Images → PDF", "PDF → Images"]

# Output-size presets for upscaling: AI-upscale with the model, then fit the
# longest edge to this many pixels (None = leave at the model's native scale).
_SIZE_PRESETS: dict[str, int | None] = {
    "Model default (×2/×4)": None,
    "HD · 1280px": 1280,
    "Full HD 1080p · 1920px": 1920,
    "QHD 1440p · 2560px": 2560,
    "4K UHD · 3840px": 3840,
    "8K · 7680px": 7680,
}


def _switch_method(choice):
    """Show only the group for the selected conversion method."""
    return (
        gr.update(visible=choice == _CONVERT_METHODS[0]),
        gr.update(visible=choice == _CONVERT_METHODS[1]),
        gr.update(visible=choice == _CONVERT_METHODS[2]),
    )


# -- Lian Li 8.8" panel builder ----------------------------------------------

# Overlay slots: up to N_TEXT styled text layers + N_STICKER image stickers.
N_TEXT = 3
N_STICKER = 2
_TEXT_FIELDS = 11   # enabled, content, font, size, color, align, x, y, rot, stroke, stroke_w
_STICKER_FIELDS = 7  # enabled, image, scale, x, y, rot, opacity
N_OVERLAY_VALS = N_TEXT * _TEXT_FIELDS + N_STICKER * _STICKER_FIELDS


def _panel_params(orientation, fit, zoom, off_x, off_y, bg_type, bg_color,
                  bg_color2, bg_angle, *ov):
    """Build PanelParams from the base controls plus the flat overlay-slot
    values (text slots first, then sticker slots)."""
    overlays = []
    i = 0
    for _ in range(N_TEXT):
        en, content, font, size, color, align, x, y, rot, stroke, stroke_w = ov[i:i + _TEXT_FIELDS]
        i += _TEXT_FIELDS
        if en and (content or "").strip():
            overlays.append(dict(
                type="text", content=content, font=font, size=int(size),
                color=color, align=align, x=float(x), y=float(y),
                rotation=float(rot), stroke=stroke, stroke_w=int(stroke_w),
            ))
    for _ in range(N_STICKER):
        en, image, scale, x, y, rot, opacity = ov[i:i + _STICKER_FIELDS]
        i += _STICKER_FIELDS
        if en and image is not None:
            overlays.append(dict(
                type="sticker", image=image, scale=float(scale), x=float(x),
                y=float(y), rotation=float(rot), opacity=float(opacity),
            ))
    return panel.PanelParams(
        orientation=orientation, fit=fit, zoom=float(zoom),
        off_x=float(off_x), off_y=float(off_y), bg_type=bg_type,
        bg_color=bg_color, bg_color2=bg_color2, bg_angle=float(bg_angle),
        overlays=overlays,
    )


def panel_preview_ui(media, *vals):
    """Live crop-dimming preview (bright = kept, dim = cropped out)."""
    return panel.preview(media, _panel_params(*vals))


def panel_on_media(media):
    """On upload: fill trim End with the clip length, reveal the animation
    controls for animated sources, and describe what was loaded."""
    kind = panel.media_kind(media)
    dur = panel.media_duration(media) if kind == "animated" else 0.0
    is_anim = kind == "animated"
    if not media:
        note = "Upload an image, GIF or video to begin."
    elif is_anim:
        note = f"Animated source · {dur:g}s — trims to ≤ 3 min on export."
    else:
        note = "Still image loaded."
    return (
        gr.update(value=dur, maximum=dur or None),
        gr.update(visible=is_anim),
        note,
    )


def panel_export_ui(media, orientation, fit, zoom, off_x, off_y, bg_type,
                    bg_color, bg_color2, bg_angle, *rest, progress=gr.Progress()):
    # rest = <overlay slot values> + [out_fmt, fps, loop, gif_colors,
    #         trim_start, trim_end, loop_mode, out_dir]
    if not media:
        raise gr.Error("Upload media first.")
    ov_vals = rest[:N_OVERLAY_VALS]
    (out_fmt, fps, loop, gif_colors, trim_start, trim_end,
     loop_mode, out_dir) = rest[N_OVERLAY_VALS:]
    p = _panel_params(orientation, fit, zoom, off_x, off_y, bg_type, bg_color,
                      bg_color2, bg_angle, *ov_vals)
    cw, ch = panel.canvas_size(orientation)
    fmt = out_fmt.lower()
    progress(0.05, desc="Preparing…")
    try:
        if fmt in ("png", "jpg"):
            f = panel.export_still(media, p, "jpeg" if fmt == "jpg" else "png")
            msg = f"✅ {out_fmt} exported — exactly {cw}×{ch}px."
        else:
            f = panel.export_animated(
                media, p, "mp4" if fmt == "mp4" else "gif", int(fps), bool(loop),
                int(gif_colors), float(trim_start or 0), float(trim_end or 0),
                loop_mode=loop_mode, progress=progress,
            )
            extra = "" if loop_mode == "normal" else f" · {loop_mode} loop"
            detail = "H.264 / yuv420p" if fmt == "mp4" else f"{int(gif_colors)} colors"
            msg = f"✅ {out_fmt} exported — {cw}×{ch}px · {detail}{extra}."
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise gr.Error(str(e)) from e

    # Optionally drop a timestamped copy into a chosen folder (e.g. the
    # L-Connect media folder) so it lands where it's actually used.
    out_dir = (out_dir or "").strip()
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            name = f"lianli_{datetime.now():%Y%m%d_%H%M%S}{os.path.splitext(f)[1]}"
            dest = os.path.join(out_dir, name)
            shutil.copyfile(f, dest)
            msg += f"\n\n📁 Saved a copy to `{dest}`"
        except OSError as e:
            msg += f"\n\n⚠ Couldn't save to `{out_dir}`: {e}"
    return f, msg


def panel_enhance_source(media, up_model, *vals, progress=gr.Progress()):
    """Run the dropped source through the AI upscaler, then replace the working
    source with the enhanced version — so fitting/export use crisp pixels. Best
    for low-res sources the 1920×480 panel would otherwise show soft."""
    if not media:
        raise gr.Error("Upload media first.")
    kind = panel.media_kind(media)
    progress(0.1, desc="Loading model…")
    try:
        if kind == "image":
            img = Image.open(media).convert("RGB")
            up = _get_upscaler(up_model, "auto", 512, False)
            progress(0.4, desc="Upscaling…")
            result = up.upscale(img)
            fd, out = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            result.save(out, "PNG")
            note = (f"✨ Source upscaled ×{up.scale} → {result.width}×{result.height}px. "
                    "Re-fit and export.")
        elif kind == "animated":
            from upscaler.video import upscale_video

            fd, out = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)

            def cb(i, n):
                progress(i / n, desc=f"Upscaling frame {i}/{n}")

            upscale_video(media, out, model=up_model, device="auto", tile=512,
                          progress_cb=cb)
            note = "✨ Video source upscaled. Re-fit and export."
        else:
            raise gr.Error("Unsupported source for AI upscale.")
    except (RuntimeError, FileNotFoundError) as e:
        raise gr.Error(str(e)) from e
    return gr.update(value=out), panel.preview(out, _panel_params(*vals)), note


_MODEL_CHOICES = [(f"{s.name}  (×{s.scale}) — {s.notes}", s.name) for s in MODELS.values()]
_DEBLUR_CHOICES = [(f"{s.name} — {s.notes}", s.name) for s in DEBLUR_MODELS.values()]
_DEVICES = ["auto", "cpu", "cuda", "mps"]

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Manrope"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
    spacing_size=gr.themes.sizes.spacing_lg,
    text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill="#E7E3DB",
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#F1EEE8",
    body_text_color="#1A1714",
    body_text_color_subdued="#6E675E",
    block_background_fill="#FFFFFF",
    block_border_color="#D5CEC2",
    block_border_width="1px",
    block_radius="16px",
    block_shadow="0 1px 2px rgba(28,25,23,0.05), 0 8px 24px rgba(28,25,23,0.06)",
    block_label_text_weight="600",
    block_label_text_color="#57534E",
    block_label_background_fill="#FFFFFF",
    block_label_border_color="#E2DED7",
    block_title_text_color="#3F3B37",
    block_info_text_color="#78716C",
    panel_background_fill="#FFFFFF",
    input_background_fill="#FFFFFF",
    input_border_color="#DED9D1",
    input_border_color_focus="#0D9488",
    button_primary_background_fill="#0D9488",
    button_primary_background_fill_hover="#0F766E",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="#0D9488",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_border_color="#E7E5E4",
    button_large_radius="12px",
    button_small_radius="10px",
    # Warm, softer dark palette with layered surfaces (not near-black), so the
    # dark toggle has depth instead of feeling like a void.
    body_background_fill_dark="#161311",
    background_fill_primary_dark="#322C27",
    background_fill_secondary_dark="#211D1A",
    body_text_color_dark="#FAF7F3",
    body_text_color_subdued_dark="#BCB2A7",
    block_background_fill_dark="#322C27",
    block_border_color_dark="#473F37",
    block_label_text_color_dark="#EFE9E2",
    block_label_background_fill_dark="#3A332D",
    block_label_border_color_dark="#473F37",
    block_title_text_color_dark="#F1ECE6",
    block_info_text_color_dark="#BCB2A7",
    panel_background_fill_dark="#322C27",
    input_background_fill_dark="#262119",
    input_border_color_dark="#473F37",
    input_border_color_focus_dark="#2DD4BF",
    button_primary_background_fill_dark="#14B8A6",
    button_primary_background_fill_hover_dark="#2DD4BF",
    button_primary_text_color_dark="#06231F",
    button_secondary_background_fill_dark="#2E2A26",
    button_secondary_border_color_dark="#3B342D",
)

# Apply the saved light/dark preference on load (default light), and a toggle
# that flips it. Both go through Gradio's own ?__theme mechanism (one reload).
_APPLY_THEME_JS = """
() => {
  const u = new URL(window.location.href);
  const saved = localStorage.getItem('upscaler-theme') || 'light';
  if (u.searchParams.get('__theme') !== saved) {
    u.searchParams.set('__theme', saved);
    window.location.replace(u.toString());
  }
}
"""

_TOGGLE_THEME_JS = """
() => {
  const u = new URL(window.location.href);
  const cur = u.searchParams.get('__theme')
              || localStorage.getItem('upscaler-theme') || 'light';
  const next = cur === 'dark' ? 'light' : 'dark';
  localStorage.setItem('upscaler-theme', next);
  u.searchParams.set('__theme', next);
  window.location.replace(u.toString());
}
"""

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

/* Custom CSS uses Gradio theme vars (--body-text-color etc.) so it adapts to
   both light and dark automatically. --ac is the accent (teal), brighter in dark. */
.gradio-container { --ac: #0F766E; --ac-weak: rgba(13,148,136,.10);
    --tex: rgba(28,25,23,.07); }
.dark .gradio-container, .dark { --ac: #2DD4BF; --ac-weak: rgba(45,212,191,.13);
    --tex: rgba(255,255,255,.06); }

/* gradio-app carries the .dark scope, so fill the viewport with IT (html/body
   sit outside the scope and would otherwise show a strip behind the app). The
   html/body fallback covers light mode; gradio-app (100vh) covers dark. */
html, body { background: var(--body-background-fill) !important; }
gradio-app { display: block; min-height: 100vh;
    background: var(--body-background-fill) !important; }
/* The container has the accent + texture vars in scope, so the warm glow and a
   faint dot texture go here (not on gradio-app, where those vars are undefined). */
.gradio-container { max-width: 100% !important; padding: 6px 44px 64px !important;
    position: relative;
    background:
      radial-gradient(1100px 460px at 18% -8%, var(--ac-weak), transparent 62%),
      radial-gradient(var(--tex) 1px, transparent 1.4px) 0 0 / 22px 22px,
      var(--body-background-fill) !important; }

/* Cheap transitions on interactive controls ONLY — color/border, no box-shadow
   or transform on every .block (that caused heavy repaints / ~20fps jank). */
button, .tab-nav button, .drop, .item, .dropdown-arrow {
    transition: background-color .18s ease, border-color .18s ease, color .18s ease; }
/* transform/box-shadow only on the single button being hovered (cheap). */
.gradio-container button.primary {
    transition: background-color .18s ease, transform .12s ease, box-shadow .2s ease; }
.gradio-container button.primary:hover { transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(13,148,136,.28); }
.gradio-container button.primary:active { transform: translateY(0); box-shadow: none; }
.tab-nav button:hover { color: var(--body-text-color) !important; }

/* Entrance fades — only opacity + a tiny transform, GPU-composited (will-change)
   so they stay smooth. Hero/section heads fill `both` (load-time); tab/accordion
   bodies use no fill (default opacity 1) so they can't get stuck invisible. */
@keyframes fadeUp { from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: none; } }
#hero, .sec-head, .tabitem, [data-testid="accordion-content"] {
    will-change: opacity, transform; }
#hero { animation: fadeUp .55s cubic-bezier(.22,.61,.36,1) both; }
.sec-head { animation: fadeUp .55s cubic-bezier(.22,.61,.36,1) .05s both; }
.tabitem { animation: fadeUp .45s cubic-bezier(.22,.61,.36,1); }
[data-testid="accordion-content"] { animation: fadeUp .4s cubic-bezier(.22,.61,.36,1); }
.label-wrap .icon { transition: transform .25s cubic-bezier(.22,.61,.36,1) !important; }

/* dropdown list: smooth open (was an abrupt instant pop) */
@keyframes ddOpen { from { opacity: 0; transform: translateY(-5px) scale(.99); }
    to { opacity: 1; transform: none; } }
/* z-index so an open list sits ABOVE other controls instead of overlapping
   them; box-shadow + solid bg so it reads as a floating popover. */
ul.options, .options { animation: ddOpen .2s cubic-bezier(.22,.61,.36,1);
    transform-origin: top center; z-index: 200 !important;
    box-shadow: 0 8px 28px rgba(28,25,23,.16) !important;
    background: var(--block-background-fill) !important; }
ul.options .item, .options .item { transition: background-color .12s ease; }
.dropdown-arrow { transition: transform .25s cubic-bezier(.22,.61,.36,1); }

@media (prefers-reduced-motion: reduce) {
    #hero, .sec-head, .tabitem, [data-testid="accordion-content"],
    ul.options, .options { animation: none; } }
@keyframes livepulse { 0% { box-shadow: 0 0 0 0 rgba(34,197,94,.45); }
    70% { box-shadow: 0 0 0 7px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); } }

#hero { padding: 32px 2px 18px; margin-bottom: 8px;
    border-bottom: 1px solid var(--border-color-primary); }
#hero .brandrow { display: flex; align-items: center; gap: 11px; }
#hero .logo { color: var(--ac); display: inline-flex; }
#hero .brand { font-size: 2.05rem; font-weight: 800; letter-spacing: -0.03em;
    margin: 0; color: var(--body-text-color); }
#hero .sub { color: var(--body-text-color-subdued); margin: 9px 0 14px;
    font-size: 1.02rem; max-width: 64ch; }
.pill { display: inline-flex; align-items: center; gap: 7px; padding: 5px 12px;
    border: 1px solid var(--border-color-primary); border-radius: 999px;
    font-size: 0.8rem; color: var(--body-text-color-subdued);
    background: var(--block-background-fill); font-weight: 500; }
.pill .dot { width: 7px; height: 7px; border-radius: 999px; background: #22C55E;
    animation: livepulse 2.4s ease-out infinite; }

/* theme toggle, floated top-right */
#theme-toggle { position: absolute; top: 20px; right: 40px; z-index: 50;
    width: auto !important; min-width: 0 !important; flex: none !important; }

/* tab bar: accent the selected tab */
.tabitem { padding-top: 28px !important; }
.tab-nav { gap: 2px; }
.tab-nav button { font-weight: 600 !important; font-size: 0.98rem !important;
    color: var(--body-text-color-subdued) !important; border: none !important;
    border-bottom: 2px solid transparent !important; border-radius: 0 !important; }
.tab-nav button.selected { color: var(--ac) !important;
    border-bottom: 2px solid var(--ac) !important; }

/* section heads: accent eyebrow w/ icon + underlined title */
.sec-head { margin-bottom: 8px; }
.sec-head .eyebrow { display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.13em;
    text-transform: uppercase; color: var(--ac); }
.sec-head .eyebrow .ic { display: inline-flex; align-items: center; }
.sec-head h2 { position: relative; display: inline-block; font-size: 1.32rem;
    font-weight: 700; letter-spacing: -0.02em; margin: 5px 0 9px;
    padding-bottom: 8px; color: var(--body-text-color); }
.sec-head h2::after { content: ""; position: absolute; left: 0; bottom: 0;
    width: 40px; height: 2px; border-radius: 2px; background: var(--ac); }
.sec-head p { color: var(--body-text-color-subdued); margin: 0; font-size: 0.92rem;
    max-width: 72ch; line-height: 1.55; }
.col-label { font-weight: 600; color: var(--body-text-color); font-size: 0.92rem;
    border-left: 3px solid var(--ac); padding-left: 9px; }
.spacer { height: 22px; }

/* clean, friendly drop zones for image/file inputs */
.drop { border: 1.5px dashed var(--border-color-primary) !important;
    background: var(--block-background-fill) !important;
    border-radius: 14px !important; box-shadow: none !important;
    transition: border-color .15s ease, background .15s ease; }
.drop:hover { border-color: var(--ac) !important; background: var(--ac-weak) !important; }

footer { display: none !important; }
"""


def _svg(paths: str) -> str:
    return (
        '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round">{paths}</svg>'
    )


# Section / brand icons (stroke = currentColor, so they pick up the accent).
ICON_AI = _svg('<path d="M12 3l1.9 4.6L18.5 9.5 13.9 11.4 12 16l-1.9-4.6L5.5 9.5'
               'l4.6-1.9z"/><path d="M19 14l.6 1.6 1.6.6-1.6.6L19 19l-.6-1.6'
               '-1.6-.6 1.6-.6z"/>')
ICON_CONVERT = _svg('<path d="M7 4 3 8l4 4"/><path d="M3 8h14"/>'
                    '<path d="m17 20 4-4-4-4"/><path d="M21 16H7"/>')
ICON_PDF = _svg('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 '
                '2-2V8z"/><path d="M14 3v5h5"/>')
ICON_PANEL = _svg('<rect x="2" y="8" width="20" height="8" rx="1.5"/>'
                  '<path d="M6 12h.01M9 12h.01"/>')
ICON_LOGO = (
    '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2.1" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="m12 3 9 5-9 5-9-5 9-5z"/>'
    '<path d="m3 13 9 5 9-5"/></svg>'
)


def _section_head(eyebrow: str, title: str, desc: str, icon: str = "") -> str:
    return (
        '<div class="sec-head">'
        f'<div class="eyebrow"><span class="ic">{icon}</span>{eyebrow}</div>'
        f"<h2>{title}</h2><p>{desc}</p></div>"
    )


def build_demo() -> gr.Blocks:
    device_name = resolve_device("auto").type
    with gr.Blocks(title="Upscaler") as demo:
        gr.HTML(
            '<div id="hero">'
            f'<div class="brandrow"><span class="logo">{ICON_LOGO}</span>'
            '<span class="brand">Upscaler</span></div>'
            '<div class="sub">A quiet little toolbox for images — convert formats, '
            "make and split PDFs, and upscale with AI. Everything runs locally; "
            "nothing is uploaded.</div>"
            f'<span class="pill"><span class="dot"></span>running locally · {device_name}</span>'
            "</div>"
        )
        theme_btn = gr.Button(
            "◐ Light / Dark", elem_id="theme-toggle", size="sm", variant="secondary"
        )
        theme_btn.click(None, js=_TOGGLE_THEME_JS)

        with gr.Tabs():
            # ---- Tab 1: Upscale & Enhance ----
            with gr.Tab("Upscale"):
                gr.HTML(_section_head(
                    "AI", "Upscale & Enhance",
                    "Real-ESRGAN super-resolution, with optional deblur and "
                    "sharpening. Tip: for an already-decent photo, prefer the ×2 "
                    "model — ×4 can over-process clean images.",
                    icon=ICON_AI,
                ))
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        inp = gr.Image(
                            label="Input", type="pil",
                            sources=["upload", "clipboard"], height=300,
                            elem_classes="drop", buttons=["download", "fullscreen"],
                        )
                        model = gr.Dropdown(
                            _MODEL_CHOICES, value="realesrgan-x4plus",
                            label="Upscale model",
                            info="×2 is gentler for already-good photos · ×4 adds "
                            "the most detail but can over-process · anime model is "
                            "for illustrations & line art.",
                        )
                        out_size = gr.Dropdown(
                            list(_SIZE_PRESETS), value="Model default (×2/×4)",
                            label="Output size",
                            info="After AI upscaling, fit the longest edge to this "
                            "size. 4K = 3840px. Pick the AI model that overshoots "
                            "your target, then it's resized down to stay crisp.",
                        )
                        sharpen = gr.Slider(
                            0.0, 3.0, value=0.0, step=0.1,
                            label="Sharpen (unsharp mask) — 0 = off",
                            info="Crispens edges after upscaling. Keep it low — too "
                            "high adds halos around edges.",
                        )
                        with gr.Accordion("Restore: deblur / denoise (NAFNet)", open=True):
                            deblur = gr.Checkbox(
                                value=False, label="Restore first (when upscaling)",
                                info="Tick to run the restoration pass before "
                                "upscaling when you click Enhance. To ONLY restore "
                                "(no upscale), use the button below instead.",
                            )
                            deblur_model = gr.Dropdown(
                                _DEBLUR_CHOICES, value="nafnet-gopro-width64",
                                label="Restoration model",
                                info="GoPro = motion deblur (width64 best, width32 "
                                "faster) · SIDD = denoise grain/noise.",
                            )
                            restore_btn = gr.Button(
                                "✨ Restore only (deblur / denoise · no upscale)",
                                variant="secondary",
                            )
                        with gr.Accordion("Advanced", open=False):
                            device = gr.Dropdown(
                                _DEVICES, value="auto", label="Device",
                                info="auto picks a GPU if available (CUDA / Apple "
                                "MPS), otherwise CPU.",
                            )
                            onnx = gr.Checkbox(
                                value=False, label="ONNX Runtime backend",
                                info="Exports the model to ONNX once, then runs "
                                "without PyTorch — often faster on CPU.",
                            )
                            tile = gr.Slider(
                                0, 1024, value=512, step=64,
                                label="Tile size (0 = off)",
                                info="Processes big images in tiles to save memory. "
                                "Lower this if you hit out-of-memory errors.",
                            )
                        with gr.Row():
                            run = gr.Button(
                                "Enhance", variant="primary", size="lg", scale=3
                            )
                            clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1):
                        out = gr.ImageSlider(
                            label="Before / after — drag the divider to compare",
                            type="pil", height=300, buttons=["download", "fullscreen"],
                        )
                        info = gr.Markdown()

            # ---- Tab 2: Video (frame-by-frame) ----
            with gr.Tab("Video"):
                gr.HTML(_section_head(
                    "AI · Video", "Video Upscaler",
                    "Upscale a clip frame-by-frame (offline, audio kept). ×2 is "
                    "faster and flickers less than ×4. Long clips take a while — "
                    "needs ffmpeg installed.",
                    icon=ICON_AI,
                ))
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        vid_in = gr.Video(label="Input video", sources=["upload"])
                        vid_model = gr.Dropdown(
                            _MODEL_CHOICES, value="realesrgan-x2plus",
                            label="Upscale model",
                            info="×2 recommended for video — faster, less shimmer "
                            "between frames.",
                        )
                        vid_size = gr.Dropdown(
                            list(_SIZE_PRESETS), value="Model default (×2/×4)",
                            label="Output size",
                            info="Fit the longest edge to this size after upscaling "
                            "(e.g. 4K = 3840px). Resizes every frame.",
                        )
                        vid_sharpen = gr.Slider(
                            0.0, 3.0, value=0.0, step=0.1,
                            label="Sharpen per frame — 0 = off",
                            info="Be gentle on video; sharpening can amplify "
                            "frame-to-frame flicker.",
                        )
                        vid_smooth = gr.Dropdown(
                            ["Off", "30", "48", "60", "120"], value="Off",
                            label="Smooth motion (interpolate to fps)",
                            info="Adds motion-interpolated frames for smoother "
                            "playback. Higher = smoother but much slower.",
                        )
                        with gr.Accordion("Trim (process only part of the clip)", open=False):
                            gr.Markdown(
                                "Render just a slice — great for testing settings "
                                "before the full clip. **End auto-fills to the clip "
                                "length on upload**; for a section, lower **End** "
                                "and/or raise **Start**."
                            )
                            with gr.Row():
                                vid_start = gr.Number(
                                    value=0, label="Start (seconds)", minimum=0,
                                )
                                vid_end = gr.Number(
                                    value=0, label="End (seconds)", minimum=0,
                                )
                        with gr.Accordion("Advanced", open=False):
                            vid_device = gr.Dropdown(
                                _DEVICES, value="auto", label="Device",
                                info="auto picks a GPU if available, else CPU.",
                            )
                            vid_tile = gr.Slider(
                                0, 1024, value=512, step=64,
                                label="Tile size (0 = off)",
                                info="Lower if you hit out-of-memory on big frames.",
                            )
                        with gr.Row():
                            vid_btn = gr.Button(
                                "Upscale video", variant="primary", size="lg", scale=3
                            )
                            vid_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1):
                        vid_out = gr.Video(label="Result", buttons=["download"])
                        vid_compare = gr.ImageSlider(
                            label="First frame — before / after (drag to compare)",
                            type="pil", height=220, buttons=["download", "fullscreen"],
                        )
                        vid_info = gr.Markdown()

            # ---- Tab 3: Convert (all conversions, picked via dropdown) ----
            with gr.Tab("Convert"):
                gr.HTML(_section_head(
                    "Convert", "Convert & Documents",
                    "Pick what you want to do — change image format, build a PDF "
                    "from images, or split a PDF back into images.",
                    icon=ICON_CONVERT,
                ))
                method = gr.Dropdown(
                    _CONVERT_METHODS, value=_CONVERT_METHODS[0],
                    label="What do you want to do?",
                    info="Switch between converting an image's format, building a "
                    "PDF from images, or splitting a PDF back into images.",
                )

                # -- Method A: change image format --
                with gr.Column(visible=True) as grp_format:
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            conv_in = gr.Image(
                                label="Image", type="pil",
                                sources=["upload", "clipboard"], height=300,
                                elem_classes="drop", buttons=["download", "fullscreen"],
                            )
                            conv_fmt = gr.Dropdown(
                                list(FORMATS), value="PNG", label="Convert to",
                                info="PNG / TIFF keep full quality · JPEG, WebP, "
                                "AVIF, HEIC are smaller but lossy.",
                            )
                            conv_quality = gr.Slider(
                                1, 100, value=90, step=1, label="Quality (lossy)",
                                info="Only affects lossy formats. Higher = better "
                                "looking but larger file.",
                            )
                            conv_lossless = gr.Checkbox(
                                value=False, label="Lossless WebP",
                                info="Encode WebP with no quality loss (bigger file).",
                            )
                            conv_btn = gr.Button("Convert", variant="primary", size="lg")
                        with gr.Column(scale=1):
                            conv_file = gr.File(label="Download converted file")
                            conv_info = gr.Markdown()

                # -- Method B: images -> PDF --
                with gr.Column(visible=False) as grp_topdf:
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            pdf_imgs_in = gr.File(
                                label="Images (multiple = multi-page, in order)",
                                file_count="multiple", file_types=["image"],
                                elem_classes="drop",
                            )
                            pdf_build_btn = gr.Button(
                                "Build PDF", variant="primary", size="lg"
                            )
                        with gr.Column(scale=1):
                            pdf_build_out = gr.File(label="Download PDF")
                            pdf_build_info = gr.Markdown()

                # -- Method C: PDF -> images --
                with gr.Column(visible=False) as grp_frompdf:
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            pdf_in = gr.File(
                                label="PDF", file_count="single",
                                file_types=[".pdf"], elem_classes="drop",
                            )
                            pdf_dpi = gr.Slider(
                                72, 300, value=150, step=1, label="Render DPI",
                                info="Higher = sharper, larger PNGs. 150 is a good "
                                "default; 300 for print quality.",
                            )
                            pdf_extract_btn = gr.Button(
                                "Extract pages", variant="primary", size="lg"
                            )
                        with gr.Column(scale=1):
                            pdf_extract_out = gr.File(label="Download pages (ZIP)")
                            pdf_gallery = gr.Gallery(
                            label="Pages", columns=4, height=220,
                            buttons=["download", "fullscreen"],
                        )
                            pdf_extract_info = gr.Markdown()

                method.change(
                    _switch_method, method, [grp_format, grp_topdf, grp_frompdf]
                )

            # ---- Tab 4: Remove background ----
            with gr.Tab("Remove BG"):
                gr.HTML(_section_head(
                    "Cut-out", "Remove Background",
                    "Cut the subject out of a photo with U²-Net and save a "
                    "transparent PNG. Pairs with the Lian Li tab — drop the "
                    "cut-out in as a sticker.",
                    icon=ICON_AI,
                ))
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        bg_in = gr.Image(
                            label="Input", type="pil",
                            sources=["upload", "clipboard"], height=300,
                            elem_classes="drop", buttons=["download", "fullscreen"],
                        )
                        bg_model = gr.Dropdown(
                            _BG_CHOICES, value=background.DEFAULT_BG_MODEL,
                            label="Model",
                            info="u2net = best all-rounder · u2netp = lighter/faster.",
                        )
                        bg_feather = gr.Slider(
                            0, 10, value=1, step=1, label="Edge feather (px)",
                            info="Softens the cut-out edge slightly. 0 = hard edge.",
                        )
                        with gr.Row():
                            bg_btn = gr.Button("Remove background", variant="primary",
                                               size="lg", scale=3)
                            bg_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1):
                        bg_preview = gr.Image(
                            label="Cut-out (checkerboard = transparent)",
                            height=300, buttons=["fullscreen"],
                        )
                        bg_file = gr.File(label="Download transparent PNG")
                        bg_info = gr.Markdown()

            # ---- Tab 5: Lian Li 8.8" Screen builder ----
            with gr.Tab("Lian Li Screen"):
                gr.HTML(_section_head(
                    "Panel", "Lian Li 8.8″ Screen",
                    "Build media sized exactly for the Lian Li 8.8″ panel "
                    "(1920×480 / 480×1920, 4:1) so L-Connect 3 never resamples it. "
                    "Fit any source into the 4:1 frame — the dim area is what gets "
                    "cropped. Export PNG/JPG, looping GIF, or H.264 MP4.",
                    icon=ICON_PANEL,
                ))
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        pn_media = gr.File(
                            label="Image, GIF or video",
                            file_count="single",
                            file_types=["image", ".gif", ".mp4", ".mov", ".webm",
                                        ".mkv", ".m4v", ".avi"],
                            elem_classes="drop",
                        )
                        with gr.Accordion("Source quality — AI upscale (optional)", open=False):
                            gr.Markdown(
                                "Run the source through Real-ESRGAN **before** "
                                "fitting — best for low-res images/clips the "
                                "panel would otherwise show soft. Replaces the "
                                "working source with the enhanced version."
                            )
                            pn_up_model = gr.Dropdown(
                                _MODEL_CHOICES, value="realesrgan-x2plus",
                                label="Upscale model",
                                info="×2 is plenty when fitting into 1920×480 · "
                                "×4 for very small sources · anime for line art.",
                            )
                            pn_enhance = gr.Button("✨ Enhance source", variant="secondary")
                        pn_orient = gr.Radio(
                            list(panel.ORIENTATIONS), value="Landscape · 1920×480",
                            label="Orientation",
                        )
                        pn_fit = gr.Radio(
                            panel.FITS, value="cover", label="Fit",
                            info="cover fills & crops · contain letterboxes · "
                            "stretch distorts · manual = free zoom.",
                        )
                        with gr.Row():
                            pn_offx = gr.Slider(
                                -100, 100, value=0, step=1, label="Pan X (%)",
                                info="Which band survives the crop (cover/manual).",
                            )
                            pn_offy = gr.Slider(
                                -100, 100, value=0, step=1, label="Pan Y (%)",
                            )
                        pn_zoom = gr.Slider(
                            0.1, 5, value=1, step=0.01, label="Zoom (manual fit)",
                        )
                        with gr.Accordion("Background (fills letterbox gaps)", open=False):
                            pn_bgtype = gr.Radio(
                                ["solid", "gradient"], value="solid",
                                label="Type",
                            )
                            with gr.Row():
                                pn_bgcol = gr.ColorPicker(value="#000000", label="Color / Stop A")
                                pn_bgcol2 = gr.ColorPicker(value="#333333", label="Stop B")
                            pn_bgang = gr.Slider(0, 360, value=90, step=1, label="Gradient angle")
                        # Overlays: up to N_TEXT text layers + N_STICKER stickers.
                        # Each slot's components are collected (in field order) so
                        # the preview/export handlers can rebuild the overlay list.
                        _text_slots = []
                        with gr.Accordion("Text overlays", open=True):
                            for _t in range(N_TEXT):
                                with gr.Accordion(f"Text {_t + 1}", open=(_t == 0)):
                                    t_en = gr.Checkbox(value=(_t == 0), label="Show this text")
                                    t_content = gr.Textbox(label="Text", lines=2,
                                                           placeholder="(your text)")
                                    with gr.Row():
                                        t_font = gr.Dropdown(panel.FONT_NAMES,
                                                             value=panel.DEFAULT_FONT,
                                                             label="Font", filterable=True)
                                        t_size = gr.Slider(16, 900, value=180, step=2,
                                                           label="Size (px)")
                                    with gr.Row():
                                        t_color = gr.ColorPicker(value="#ffffff", label="Color")
                                        t_align = gr.Radio(["left", "center", "right"],
                                                           value="center", label="Align")
                                    with gr.Row():
                                        t_x = gr.Slider(-100, 100, value=0, step=1, label="X (%)")
                                        t_y = gr.Slider(-100, 100, value=0, step=1, label="Y (%)")
                                    t_rot = gr.Slider(-180, 180, value=0, step=1,
                                                      label="Rotation (°)")
                                    with gr.Row():
                                        t_stroke = gr.ColorPicker(value="#000000", label="Stroke")
                                        t_strokew = gr.Slider(0, 40, value=0, step=1,
                                                              label="Stroke width")
                                _text_slots.append([t_en, t_content, t_font, t_size, t_color,
                                                    t_align, t_x, t_y, t_rot, t_stroke, t_strokew])
                        _sticker_slots = []
                        with gr.Accordion("Stickers (image overlays)", open=False):
                            for _s in range(N_STICKER):
                                with gr.Accordion(f"Sticker {_s + 1}", open=False):
                                    s_en = gr.Checkbox(value=False, label="Show this sticker")
                                    # image_mode="RGBA" preserves transparency —
                                    # without it Gradio drops alpha and PNG cut-outs
                                    # composite as opaque black.
                                    s_img = gr.Image(label="Sticker image (PNG with "
                                                     "transparency works best)", type="pil",
                                                     image_mode="RGBA",
                                                     sources=["upload", "clipboard"], height=120)
                                    with gr.Row():
                                        s_scale = gr.Slider(2, 100, value=40, step=1,
                                                            label="Size (% of panel height)")
                                        s_op = gr.Slider(0, 1, value=1, step=0.01, label="Opacity")
                                    with gr.Row():
                                        s_x = gr.Slider(-100, 100, value=0, step=1, label="X (%)")
                                        s_y = gr.Slider(-100, 100, value=0, step=1, label="Y (%)")
                                    s_rot = gr.Slider(-180, 180, value=0, step=1,
                                                      label="Rotation (°)")
                                _sticker_slots.append([s_en, s_img, s_scale, s_x, s_y, s_rot, s_op])
                        _overlay_inputs = [c for slot in _text_slots for c in slot] + \
                                          [c for slot in _sticker_slots for c in slot]
                        with gr.Group(visible=False) as pn_anim_group:
                            gr.Markdown("**Animation** — for GIF / MP4 export.")
                            with gr.Row():
                                pn_start = gr.Number(value=0, label="Trim start (s)", minimum=0)
                                pn_end = gr.Number(value=0, label="Trim end (s)", minimum=0)
                            with gr.Row():
                                pn_fps = gr.Dropdown(
                                    ["10", "12", "15", "24", "25", "30", "48", "50", "60"],
                                    value="30", label="FPS (≤ 60)",
                                )
                                pn_loop = gr.Checkbox(value=True, label="Loop")
                            pn_colors = gr.Slider(
                                2, 256, value=128, step=1, label="GIF colors",
                            )
                            pn_loopmode = gr.Radio(
                                panel.LOOP_STYLES, value="normal", label="Loop style",
                                info="boomerang plays forward then back · crossfade "
                                "blends the end into the start — both remove the loop seam.",
                            )
                        pn_fmt = gr.Radio(
                            ["PNG", "JPG", "GIF", "MP4"], value="GIF",
                            label="Export format",
                        )
                        pn_outdir = gr.Textbox(
                            label="Save a copy to folder (optional)",
                            placeholder="/path/to/your L-Connect media folder",
                            info="A timestamped copy is written here on export, in "
                            "addition to the download below.",
                        )
                        with gr.Row():
                            pn_export = gr.Button("Export", variant="primary", size="lg", scale=3)
                            pn_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1):
                        pn_preview = gr.Image(
                            label="Preview — bright = kept, dim = cropped out",
                            height=300, buttons=["fullscreen"],
                        )
                        pn_file = gr.File(label="Download export")
                        pn_info = gr.Markdown()

        conv_btn.click(
            convert_image,
            [conv_in, conv_fmt, conv_quality, conv_lossless],
            [conv_file, conv_info],
        )
        pdf_build_btn.click(build_pdf, [pdf_imgs_in], [pdf_build_out, pdf_build_info])
        bg_btn.click(
            remove_bg_ui, [bg_in, bg_model, bg_feather],
            [bg_preview, bg_file, bg_info], show_progress_on=[bg_preview],
        )
        bg_clear.click(lambda: (None, None, None, None), None,
                       [bg_in, bg_preview, bg_file, bg_info])
        pdf_extract_btn.click(
            extract_pdf, [pdf_in, pdf_dpi], [pdf_extract_out, pdf_gallery, pdf_extract_info]
        )
        run.click(
            enhance,
            [inp, model, device, deblur, deblur_model, sharpen, tile, onnx, out_size],
            [out, info],
            show_progress_on=[out],
        )
        restore_btn.click(
            restore_only,
            [inp, deblur_model, sharpen, device, onnx],
            [out, info],
            show_progress_on=[out],
        )
        clear.click(lambda: (None, None, None), None, [inp, out, info])
        vid_btn.click(
            upscale_video_ui,
            [vid_in, vid_model, vid_size, vid_sharpen, vid_smooth, vid_start,
             vid_end, vid_device, vid_tile],
            [vid_out, vid_compare, vid_info],
            show_progress_on=[vid_out],
        )
        vid_clear.click(
            lambda: (None, None, None, None), None,
            [vid_in, vid_out, vid_compare, vid_info],
        )
        vid_in.change(_on_video_change, vid_in, [vid_start, vid_end])

        # ---- Lian Li panel builder wiring ----
        # Controls that affect the static composite → live preview. Order must
        # match _panel_params: base controls, then the flat overlay-slot values.
        _pn_preview_inputs = [
            pn_media, pn_orient, pn_fit, pn_zoom, pn_offx, pn_offy, pn_bgtype,
            pn_bgcol, pn_bgcol2, pn_bgang,
        ] + _overlay_inputs
        for _c in _pn_preview_inputs:
            # show_progress="hidden" removes the loading flash on every slider
            # tick; the source frame is cached and the preview composites at
            # display resolution, so the re-render is ~25 ms.
            _c.change(
                panel_preview_ui, _pn_preview_inputs, pn_preview,
                show_progress="hidden",
            )
        pn_media.change(
            panel_on_media, pn_media, [pn_end, pn_anim_group, pn_info]
        )
        pn_enhance.click(
            panel_enhance_source,
            [pn_media, pn_up_model] + _pn_preview_inputs[1:],
            [pn_media, pn_preview, pn_info],
            show_progress_on=[pn_preview],
        )
        pn_export.click(
            panel_export_ui,
            _pn_preview_inputs + [pn_fmt, pn_fps, pn_loop, pn_colors, pn_start,
                                  pn_end, pn_loopmode, pn_outdir],
            [pn_file, pn_info],
            show_progress_on=[pn_file],
        )
        pn_clear.click(
            lambda: (None, None, None), None, [pn_media, pn_preview, pn_file]
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("UPSCALER_PORT", "7860")),
        theme=THEME,
        css=_CSS,
        js=_APPLY_THEME_JS,
    )
