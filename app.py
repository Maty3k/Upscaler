"""Local drag-and-drop GUI for Upscaler. Run: `python app.py` then open the URL.

Everything runs on your machine — Gradio just serves a local web UI. No data
leaves your computer.

Two tools on one page:
  • File Converter — fast, lossless-where-possible format conversion (no models).
  • Upscale & Enhance — Real-ESRGAN upscaling, optional NAFNet deblur + sharpen.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

# Windows: a redirected stdout/stderr defaults to cp1252, and libraries print
# emoji status lines (torch's ONNX exporter, tqdm) — never let a glyph that
# cp1252 can't encode crash a running job.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from upscaler import (adjust, background, blur, config, effects, face, library, manage,
                      panel, steam)
from upscaler.convert import FORMATS, convert, extension_for
from upscaler.document import images_to_pdf, pdf_to_images
from upscaler.deblur import Deblurrer, DeblurTooLargeError
from upscaler.engine import (CancelledError, OutputTooLargeError, Upscaler,
                             resolve_device)
from upscaler import depth as depth_tools
from upscaler import fit, frame as frame_tools, metadata as md_tools
from upscaler import optimize as opt_tools
from upscaler import recipe as recipe_tools
from upscaler import design as dz_tools
from upscaler import screenshot as shot_tools
from upscaler import watermark as wm_tools
from upscaler.models.registry import (
    COLORIZE_MODELS,
    DEBLUR_MODELS,
    DEFAULT_COLORIZE_MODEL,
    DEFAULT_FACE_MODEL,
    DEFAULT_INPAINT_MODEL,
    FACE_MODELS,
    INPAINT_MODELS,
    MODELS,
)
from upscaler import sharpen as sharpen_tools
from upscaler.sharpen import unsharp_mask

# Cache loaded models so switching images doesn't reload weights every run.
# Keyed by (model, device, onnx) so torch and ONNX engines are cached separately.
_UP_CACHE: dict[tuple, object] = {}
_DB_CACHE: dict[tuple, object] = {}
_FACE_CACHE: dict[tuple, object] = {}
_FBCNN_CACHE: dict[tuple, object] = {}
_COLOR_CACHE: dict[tuple, object] = {}
_INPAINT_CACHE: dict[tuple, object] = {}

# Cooperative cancel flags for long jobs. Gradio's `cancels=` only stops the
# event stream — the compute would keep running to completion server-side, so
# the Cancel buttons also set these and the tile/file loops poll them.
_ENHANCE_CANCEL = threading.Event()
_BATCH_CANCEL = threading.Event()
_VIDEO_CANCEL = threading.Event()
_STEAM_CANCEL = threading.Event()

# One-shot download files (converted images, ZIPs, video/panel exports) go in
# this dedicated dir instead of loose in the OS temp dir, and anything older
# than a day is purged at startup — otherwise every export leaks a file that
# lives until the OS cleans the temp dir.
_EXPORT_DIR = Path(tempfile.gettempdir()) / "upscaler-exports"


def _ensure_export_dir() -> str:
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return str(_EXPORT_DIR)


def _purge_old_exports(max_age_hours: int = 24) -> None:
    cutoff = time.time() - max_age_hours * 3600
    try:
        for p in _EXPORT_DIR.glob("*"):
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink()
            except OSError:
                continue
    except OSError:
        pass


def _fmt_mmss(sec: float) -> str:
    """0:07 · 3:24 · 1:02:09 — humans read minutes, not '204 seconds'."""
    sec = max(0, int(round(sec)))
    m, s = divmod(sec, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class _JobProgress:
    """A (done, total) callback that renders a rich progress line on a
    gr.Progress bar: percent, m:ss elapsed, and an m:ss ETA once there's
    enough signal to extrapolate from."""

    def __init__(self, progress, label: str):
        self._p = progress
        self._label = label
        self._t0 = time.perf_counter()

    def __call__(self, done: int, total: int) -> None:
        total = max(int(total), 1)
        frac = min(max(done / total, 0.0), 1.0)
        elapsed = time.perf_counter() - self._t0
        desc = f"{self._label} {done}/{total} · {frac:.0%} · {_fmt_mmss(elapsed)} elapsed"
        if 0 < done < total:
            eta = elapsed / done * (total - done)
            desc += f" · ~{_fmt_mmss(eta)} left"
        self._p(frac, desc=desc)


def _torch_dev_key(device: str) -> str:
    """Cache key for a torch engine: the *resolved* device, so 'auto' and the
    device it resolves to share one cached engine instead of loading the same
    weights twice."""
    return resolve_device(device).type


def _onnx_engines():
    try:
        from upscaler.onnx_engine import OnnxDeblurrer, OnnxUpscaler
    except ImportError as e:
        raise gr.Error(
            "The ONNX engine needs extra packages. Install them with: "
            'pip install -e ".[onnx]"'
        ) from e
    return OnnxUpscaler, OnnxDeblurrer


def _get_upscaler(model: str, device: str, tile: int, onnx: bool):
    # ONNX picks its execution provider from the raw string ('auto' may mean
    # CUDA even when torch is CPU-only), so only torch engines key on the
    # resolved device.
    key = (model, device if onnx else _torch_dev_key(device), onnx)
    up = _UP_CACHE.get(key)
    if up is None:
        if onnx:
            OnnxUpscaler, _ = _onnx_engines()
            up = OnnxUpscaler(model=model, device=device, tile=tile)
        else:
            up = Upscaler(model=model, device=device, tile=tile)
        _UP_CACHE[key] = up
    elif getattr(up, "tile", None) != tile:
        up.tile = tile  # both engines read .tile per run — no weight reload
    return up


def _get_deblurrer(model: str, device: str, onnx: bool):
    key = (model, device if onnx else _torch_dev_key(device), onnx)
    db = _DB_CACHE.get(key)
    if db is None:
        if onnx:
            _, OnnxDeblurrer = _onnx_engines()
            db = OnnxDeblurrer(model=model, device=device)
        else:
            db = Deblurrer(model=model, device=device)
        _DB_CACHE[key] = db
    return db


def _get_face_restorer(model: str, device: str):
    key = (model, _torch_dev_key(device))  # keyed by model too, so switching models isn't stale
    fr = _FACE_CACHE.get(key)
    if fr is None:
        from upscaler.face import FaceRestorer  # optional dep, imported lazily
        fr = FaceRestorer(model=model, device=device)
        _FACE_CACHE[key] = fr
    return fr


def _get_fbcnn(device: str):
    key = (_torch_dev_key(device),)
    fr = _FBCNN_CACHE.get(key)
    if fr is None:
        from upscaler.restore import ArtifactRemover  # optional dep, lazy import
        fr = ArtifactRemover(device=device)
        _FBCNN_CACHE[key] = fr
    return fr


def _get_colorizer(model: str, device: str):
    key = (model, device)
    c = _COLOR_CACHE.get(key)
    if c is None:
        from upscaler.colorize import Colorizer  # optional dep, lazy import
        c = Colorizer(model=model, device=device)
        _COLOR_CACHE[key] = c
    return c


def _get_inpainter(model: str, device: str):
    key = (model, device)
    c = _INPAINT_CACHE.get(key)
    if c is None:
        from upscaler.inpaint import Inpainter  # lazy import
        c = Inpainter(model=model, device=device)
        _INPAINT_CACHE[key] = c
    return c


# -- File converter ----------------------------------------------------------

def convert_image(image, fmt, quality, lossless):
    if image is None:
        raise gr.Error("Upload an image to convert first.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    try:
        data = convert(img, fmt, quality=int(quality), lossless=bool(lossless))
    except (ValueError, OSError, KeyError) as e:
        raise gr.Error(f"Couldn't convert to {fmt}: {e}") from e

    fd, path = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=f".{extension_for(fmt)}")
    with os.fdopen(fd, "wb") as f:
        f.write(data)

    kb = len(data) / 1024
    if fmt == "GIF":
        note = "256-color palette"  # GIF quantizes; calling it lossless would mislead
    elif (lossless and fmt == "WebP") or not FORMATS[fmt][2]:
        note = "lossless"
    else:
        note = f"q{int(quality)}"
    library.save_path(path, "convert")  # auto-add to the Library
    return path, f"✅ Converted to **{fmt}** ({note}) · {kb:,.1f} KB · {img.width}×{img.height}px"


# -- Image <-> PDF -----------------------------------------------------------

def build_pdf(files):
    if not files:
        raise gr.Error("Add at least one image to build a PDF.")
    images = [Image.open(f) for f in files]
    data = images_to_pdf(images)
    fd, path = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    library.save_path(path, "pdf")  # auto-add to the Library
    return path, f"✅ {len(images)} image(s) → PDF · {len(data) / 1024:,.1f} KB"


def extract_pdf(pdf_file, dpi):
    if pdf_file is None:
        raise gr.Error("Upload a PDF first.")
    try:
        pages = pdf_to_images(pdf_file, dpi=int(dpi))
    except ImportError as e:
        raise gr.Error(str(e)) from e
    except Exception as e:  # pdfium raises its own error types
        raise gr.Error(
            "Couldn't read that PDF — it may be corrupt or password-protected. "
            f"({e})"
        ) from e

    fd, zpath = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".zip")
    with os.fdopen(fd, "wb") as fh, zipfile.ZipFile(fh, "w") as z:
        for i, im in enumerate(pages, 1):
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            z.writestr(f"page_{i:03d}.png", buf.getvalue())
    library.save_path(zpath, "pdf-pages")  # auto-add to the Library
    return zpath, pages, f"✅ {len(pages)} page(s) → PNG · {int(dpi)} dpi (zip)"


# -- Background removal -------------------------------------------------------

_BG_CHOICES = [(f"{s.name} — {s.notes}", s.name) for s in background.BG_MODELS.values()]


def remove_bg_ui(image, model, feather, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload an image to remove its background.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    progress(0.2, desc="Loading model…")
    try:
        cut = background.remove_background(img.convert("RGB"), model=model,
                                           feather=int(feather))
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise gr.Error(
            "Couldn't remove the background. The model downloads on first use, so "
            "check your internet connection and try again."
        ) from e
    progress(0.9, desc="Saving transparent PNG…")
    fd, path = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".png")
    os.close(fd)
    cut.save(path, "PNG")  # PNG keeps the alpha channel
    library.save_path(path, "removebg")  # auto-add to the Library
    preview = background.on_checkerboard(cut)
    return preview, path, (
        f"✅ Background removed — {cut.width}×{cut.height}px transparent PNG. "
        "Drop it into the Lian Li tab as a sticker."
    )


# -- Colorize (DDColor) ------------------------------------------------------

_COLORIZE_CHOICES = [(f"{s.name} — {s.notes}", s.name) for s in COLORIZE_MODELS.values()]


def colorize_ui(image, model, strength, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload a photo to colorize.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    progress(0.2, desc="Loading model…")
    try:
        result = _get_colorizer(model, "auto").colorize(img, float(strength))
    except RuntimeError as e:  # missing [face] deps → friendly install message
        raise gr.Error(str(e)) from e
    except (OSError, ValueError, AssertionError) as e:
        raise gr.Error(
            "Couldn't colorize. The model downloads on first use (~870MB), so "
            "check your connection and try again."
        ) from e
    progress(0.9, desc="Saving…")
    library.save_image(result, "colorize")  # auto-add to the Library
    return (img.convert("RGB"), result), (
        f"✅ Colorized — {result.width}×{result.height}px."
    )


# -- Inpaint / object removal (LaMa) -----------------------------------------

_INPAINT_CHOICES = [(f"{s.name} — {s.notes}", s.name) for s in INPAINT_MODELS.values()]


def _mask_from_editor(value):
    """Return (background RGB, mask L) from a gr.ImageEditor value. The mask is
    the union of the painted layers' alpha (anything the user drew). Returns
    (bg, None) if nothing was painted, or (None, None) if there's no image."""
    if not value:
        return None, None
    bg = value.get("background")
    if bg is None:
        return None, None
    bg = bg.convert("RGB")
    w, h = bg.size
    acc = np.zeros((h, w), dtype=np.uint8)
    for layer in value.get("layers") or []:
        if layer is None:
            continue
        alpha = np.asarray(layer.convert("RGBA").resize((w, h)))[..., 3]
        acc = np.maximum(acc, (alpha > 0).astype(np.uint8) * 255)
    if acc.max() == 0:
        return bg, None
    return bg, Image.fromarray(acc, "L")


def inpaint_ui(editor_value, model, progress=gr.Progress()):
    bg, mask = _mask_from_editor(editor_value)
    if bg is None:
        raise gr.Error("Upload an image and paint over the object to remove.")
    if mask is None:
        raise gr.Error("Paint over the object you want to remove first (use the brush).")
    progress(0.2, desc="Loading model…")
    try:
        result = _get_inpainter(model, "auto").inpaint(bg, mask)
    except (RuntimeError, OSError, ValueError, AssertionError) as e:
        raise gr.Error(
            "Couldn't remove the object. The model downloads on first use (~196MB), "
            "so check your connection and try again."
        ) from e
    progress(0.9, desc="Saving…")
    library.save_image(result, "inpaint")  # auto-add to the Library
    return (bg, result), f"✅ Object removed — {result.width}×{result.height}px."


# -- Upscale & enhance -------------------------------------------------------

def _structural_ok(a_img, b_img) -> bool:
    """True if b preserves a's structure (luma correlation). A real deblur/
    denoise keeps the image; a failed one (e.g. GoPro motion-deblur on a grainy
    photo) returns garbage that doesn't correlate with the input at all."""
    a = np.asarray(a_img.convert("RGB"), dtype=np.float32)
    b = np.asarray(b_img.convert("RGB"), dtype=np.float32)
    la = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).ravel()
    lb = (0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]).ravel()
    la -= la.mean()
    lb -= lb.mean()
    denom = float(np.linalg.norm(la) * np.linalg.norm(lb))
    if denom < 1e-6:
        return True  # flat image — can't tell, assume fine
    return float(la @ lb) / denom > 0.5


def _restore(src_img, deblur_model, device, onnx, strength):
    """Run a NAFNet restore (deblur/denoise) and blend it back over the source
    by `strength` (1 = full effect, lower keeps more original detail/noise).

    Returns (image, ok). If the model produced garbage (doesn't resemble the
    input), the original is returned with ok=False so callers can skip it and
    warn instead of feeding garbage downstream. Backend-agnostic blend works for
    torch and ONNX deblurrers.
    """
    rgb = src_img.convert("RGB")
    out = _get_deblurrer(deblur_model, device, onnx).deblur(rgb)
    if not _structural_ok(rgb, out):
        return rgb, False
    strength = max(0.0, min(1.0, float(strength)))
    blended = out if strength >= 0.999 else Image.blend(rgb, out, strength)
    return blended, True


def _restore_fbcnn(src_img, device):
    """Remove JPEG artifacts (FBCNN). Returns (image, ok); ok=False if the model
    produced something that doesn't resemble the input (same garbage-guard as
    _restore), so callers can skip it and warn instead."""
    rgb = src_img.convert("RGB")
    out = _get_fbcnn(device).restore(rgb)
    if not _structural_ok(rgb, out):
        return rgb, False
    return out, True


def enhance(image, model, device, deblur, deblur_model, restore_strength, sharpen,
            tile, onnx, out_size, face=False, face_strength=1.0,
            face_model=DEFAULT_FACE_MODEL, face_fidelity=0.5, fbcnn=False,
            custom_size="", crop_position=50.0, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload an image to enhance first.")
    _ENHANCE_CANCEL.clear()
    original = image if isinstance(image, Image.Image) else Image.fromarray(image)
    src = original
    stages = []
    # Crop to the target's aspect ratio before anything else runs: it's the one
    # step that removes pixels, and every pixel kept here is cleaned up and
    # enlarged at full cost. See upscaler/fit.py for the tradeoff.
    exact = _exact_target(out_size, custom_size)
    if exact:
        # The slider speaks percent (whole numbers read better in the stages
        # trail); fit speaks 0–1. None can arrive from a stale UI state — treat
        # it as the centred default rather than crash the job.
        pos = 50.0 if crop_position is None else float(crop_position)
        cropped = fit.crop(src, *exact, position=pos / 100)
        if cropped.size != src.size:
            stages.append(f"crop to {exact[0]}:{exact[1]} @ {pos:g}%")
        src = cropped
    compare_base = src  # what the result is actually derived from
    if fbcnn:  # de-block JPEGs first, before any denoise/deblur
        try:
            src, ok = _restore_fbcnn(src, device)
        except RuntimeError as e:  # missing [face] deps → friendly install message
            raise gr.Error(str(e)) from e
        except (OSError, AssertionError, ValueError) as e:
            raise gr.Error(
                "Couldn't remove JPEG artifacts. The model downloads on first use, "
                "so check your connection and try again."
            ) from e
        stages.append(
            "remove JPEG artifacts (FBCNN)" if ok
            else "⚠ JPEG-artifact removal skipped — didn't suit this image"
        )
    try:
        if deblur:
            src, ok = _restore(src, deblur_model, device, onnx, restore_strength)
            if ok:
                pct = "" if restore_strength >= 0.999 else f" @{int(round(restore_strength * 100))}%"
                stages.append(f"clean up `{deblur_model}`{pct}")
            else:
                stages.append(f"⚠ clean-up skipped — `{deblur_model}` didn't suit this image")
        up = _get_upscaler(model, device, int(tile), onnx)

        # Both engines share the signature (per-tile progress + cooperative cancel).
        result = up.upscale(src, progress_cb=_JobProgress(progress, "Upscaling tile"),
                            should_cancel=_ENHANCE_CANCEL.is_set)
    except CancelledError:
        raise gr.Error("Cancelled — nothing was saved.") from None
    except (DeblurTooLargeError, OutputTooLargeError) as e:
        raise gr.Error(str(e)) from e  # already worded for the user
    except (RuntimeError, AssertionError, OSError, ValueError) as e:
        if "out of memory" in str(e).lower():
            # Blaming the device / the download here sends people hunting in the
            # wrong place — the image is simply too big for the GPU.
            raise gr.Error(
                "Ran out of GPU memory on this image. Try a smaller Tile size, "
                "turn off Clean up, or scale the image down before enhancing."
            ) from e
        raise gr.Error(
            "Couldn't run the enhancement. If you set a specific Device "
            "(cuda / mps) your machine may not support it — try \"auto\". Models "
            "also download on first use, so check your connection."
        ) from e
    stages.append(f"upscale ×{up.scale}")
    if face:
        try:
            result = _get_face_restorer(face_model, device).restore(
                result, face_strength, fidelity=face_fidelity
            )
        except RuntimeError as e:  # missing [face] deps → friendly install message
            raise gr.Error(str(e)) from e
        except (OSError, AssertionError, ValueError) as e:
            raise gr.Error(
                "Couldn't restore faces. The model downloads on first use, so "
                "check your connection and try again."
            ) from e
        fpct = "" if face_strength >= 0.999 else f" @{int(round(face_strength * 100))}%"
        stages.append(f"faces ({face_model}){fpct}")
    if sharpen > 0:
        result = unsharp_mask(result, strength=float(sharpen))
        stages.append(f"sharpen {sharpen:g}")

    # Exact resolution wins over the longest-edge presets: the source was
    # already cropped to this ratio, so this only resamples to the pixel count.
    if exact:
        if result.size != exact:
            result = fit.resize_exact(result, *exact)
        stages.append(f"fit {exact[0]}×{exact[1]}")
        target = None
    else:
        target = _SIZE_PRESETS.get(out_size)
    if target:
        w, h = result.size
        longest = max(w, h)
        if longest != target:
            r = target / longest
            result = result.resize(
                (max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS
            )
            stages.append(f"fit {target}px")

    if onnx:
        prov = getattr(up, "provider", "")
        backend = "onnx · GPU (DirectML)" if prov.startswith("Dml") else "onnx · CPU"
    else:
        backend = getattr(up, "device", None) and up.device.type
    info = (
        "✅ " + " → ".join(stages)
        + f" · backend `{backend}` · {result.width}×{result.height}px"
    )
    library.save_image(result, "upscale")  # auto-add to the Library
    # (before, after) for the comparison slider. A crop changes the shape, so
    # the untouched original would slide against the result misaligned — show
    # the cropped region instead, scaled to match.
    before = compare_base if compare_base.size == result.size else \
        compare_base.resize(result.size, Image.LANCZOS)
    return (before, result), info


def restore_only(image, deblur_model, restore_strength, sharpen, device, onnx,
                 fbcnn=False):
    """Run just the clean-up passes (FBCNN de-block and/or NAFNet deblur/denoise)
    — no upscaling."""
    if image is None:
        raise gr.Error("Upload an image to clean up first.")
    original = image if isinstance(image, Image.Image) else Image.fromarray(image)
    src = original
    stages = []
    if fbcnn:  # de-block JPEGs first
        try:
            src, fok = _restore_fbcnn(src, device)
        except RuntimeError as e:  # missing [face] deps → friendly install message
            raise gr.Error(str(e)) from e
        except (OSError, AssertionError, ValueError) as e:
            raise gr.Error(
                "Couldn't remove JPEG artifacts. The model downloads on first use, "
                "so check your connection and try again."
            ) from e
        stages.append(
            "remove JPEG artifacts (FBCNN)" if fok
            else "⚠ JPEG-artifact removal skipped — didn't suit this image"
        )
    try:
        result, ok = _restore(src, deblur_model, device, onnx, restore_strength)
    except (RuntimeError, AssertionError, OSError, ValueError) as e:
        raise gr.Error(
            "Couldn't run the clean-up. If you set a specific Device (cuda / mps) "
            "your machine may not support it — try \"auto\". Models also download "
            "on first use, so check your connection."
        ) from e
    if ok:
        pct = "" if restore_strength >= 0.999 else f" @{int(round(restore_strength * 100))}%"
        stages.append(f"clean up `{deblur_model}`{pct}")
    elif fbcnn:
        # NAFNet didn't suit the image, but FBCNN already cleaned it — keep that
        # and still tell the user the deblur/denoise pass was skipped.
        result = src
        stages.append(f"⚠ clean-up skipped — `{deblur_model}` didn't suit this image")
    else:
        return (original, original), (
            f"⚠ The `{deblur_model}` clean-up didn't suit this image, so it was "
            "skipped. For a noisy or grainy photo, choose the **SIDD (denoise)** "
            "model — GoPro only fixes genuine motion blur."
        )
    if sharpen > 0:
        result = unsharp_mask(result, strength=float(sharpen))
        stages.append(f"sharpen {sharpen:g}")
    info = (
        "✅ " + " → ".join(stages)
        + f" · {result.width}×{result.height}px (no upscale)"
    )
    library.save_image(result, "restore")  # auto-add to the Library
    return (original, result), info


# -- Video (frame-by-frame) --------------------------------------------------

def _video_duration(path):
    """Clip length in seconds (0 if unknown)."""
    import re
    import shutil
    import subprocess

    if not path:
        return 0
    fp = shutil.which("ffprobe")
    if not fp:
        # Only the bundled imageio-ffmpeg binary (no ffprobe): parse the
        # "Duration: HH:MM:SS.cc" line from the `-i` banner instead of giving
        # up — the trim fields and the compare scrubber both need a length.
        from upscaler.video import _ffmpeg

        try:
            info = subprocess.run(
                [_ffmpeg(), "-hide_banner", "-i", str(path)],
                capture_output=True, text=True,
            )
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info.stderr or "")
            if m:
                h, mnt, s = m.groups()
                return round(int(h) * 3600 + int(mnt) * 60 + float(s), 1)
        except (OSError, RuntimeError):
            pass
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


def _first_frame(video_path, at: float = 0.0):
    """Grab a frame of a video as a PIL image (for the comparison). ``at``
    seeks that many seconds in first — so a trimmed render can be compared
    against the matching source frame, not always the clip's very first."""
    import subprocess

    from upscaler.video import _ffmpeg

    fd, p = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".png")
    os.close(fd)
    seek = ["-ss", str(at)] if at and at > 0 else []
    subprocess.run(
        [_ffmpeg(), "-y", *seek, "-i", str(video_path), "-frames:v", "1", p],
        capture_output=True,
    )
    img = Image.open(p)
    img.load()  # read fully into memory, then drop the temp file (no leak)
    try:
        os.remove(p)
    except OSError:
        pass
    return img


def _compare_pair(src_path, out_path, at_src: float, at_out: float):
    """(before, after) frames at matching timestamps for the comparison slider.
    The before-frame is resized to the after-frame's dimensions so the swipe
    lines up pixel-for-pixel and shows the detail gained, not two differently
    sized images."""
    before = _first_frame(src_path, at=at_src)
    after = _first_frame(out_path, at=at_out)
    if before.size != after.size:
        before = before.resize(after.size, Image.LANCZOS)
    return before, after


def video_compare_at(video_path, out_path, t, trim_start):
    """Scrub the before/after comparison to ``t`` seconds into the render."""
    if not (video_path and out_path):
        return gr.update()
    start = float(trim_start) if trim_start and trim_start > 0 else 0.0
    try:
        return _compare_pair(video_path, out_path, start + float(t), float(t))
    except Exception:  # seeking past the last frame etc. — keep the old pair
        return gr.update()


def upscale_video_ui(video_path, model, out_size, sharpen, smooth, trim_start,
                     trim_end, device, tile, onnx=False, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a video first.")
    _VIDEO_CANCEL.clear()
    from upscaler.video import upscale_video

    fd, out = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".mp4")
    os.close(fd)
    fps = None if smooth in (None, "Off") else int(smooth)
    target = _SIZE_PRESETS.get(out_size)
    start = float(trim_start) if trim_start and trim_start > 0 else None
    end = float(trim_end) if trim_end and trim_end > 0 else None
    # End auto-fills to the (0.1s-rounded) clip length on upload — treat "at or
    # past the end" as no trim, so a default render can't clip tail frames or
    # misreport itself as trimmed.
    dur = _video_duration(video_path)
    if end is not None and dur and end >= dur:
        end = None

    cb = _JobProgress(progress, "Upscaling frame")

    try:
        upscale_video(
            video_path, out, model=model, device=device, tile=int(tile),
            sharpen=float(sharpen), interpolate_fps=fps, target_long_edge=target,
            trim_start=start, trim_end=end, progress_cb=cb,
            should_cancel=_VIDEO_CANCEL.is_set, onnx=bool(onnx),
        )
    except CancelledError:
        try:
            os.remove(out)
        except OSError:
            pass
        raise gr.Error(
            "Cancelled — finished frames are kept, so running the same video "
            "with the same settings again will pick up where it left off."
        ) from None
    except (RuntimeError, FileNotFoundError) as e:
        try:  # don't leave the pre-created output tempfile behind on failure
            os.remove(out)
        except OSError:
            pass
        raise gr.Error(
            "Couldn't process the video. Make sure ffmpeg is installed (e.g. "
            '"brew install ffmpeg" on macOS) and the file is a standard video, '
            "then try again."
        ) from e

    try:
        compare = _compare_pair(video_path, out, start or 0, 0)
    except Exception:
        compare = None
    # arm the compare scrubber over the rendered clip (stop just short of the
    # end — seeking exactly to the last timestamp often yields no frame)
    out_dur = _video_duration(out)
    scrub = (gr.update(visible=True, value=0.0, maximum=max(out_dur - 0.1, 0.1))
             if compare and out_dur > 0.2 else gr.update(visible=False))
    extra = (f" · {target}px" if target else "") + (f" · {fps} fps" if fps else "")
    if start or end:
        extra += f" · trim {start or 0:g}–{end if end else 'end'}s"
    library.save_path(out, "video")  # auto-add to the Library
    return out, compare, f"✅ Done — preview and download below.{extra}", scrub


_CONVERT_METHODS = ["Change image format", "Fit a file-size budget",
                    "Remove metadata (privacy)", "Images → PDF", "PDF → Images"]

# Output-size presets for upscaling: AI-upscale with the model, then fit the
# longest edge to this many pixels (None = leave at the model's native scale).
_EXACT_CUSTOM = "Custom size…"


def _exact_target(out_size, custom_size):
    """Resolve the Output size choice to an exact (w, h), or None.

    None means the longest-edge presets apply instead — the aspect ratio is
    left alone and nothing is cropped.
    """
    if out_size == _EXACT_CUSTOM:
        return fit.parse_target(custom_size or "")
    return fit.TARGET_PRESETS.get(out_size)


_SIZE_PRESETS: dict[str, int | None] = {
    "Model default (×2/×4)": None,
    "HD · 1280px": 1280,
    "Full HD 1080p · 1920px": 1920,
    "QHD 1440p · 2560px": 2560,
    "4K UHD · 3840px": 3840,
    "8K · 7680px": 7680,
}

# The dark-mode accent from _CSS (--ac). The preview dims everything outside
# the crop, so the brighter teal reads on both light and dark screenshots.
_PREVIEW_ACCENT = "#2DD4BF"


def _crop_preview(image, out_size, custom_size, position_pct):
    """Show which part of the image an exact-size crop would keep.

    The whole point is answering "what am I about to lose?" *before* the job
    runs, so this has to be instant: everything happens on a ≤640px thumbnail,
    never the full image, and nothing touches disk. The kept region stays at
    full brightness inside an accent frame; the doomed rest is dimmed rather
    than blacked out, so it's still recognisable while clearly not surviving.
    """
    exact = _exact_target(out_size, custom_size)
    if image is None or exact is None:
        return gr.update(visible=False)
    src = image if isinstance(image, Image.Image) else Image.fromarray(image)
    # convert() always returns a new image, so the caller's original is never
    # touched — thumbnail() below mutates in place.
    thumb = src.convert("RGB")
    thumb.thumbnail((640, 640))
    pos = 50.0 if position_pct is None else float(position_pct)
    box = fit.crop_box_for_aspect(thumb.width, thumb.height, *exact,
                                  position=pos / 100)
    preview = ImageEnhance.Brightness(thumb).enhance(0.35)
    preview.paste(thumb.crop(box), box[:2])
    # Pillow draws the outline inward from the box edge, so the frame never
    # bleeds onto the dimmed region and the bright area stays exactly the crop.
    ImageDraw.Draw(preview).rectangle(
        (box[0], box[1], box[2] - 1, box[3] - 1),
        outline=_PREVIEW_ACCENT, width=1,
    )
    return gr.update(value=preview, visible=True)


def _switch_method(choice):
    """Show only the group for the selected conversion method."""
    return tuple(gr.update(visible=choice == m) for m in _CONVERT_METHODS)


def metadata_inspect_ui(file_obj):
    """Report what an uploaded file is carrying.

    The upload has to be a File, not an Image: Gradio decodes and re-encodes
    an Image on the way in, which would throw the metadata away before we ever
    saw it.
    """
    if not file_obj:
        return None, gr.update(value=""), gr.update(visible=False)
    path = file_obj if isinstance(file_obj, str) else file_obj.name
    try:
        report = md_tools.read(path)
        preview = Image.open(path)
        preview.load()
    except (OSError, ValueError) as e:
        return None, gr.update(value=f"⚠ Couldn't read that file: {e}"), \
            gr.update(visible=False)

    if report.is_clean:
        text = (f"**{report.fmt} {report.size[0]}×{report.size[1]}** — "
                "no metadata found. This file is already clean.")
    else:
        rows = ["| | What | Value |", "|---|---|---|"]
        for f in report.findings:
            rows.append(f"| {'⚠' if f.sensitive else ''} | {f.label} | {f.value} |")
        text = (f"**{report.fmt} {report.size[0]}×{report.size[1]}** — "
                f"{len(report.findings)} item(s), {report.metadata_bytes} bytes of "
                f"metadata.\n\n" + "\n".join(rows))
        if report.gps:
            text += (f"\n\n⚠ **This photo records where it was taken:** "
                     f"{report.gps[0]}, {report.gps[1]}")
        if not report.lossless:
            text += ("\n\n*This format can't be cleaned without re-encoding, so "
                     "removing the metadata will cost a little quality.*")
    return preview, gr.update(value=text), gr.update(visible=True)


def metadata_strip_ui(file_obj, mode, keep_orientation, progress=gr.Progress()):
    """Write a cleaned copy and say exactly what came out of it."""
    if not file_obj:
        raise gr.Error("Upload a photo to check or clean.")
    path = file_obj if isinstance(file_obj, str) else file_obj.name
    progress(0.3, desc="Removing metadata…")
    try:
        res = md_tools.strip(path, mode=mode, keep_orientation=bool(keep_orientation))
    except (OSError, ValueError) as e:
        raise gr.Error(str(e)) from e

    stem = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1] or ".jpg"
    out = os.path.join(_ensure_export_dir(), f"{stem}_clean{ext}")
    with open(out, "wb") as fh:
        fh.write(res.data)
    library.save_path(out, "clean")  # auto-add to the Library

    after = md_tools.read(res.data)
    note = f"✅ {md_tools.describe(res)}"
    if not after.is_clean:
        kept = ", ".join(f.label for f in after.findings)
        note += f"\n\nStill in the file on purpose: {kept}."
    return out, note


def optimize_ui(image, target, fmt, min_quality, allow_resize, max_edge,
                progress=gr.Progress()):
    """Encode the photo down to a file-size budget and report what it cost."""
    if image is None:
        raise gr.Error("Upload an image to fit to a size.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    p = opt_tools.OptimizeParams(
        target=str(target or ""), fmt=fmt, min_quality=int(min_quality),
        allow_resize=bool(allow_resize), max_edge=int(max_edge or 0),
    )
    progress(0.2, desc="Searching for the best quality that fits…")
    try:
        res = opt_tools.optimize(img, p)
    except ValueError as e:
        raise gr.Error(str(e)) from e

    fd, path = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=f".{res.extension}")
    os.close(fd)
    with open(path, "wb") as fh:
        fh.write(res.data)
    library.save_path(path, "optimized")  # auto-add to the Library

    shown = Image.open(io.BytesIO(res.data))
    mark = "✅" if res.fits else "⚠"
    note = (f"{mark} {opt_tools.describe(res)}\n\n"
            f"Budget **{opt_tools.human_size(res.target_bytes)}** · "
            f"result **{opt_tools.human_size(res.nbytes)}**")
    return (img, shown), path, note


# -- Batch processing (one operation over many images) -----------------------

_BATCH_OPS = ["Upscale", "Convert format", "Remove background", "Recipe"]


def _switch_batch_op(op):
    """Show only the settings group for the selected batch operation."""
    return tuple(gr.update(visible=op == name) for name in _BATCH_OPS)


def _resolve_recipe_ui(name, text):
    """The recipe to run: a pasted/edited one wins over the chosen built-in,
    so editing the box is never silently ignored."""
    if (text or "").strip():
        return recipe_tools.from_json(text)
    return recipe_tools.built_in(name)


def recipe_show_ui(name, text):
    """Describe whatever recipe is currently selected, and fill the box when a
    built-in is picked so it can be edited."""
    try:
        chosen = _resolve_recipe_ui(name, text)
    except recipe_tools.RecipeError as e:
        return gr.update(), gr.update(value=f"⚠ {e}")
    return gr.update(), gr.update(value=f"**{chosen.describe()}**")


def recipe_load_built_in(name):
    """Picking a built-in loads its JSON into the box, ready to tweak."""
    try:
        chosen = recipe_tools.built_in(name)
    except recipe_tools.RecipeError as e:
        return gr.update(), gr.update(value=f"⚠ {e}")
    return (gr.update(value=recipe_tools.to_json(chosen)),
            gr.update(value=f"**{chosen.describe()}**"))


def batch_process(files, op, model, out_size, sharpen, fmt, quality,
                  bg_model, feather, device, tile, recipe_name="", recipe_json="",
                  progress=gr.Progress()):
    """Run one operation over many uploaded images. Returns (gallery, zip, info).

    Resilient: a file that can't be read or fails is skipped and counted, so one
    bad image never sinks the whole batch. Every result is also saved to the
    Library.
    """
    if not files:
        raise gr.Error("Add at least one image to process.")
    chosen_recipe = None
    if op == "Recipe":
        try:
            chosen_recipe = _resolve_recipe_ui(recipe_name, recipe_json)
        except recipe_tools.RecipeError as e:
            raise gr.Error(str(e)) from e
    _BATCH_CANCEL.clear()
    work = tempfile.mkdtemp()
    saved: list[str] = []
    gallery: list = []
    failed = 0
    cancelled = False
    seen_stems: dict[str, int] = {}
    n = len(files)
    jp = _JobProgress(progress, f"{op} · image")
    for i, f in enumerate(files):
        if _BATCH_CANCEL.is_set():
            cancelled = True
            break
        jp(i, n)
        try:
            src = Image.open(str(f))
            base = os.path.splitext(os.path.basename(str(f)))[0]
            # Same-named files from different folders must not clobber each
            # other in the work dir / ZIP — suffix repeats: photo, photo_2, …
            count = seen_stems.get(base, 0)
            seen_stems[base] = count + 1
            if count:
                base = f"{base}_{count + 1}"
            if op == "Upscale":
                res = _get_upscaler(model, device, int(tile), False).upscale(
                    src.convert("RGB"), should_cancel=_BATCH_CANCEL.is_set)
                if sharpen > 0:
                    res = unsharp_mask(res, strength=float(sharpen))
                target = _SIZE_PRESETS.get(out_size)
                if target:
                    w, h = res.size
                    longest = max(w, h)
                    if longest != target:
                        r = target / longest
                        res = res.resize(
                            (max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS
                        )
                out = os.path.join(work, f"{base}_upscaled.png")
                res.save(out, "PNG")
                gallery.append(res)
                library.save_image(res, "upscale")
            elif op == "Convert format":
                data = convert(src, fmt, quality=int(quality), lossless=False)
                out = os.path.join(work, f"{base}.{extension_for(fmt)}")
                with open(out, "wb") as fo:
                    fo.write(data)
                gallery.append(src.convert("RGB"))  # AVIF/HEIC may not render; show source
                library.save_path(out, "convert")
            elif op == "Recipe":
                res = recipe_tools.run(src, chosen_recipe)
                out = os.path.join(work, f"{base}_recipe.{res.extension}")
                if res.data:
                    with open(out, "wb") as fo:
                        fo.write(res.data)
                else:
                    res.image.save(out)
                gallery.append(res.image.convert("RGB"))
                library.save_path(out, "recipe")
            else:  # Remove background
                cut = background.remove_background(
                    src.convert("RGB"), model=bg_model, feather=int(feather)
                )
                out = os.path.join(work, f"{base}_cutout.png")
                cut.save(out, "PNG")
                gallery.append(background.on_checkerboard(cut))
                library.save_path(out, "removebg")
            saved.append(out)
        except CancelledError:
            cancelled = True
            break
        except Exception:  # noqa: BLE001 — batch must survive a single bad file
            failed += 1
            continue

    if not saved:
        shutil.rmtree(work, ignore_errors=True)
        if cancelled:
            raise gr.Error("Cancelled before any image finished.")
        raise gr.Error("None of those files could be processed as images.")
    fd, zpath = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".zip")
    with os.fdopen(fd, "wb") as fh, zipfile.ZipFile(fh, "w") as z:
        for sp in saved:
            z.write(sp, arcname=os.path.basename(sp))
    shutil.rmtree(work, ignore_errors=True)  # zipped + in the Library; don't leak
    jp(n, n)
    skipped = f" · {failed} skipped" if failed else ""
    head = "⚠ Cancelled — finished" if cancelled else "✅ Processed"
    return gallery, zpath, (
        f"{head} **{len(saved)}** of {n} image(s) · {op}{skipped}. "
        "Download the ZIP below — results are also saved to your Library."
    )


# -- Lian Li 8.8" panel builder ----------------------------------------------

# Overlay slots: N_TEXT styled text layers + N_STICKER stickers + N_CLOCK clock.
# The slot order here (text, sticker, clock) is the single source of truth for
# the flat Gradio component list — _panel_params, the UI builder, and
# _fan_layout_to_components (layout load) MUST all agree on it.
N_TEXT = 3
N_STICKER = 2
N_CLOCK = 1
# enabled, content, font, size, color, align, x, y, rot, stroke, stroke_w, motion, speed, cps
_TEXT_FIELDS = 14
_STICKER_FIELDS = 7  # enabled, image, scale, x, y, rot, opacity
# enabled, template, font, size, color, align, x, y, rot, stroke, stroke_w
_CLOCK_FIELDS = 11
N_OVERLAY_VALS = N_TEXT * _TEXT_FIELDS + N_STICKER * _STICKER_FIELDS + N_CLOCK * _CLOCK_FIELDS


def _panel_params(orientation, fit, zoom, off_x, off_y, bg_type, bg_color,
                  bg_color2, bg_angle, *ov):
    """Build PanelParams from the base controls plus the flat overlay-slot
    values (text slots, then sticker slots, then the clock slot)."""
    overlays = []
    i = 0
    for _ in range(N_TEXT):
        (en, content, font, size, color, align, x, y, rot, stroke, stroke_w,
         motion, speed, cps) = ov[i:i + _TEXT_FIELDS]
        i += _TEXT_FIELDS
        if en and (content or "").strip():
            overlays.append(dict(
                type="text", content=content, font=font, size=int(size),
                color=color, align=align, x=float(x), y=float(y),
                rotation=float(rot), stroke=stroke, stroke_w=int(stroke_w),
                motion=motion, speed=float(speed), cps=float(cps),
            ))
    for _ in range(N_STICKER):
        en, image, scale, x, y, rot, opacity = ov[i:i + _STICKER_FIELDS]
        i += _STICKER_FIELDS
        if en and image is not None:
            overlays.append(dict(
                type="sticker", image=image, scale=float(scale), x=float(x),
                y=float(y), rotation=float(rot), opacity=float(opacity),
            ))
    for _ in range(N_CLOCK):
        (en, template, font, size, color, align, x, y, rot,
         stroke, stroke_w) = ov[i:i + _CLOCK_FIELDS]
        i += _CLOCK_FIELDS
        if en and (template or "").strip():
            overlays.append(dict(
                type="clock", content=template, font=font, size=int(size),
                color=color, align=align, x=float(x), y=float(y),
                rotation=float(rot), stroke=stroke, stroke_w=int(stroke_w),
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


def panel_mockup_ui(media, *vals):
    """Render a 3D-style product mockup of the composed panel on the screen."""
    if not media:
        raise gr.Error("Upload an image, GIF or video first.")
    return panel.mockup(media, _panel_params(*vals))


def _fan_layout_to_components(p):
    """Inverse of _panel_params: spread a PanelParams back across the flat list
    of [9 base controls] + _overlay_inputs (text slots, sticker slots, clock
    slot). Empties fill with disabled defaults. Order MUST match _panel_params."""
    vals = [p.orientation, p.fit, p.zoom, p.off_x, p.off_y, p.bg_type,
            p.bg_color, p.bg_color2, p.bg_angle]
    texts = [o for o in p.overlays if o.get("type") == "text"]
    stickers = [o for o in p.overlays if o.get("type") == "sticker"]
    clocks = [o for o in p.overlays if o.get("type") == "clock"]
    for j in range(N_TEXT):
        o = texts[j] if j < len(texts) else None
        if o:
            vals += [True, o.get("content", ""), o.get("font", panel.DEFAULT_FONT),
                     o.get("size", 180), o.get("color", "#ffffff"),
                     o.get("align", "center"), o.get("x", 0), o.get("y", 0),
                     o.get("rotation", 0), o.get("stroke", "#000000"),
                     o.get("stroke_w", 0), o.get("motion", "none"),
                     o.get("speed", 120), o.get("cps", 10)]
        else:
            vals += [False, "", panel.DEFAULT_FONT, 180, "#ffffff", "center",
                     0, 0, 0, "#000000", 0, "none", 120, 10]
    for j in range(N_STICKER):
        o = stickers[j] if j < len(stickers) else None
        if o:
            vals += [True, o.get("image"), o.get("scale", 40), o.get("x", 0),
                     o.get("y", 0), o.get("rotation", 0), o.get("opacity", 1.0)]
        else:
            vals += [False, None, 40, 0, 0, 0, 1.0]
    for j in range(N_CLOCK):
        o = clocks[j] if j < len(clocks) else None
        if o:
            vals += [True, o.get("content", "%H:%M:%S"),
                     o.get("font", panel.DEFAULT_FONT), o.get("size", 180),
                     o.get("color", "#ffffff"), o.get("align", "center"),
                     o.get("x", 0), o.get("y", 0), o.get("rotation", 0),
                     o.get("stroke", "#000000"), o.get("stroke_w", 0)]
        else:
            vals += [False, "%H:%M:%S", panel.DEFAULT_FONT, 180, "#ffffff",
                     "center", 0, 0, 0, "#000000", 0]
    return vals


def panel_layout_download(*vals):
    """Serialize the current layout to a shareable .json tempfile for download."""
    from upscaler import panel_presets
    p = _panel_params(*vals)
    fd, path = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".json")
    os.close(fd)
    Path(path).write_text(json.dumps(panel_presets.to_dict(p), indent=2))
    return path


def panel_layout_upload(file):
    """Load a layout .json and spread it back across every panel control."""
    from upscaler import panel_presets
    if not file:
        return _fan_layout_to_components(panel.PanelParams())
    path = file.name if hasattr(file, "name") else file
    try:
        p = panel_presets.load_layout(path)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        raise gr.Error(f"Couldn't read that layout file: {e}") from e
    return _fan_layout_to_components(p)


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
        raise gr.Error("Upload an image, GIF or video first.")
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
            detail = "H.264" if fmt == "mp4" else f"{int(gif_colors)} colors"
            msg = f"✅ {out_fmt} exported — {cw}×{ch}px · {detail}{extra}."
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise gr.Error(
            "Couldn't create the export. GIF and MP4 need ffmpeg installed; "
            "otherwise check your source file and settings, then try again."
        ) from e

    # panel.py writes to the OS temp dir — move the export into the purged
    # exports dir so one-shot files don't accumulate there forever.
    try:
        dest = os.path.join(_ensure_export_dir(), os.path.basename(f))
        shutil.move(f, dest)
        f = dest
    except OSError:
        pass

    library.save_path(f, "lianli")  # auto-add to the Library

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
        raise gr.Error("Upload an image, GIF or video first.")
    kind = panel.media_kind(media)
    # Honor the saved device preference like every other tab (read fresh so a
    # settings change doesn't need a rebuild of this handler's defaults).
    dev = config.load().get("device", "auto")
    if dev not in _DEVICES:
        dev = "auto"
    progress(0.1, desc="Loading model…")
    try:
        if kind == "image":
            img = Image.open(media).convert("RGB")
            up = _get_upscaler(up_model, dev, 512, False)
            progress(0.4, desc="Upscaling…")
            result = up.upscale(img)
            fd, out = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".png")
            os.close(fd)
            result.save(out, "PNG")
            note = (f"✨ Source upscaled ×{up.scale} → {result.width}×{result.height}px. "
                    "Re-fit and export.")
        elif kind == "animated":
            from upscaler.video import upscale_video

            fd, out = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".mp4")
            os.close(fd)

            upscale_video(media, out, model=up_model, device=dev, tile=512,
                          progress_cb=_JobProgress(progress, "Upscaling frame"))
            note = "✨ Video source upscaled. Re-fit and export."
        else:
            raise gr.Error("That file type can't be enhanced — use an image, GIF or video.")
    except (RuntimeError, OSError) as e:  # OSError covers unreadable/corrupt uploads
        raise gr.Error(
            "Couldn't enhance the source. Check the file is a valid image or "
            "video; models also download on first use, so check your connection."
        ) from e
    return gr.update(value=out), panel.preview(out, _panel_params(*vals)), note


# ── Steam Workshop Showcase ───────────────────────────────────────────────────
_STEAM_FPS = ["10", "12", "15", "20", "24", "30"]
_STEAM_FMT_ANIM = "APNG (animated)"
_STEAM_FMT_GIF = "GIF (animated)"
_STEAM_FMT_STILL = "PNG (still)"


def _steam_params(fit, zoom, off_x, off_y, bg_color, transparent, tile_w, tile_h, repeat):
    """Build ShowcaseParams from the tab's controls (order = the preview
    input list minus the media file)."""
    return steam.ShowcaseParams(
        fit=fit, zoom=float(zoom), off_x=float(off_x), off_y=float(off_y),
        bg_color=steam.TRANSPARENT if transparent else bg_color,
        tile_w=int(tile_w), tile_h=int(tile_h), repeat=bool(repeat),
    )


def steam_apply_preset(media, preset, *vals):
    """Re-shape the controls for the chosen preset from the source's aspect.
    Returns updates for (fit, zoom, pan x, pan y, transparent, tile height,
    repeat); no-ops when there's no media or the preset is Custom."""
    keep = tuple(gr.update() for _ in range(7))
    src = panel._first_image(media) if media else None
    if src is None or preset == steam.PRESET_CUSTOM:
        return keep
    p = steam.apply_preset(preset, src.width, src.height, _steam_params(*vals))
    return (p.fit, p.zoom, p.off_x, p.off_y, p.transparent, p.tile_h, p.repeat)


def steam_preview_ui(media, *vals):
    """Live framing view + Steam profile mockup."""
    return steam.preview(media, _steam_params(*vals))


def steam_on_media(media):
    """On upload: fill trim End with the clip length, reveal the animation
    controls and pick the matching export format."""
    kind = panel.media_kind(media)
    is_anim = kind == "animated"
    dur = panel.media_duration(media) if is_anim else 0.0
    if not media:
        note = "Upload an image, GIF or video to begin."
    elif is_anim:
        note = (f"Animated source · {dur:g}s — exports five looping APNG tiles "
                f"(clips are capped at {steam.MAX_DURATION_SEC}s).")
    else:
        note = "Still image loaded — exports five PNG tiles."
    return (
        gr.update(value=dur, maximum=dur or None),
        gr.update(visible=is_anim),
        gr.update(value=_STEAM_FMT_ANIM if is_anim else _STEAM_FMT_STILL),
        note,
    )


def steam_export_ui(media, fit, zoom, off_x, off_y, bg_color, transparent, tile_w, tile_h,
                    repeat, out_fmt, fps, trim_start, trim_end, loop_mode, max_mb, hexify,
                    out_dir, progress=gr.Progress()):
    if not media:
        raise gr.Error("Upload an image, GIF or video first.")
    p = _steam_params(fit, zoom, off_x, off_y, bg_color, transparent, tile_w, tile_h, repeat)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"steam_{stamp}"
    animated = (out_fmt in (_STEAM_FMT_ANIM, _STEAM_FMT_GIF)
                and panel.media_kind(media) == "animated")
    max_mb = float(max_mb or 0)
    _STEAM_CANCEL.clear()
    progress(0.02, desc="Preparing…")
    try:
        if animated:
            res = steam.export_animated(
                media, p, fps=int(fps), trim_start=float(trim_start or 0),
                trim_end=float(trim_end or 0), loop_mode=loop_mode, max_mb=max_mb,
                out_dir=_ensure_export_dir(), stem=stem,
                fmt="gif" if out_fmt == _STEAM_FMT_GIF else "apng",
                hexify_for_steam=bool(hexify), progress=progress,
                cancel=_STEAM_CANCEL.is_set,
            )
        else:
            res = steam.export_stills(media, p, out_dir=_ensure_export_dir(), stem=stem,
                                      hexify_for_steam=bool(hexify))
    except steam.CancelledError:
        return gr.update(), gr.update(), "⏹ Export cancelled."
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise gr.Error(
            "Couldn't create the tiles. Animated export needs ffmpeg installed; "
            "otherwise check your source file and settings, then try again."
        ) from e

    # One ZIP with the five tiles (numbered in showcase order) + the upload steps.
    zpath = os.path.join(_ensure_export_dir(), f"steam_showcase_{stamp}.zip")
    with zipfile.ZipFile(zpath, "w") as z:
        for path in res.paths:
            z.write(path, os.path.basename(path))
        z.writestr("HOW-TO-UPLOAD.txt", steam.UPLOAD_GUIDE)

    for i, path in enumerate(res.paths, 1):
        library.save_path(path, f"steam-tile{i}")  # auto-add each tile to the Library
    library.save_path(zpath, "steam")

    msg = "✅ Tiles exported · " + steam.describe(res, max_mb).replace("\n", "  \n")
    out_dir = (out_dir or "").strip()
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            for path in res.paths:
                shutil.copyfile(path, os.path.join(out_dir, os.path.basename(path)))
            msg += f"\n\n📁 Saved copies to `{out_dir}`"
        except OSError as e:
            msg += f"\n\n⚠ Couldn't save to `{out_dir}`: {e}"
    gallery = [(path, f"Tile {i}") for i, path in enumerate(res.paths, 1)]
    return gallery, zpath, msg


def estimate_depth_ui(image, shape):
    """Work out how far away everything is, when the region is set to depth.

    Returns the map itself (kept in State for the blur, and shown so you can
    see what the model saw and click the part you want sharp) plus a status
    line. Never raises: without onnxruntime the other regions keep working.
    """
    if image is None or shape != blur.DEPTH:
        return None, gr.update(visible=False), gr.update(visible=False, value="")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    try:
        dmap = depth_tools.estimate(img)
    except (RuntimeError, OSError, ValueError) as e:
        return None, gr.update(visible=False), gr.update(visible=True, value=f"⚠ {e}")
    return (dmap, gr.update(visible=True, value=dmap),
            gr.update(visible=True,
                      value="Bright is near, dark is far. **Click the picture above "
                            "where you want it sharp**, the way you tap to focus on a "
                            "phone, then set how deep that sharp zone runs."))


def depth_focus_from_click(dmap, evt: gr.SelectData):
    """Clicking the depth map focuses there, like tapping a phone screen."""
    if dmap is None or evt is None or evt.index is None:
        return gr.update()
    x, y = evt.index
    return depth_tools.focus_at(dmap, x / max(1, dmap.width - 1) * 100.0,
                                y / max(1, dmap.height - 1) * 100.0)


def _blur_depth_vis(shape):
    """The depth controls appear only when the region is depth."""
    on = shape == blur.DEPTH
    return (gr.update(visible=on), gr.update(visible=on))


def detect_faces_ui(image, shape):
    """Find the faces in the input when the region is set to "faces".

    Returns the boxes (as fractions, so one detection serves both the
    downscaled preview and the full-size export) plus a status line. Never
    raises: without OpenCV the tab keeps working for every other region.
    """
    if image is None or shape != "faces":
        return [], gr.update(visible=False, value="")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    try:
        found = face.detect_faces(img)
    except (RuntimeError, OSError) as e:
        return [], gr.update(visible=True, value=f"⚠ {e}")
    if not found:
        return [], gr.update(
            visible=True,
            value="⚠ No faces found. Try a clearer or larger photo, or use the "
                  "ellipse / painted region instead.")
    return ([f.box() for f in found],
            gr.update(visible=True,
                      value=f"✅ Found **{len(found)}** face"
                            f"{'s' if len(found) != 1 else ''}."))


# ── Color & light ─────────────────────────────────────────────────────────────
# The tab's controls, in the order they're passed to the handlers. Keeping one
# list means the params mapping, the preset fan-out and Reset can't drift apart.
_ADJUST_FIELDS = [
    "exposure", "contrast", "highlights", "shadows", "black_point", "white_point",
    "gamma", "clarity", "temperature", "tint", "hue", "saturation", "vibrance",
    "mono", "mono_red", "mono_green", "mono_blue", "tone_color", "tone_strength",
]


def _adjust_params(*vals):
    kw = dict(zip(_ADJUST_FIELDS, vals))
    kw["mono"] = bool(kw["mono"])
    kw["tone_color"] = str(kw["tone_color"] or "#ffffff")
    for k, v in kw.items():
        if k not in ("mono", "tone_color"):
            kw[k] = float(v)
    return adjust.AdjustParams(**kw)


def _adjust_mask(shape, x, y, w, h, mangle, roundness, feather, outside, face_pad,
                 faces, editor):
    painted = None
    if shape == "painted":
        _bg, painted = _mask_from_editor(editor)
    return blur.MaskParams(shape=shape, x=float(x), y=float(y), w=float(w), h=float(h),
                           angle=float(mangle), roundness=float(roundness),
                           feather=float(feather), outside=bool(outside),
                           progressive=False, painted=painted,
                           faces=list(faces or []), face_pad=float(face_pad))


def _adjust_split(vals):
    n = len(_ADJUST_FIELDS)
    return _adjust_params(*vals[:n]), _adjust_mask(*vals[n:])


def _fan_adjust(p):
    """An AdjustParams → one value per control, in _ADJUST_FIELDS order."""
    return tuple(getattr(p, name) for name in _ADJUST_FIELDS)


def adjust_preview_ui(image, *vals):
    """Live before/after at preview size."""
    if image is None:
        return None
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    p, m = _adjust_split(vals)
    return adjust.preview_pair(img, p, m)


def adjust_preset_ui(name):
    """Load a preset into the controls (Reset uses the same path via "None")."""
    return _fan_adjust(adjust.preset(name))


def adjust_auto_ui(image, *vals):
    """Read the photo and set levels, gamma and white balance from it."""
    if image is None:
        raise gr.Error("Upload a photo first.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    p, _m = _adjust_split(vals)
    return _fan_adjust(adjust.auto_params(img, p))


def adjust_apply_ui(image, *vals, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload a photo to adjust.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    p, m = _adjust_split(vals)
    if m.shape == "painted" and m.painted is None:
        raise gr.Error("Paint over the area to adjust first (or set the region to whole).")
    if m.shape == "faces" and not m.faces:
        raise gr.Error("No faces were found in this photo — pick another region.")
    progress(0.2, desc="Adjusting at full size…")
    out = adjust.apply(img, p, m)
    progress(0.9, desc="Saving PNG…")
    fd, path = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".png")
    os.close(fd)
    out.save(path, "PNG")
    library.save_path(path, "adjust")  # auto-add to the Library
    return (img, out), path, f"✅ {adjust.describe(p, m)} · {out.width}×{out.height}px PNG"


# ── Effects & film looks ──────────────────────────────────────────────────────
# One ordered list so the params mapping, the preset fan-out and the control
# column can't drift apart (a test checks it against EffectParams).
_EFFECT_FIELDS = [
    "grain", "grain_size", "halation", "halation_threshold", "halation_radius",
    "halation_color", "leak", "leak_angle", "leak_color", "leak_softness",
    "vignette", "vignette_radius", "vignette_feather", "aberration",
    "duotone", "duotone_dark", "duotone_light", "posterize", "dither", "dither_levels",
    "halftone", "halftone_cell", "halftone_angle",
    "scanlines", "scanline_spacing", "glitch", "glitch_seed",
]
_EFFECT_COLORS = {"halation_color", "leak_color", "duotone_dark", "duotone_light"}
_EFFECT_INTS = {"posterize", "dither_levels", "glitch_seed"}


def _effect_params(*vals):
    kw = {}
    for name, value in zip(_EFFECT_FIELDS, vals):
        if name in _EFFECT_COLORS:
            kw[name] = str(value or "#ffffff")
        elif name in _EFFECT_INTS:
            kw[name] = int(value or 0)
        else:
            kw[name] = float(value)
    return effects.EffectParams(**kw)


def _effect_split(vals):
    n = len(_EFFECT_FIELDS)
    return _effect_params(*vals[:n]), _adjust_mask(*vals[n:])


def _fan_effects(p):
    return tuple(getattr(p, name) for name in _EFFECT_FIELDS)


def effects_preview_ui(image, *vals):
    """Live before/after at preview size."""
    if image is None:
        return None
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    p, m = _effect_split(vals)
    return effects.preview_pair(img, p, m)


def effects_preset_ui(name):
    """Load a look into the controls ("None" clears them)."""
    return _fan_effects(effects.preset(name))


def effects_apply_ui(image, *vals, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload a photo to add effects to.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    p, m = _effect_split(vals)
    if m.shape == "painted" and m.painted is None:
        raise gr.Error("Paint over the area for the effects first (or set the region to whole).")
    if m.shape == "faces" and not m.faces:
        raise gr.Error("No faces were found in this photo — pick another region.")
    progress(0.2, desc="Rendering at full size…")
    out = effects.apply(img, p, m)
    progress(0.9, desc="Saving PNG…")
    fd, path = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".png")
    os.close(fd)
    out.save(path, "PNG")
    library.save_path(path, "effects")  # auto-add to the Library
    return (img, out), path, f"✅ {effects.describe(p, m)} · {out.width}×{out.height}px PNG"


# ── Watermark ─────────────────────────────────────────────────────────────────
_WM_FIELDS = [
    "kind", "text", "font", "size", "color", "outline", "outline_width", "shadow",
    "logo_scale", "position", "margin", "rotation", "opacity", "tile_gap", "tile_angle",
    "behind", "cutout_feather", "subject_shadow",
]
_WM_BOOL = {"behind"}
_WM_TEXT = {"kind", "text", "font", "color", "outline", "position"}


def _wm_params(*vals):
    kw = {}
    for name, value in zip(_WM_FIELDS, vals):
        if name in _WM_TEXT:
            kw[name] = "" if value is None else str(value)
        elif name in _WM_BOOL:
            kw[name] = bool(value)
        elif name == "cutout_feather":
            kw[name] = int(value or 0)
        else:
            kw[name] = float(value)
    return wm_tools.WatermarkParams(**kw)


def wm_cutout_ui(image, behind, feather):
    """Lift the subject off its background, once, so every slider afterwards
    stays instant instead of re-running the model on each move."""
    if image is None or not behind:
        return None, gr.update(visible=False, value="")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    p = wm_tools.WatermarkParams(cutout_feather=int(feather or 0))
    try:
        cut = wm_tools.subject_cutout(img, p)
    except (RuntimeError, ValueError, OSError) as e:
        return None, gr.update(visible=True, value=f"⚠ {e}")
    covered = np.asarray(cut.getchannel("A"), dtype=np.float32) > 128
    share = float(covered.mean()) * 100.0
    note = (f"Subject lifted — it covers **{share:.0f}%** of the frame. The mark "
            "goes behind it.")
    if share < 1:
        note = ("⚠ Almost nothing was found to put the text behind. This works on "
                "photos with a clear subject, like a person or an object.")
    return cut, gr.update(visible=True, value=note)


def _wm_behind_vis(behind):
    """The cut-out controls appear only when the mark goes behind."""
    return (gr.update(visible=bool(behind)), gr.update(visible=bool(behind)))


def _wm_vis(kind, position):
    """Show the controls the chosen mark and placement actually use."""
    is_text = kind == "text"
    tiled = position == wm_tools.TILED
    return (
        gr.update(visible=is_text),      # text box
        gr.update(visible=is_text),      # font
        gr.update(visible=is_text),      # size
        gr.update(visible=is_text),      # colour
        gr.update(visible=is_text),      # outline colour
        gr.update(visible=is_text),      # outline width
        gr.update(visible=is_text),      # shadow
        gr.update(visible=not is_text),  # logo upload
        gr.update(visible=not is_text),  # logo scale
        gr.update(visible=not tiled),    # margin
        gr.update(visible=tiled),        # tile gap
        gr.update(visible=tiled),        # tile angle
    )


def watermark_preview_ui(image, logo, cutout, *vals):
    if image is None:
        return None, gr.update(value="")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    p = _wm_params(*vals)
    if p.kind == "logo" and logo is None:
        return wm_tools.preview(img, p, cutout=cutout), gr.update(
            value="⚠ Upload a logo image, or switch the mark back to text.")
    return (wm_tools.preview(img, p, logo, cutout=cutout),
            gr.update(value=wm_tools.describe(p)))


def watermark_preset_ui(name):
    p = wm_tools.preset(name)
    return tuple(getattr(p, f) for f in _WM_FIELDS)


def watermark_apply_ui(image, logo, cutout, *vals, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload a photo to watermark.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    p = _wm_params(*vals)
    progress(0.3, desc="Stamping at full size…")
    try:
        out = wm_tools.apply(img, p, logo, cutout)
    except (ValueError, RuntimeError) as e:
        raise gr.Error(str(e)) from e
    progress(0.9, desc="Saving PNG…")
    fd, path = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".png")
    os.close(fd)
    out.save(path, "PNG")
    library.save_path(path, "watermark")  # auto-add to the Library
    return out, path, f"✅ {wm_tools.describe(p)} · {out.width}×{out.height}px PNG"


# ── Design templates ──────────────────────────────────────────────────────────
DZ_SLOTS = 5          # how many text boxes the tab keeps for a template's slots


def _dz_build(tpl_json, canvas, primary, secondary, accent, ink):
    """The template the controls currently describe: structure from the JSON,
    palette and canvas from the live pickers."""
    tpl = dz_tools.from_json(tpl_json) if tpl_json else dz_tools.built_in(
        dz_tools.BUILT_IN_NAMES[0])
    tpl.palette = dz_tools.Palette(primary=primary or "#000000",
                                   secondary=secondary or "#000000",
                                   accent=accent or "#ffffff",
                                   ink=ink or "#ffffff")
    if canvas:
        tpl.canvas = canvas
    return tpl


def _dz_texts(tpl, values):
    """Pair the text boxes with the template's slots, in order."""
    return {slot: value for slot, value in zip(tpl.text_slots(), values)
            if value is not None}


def _dz_fields(tpl):
    """Label, prefill and show one box per text slot; hide the spare ones."""
    slots = tpl.text_slots()
    defaults = {}
    for layer in tpl.layers:
        if layer.kind == dz_tools.TEXT and layer.slot not in defaults:
            defaults[layer.slot] = layer.text
    out = []
    for i in range(DZ_SLOTS):
        if i < len(slots):
            out.append(gr.update(visible=True, label=slots[i].replace("_", " ").title(),
                                 value=defaults.get(slots[i], "")))
        else:
            out.append(gr.update(visible=False, value=""))
    return out


def design_pick_ui(name):
    """Load a template that ships: its JSON, its palette, its canvas, its slots."""
    tpl = dz_tools.built_in(name)
    pal = tpl.palette
    return (dz_tools.to_json(tpl), f"*{tpl.note}*", tpl.canvas,
            pal.primary, pal.secondary, pal.accent, pal.ink,
            *_dz_fields(tpl),
            gr.update(visible=tpl.wants_photo()),
            gr.update(visible=tpl.wants_logo()))


def design_json_ui(text):
    """Take an edited template. A broken one says so instead of blanking the tab."""
    try:
        tpl = dz_tools.from_json(text)
    except dz_tools.DesignError as e:
        return (gr.update(), f"⚠ {e}", gr.update(), *(gr.update() for _ in range(4)),
                *(gr.update() for _ in range(DZ_SLOTS)), gr.update(), gr.update())
    pal = tpl.palette
    return (dz_tools.to_json(tpl), f"*{tpl.note}*" if tpl.note else "", tpl.canvas,
            pal.primary, pal.secondary, pal.accent, pal.ink,
            *_dz_fields(tpl),
            gr.update(visible=tpl.wants_photo()),
            gr.update(visible=tpl.wants_logo()))


def design_cutout_ui(photo, tpl_json):
    """Lift the subject once, for the templates that put type behind a person."""
    try:
        tpl = dz_tools.from_json(tpl_json) if tpl_json else None
    except dz_tools.DesignError:
        return None, gr.update(visible=False, value="")
    if photo is None or tpl is None or not tpl.wants_subject():
        return None, gr.update(visible=False, value="")
    img = photo if isinstance(photo, Image.Image) else Image.fromarray(photo)
    try:
        cut = dz_tools.subject_cutout(img)
    except (RuntimeError, ValueError, OSError) as e:
        return None, gr.update(visible=True, value=f"⚠ {e}")
    share = float((np.asarray(cut.getchannel("A"), np.float32) > 128).mean()) * 100
    note = f"Subject lifted — it covers **{share:.0f}%** of the photo."
    if share < 1:
        note = ("⚠ Almost nothing was found to cut out. This template wants a photo "
                "with one clear subject.")
    return cut, gr.update(visible=True, value=note)


def design_preview_ui(tpl_json, photo, logo, cutout, canvas, primary, secondary,
                      accent, ink, *texts):
    try:
        tpl = _dz_build(tpl_json, canvas, primary, secondary, accent, ink)
        img = photo if photo is None or isinstance(photo, Image.Image) \
            else Image.fromarray(photo)
        out = dz_tools.preview(tpl, photo=img, texts=_dz_texts(tpl, texts), logo=logo,
                               cutout=cutout)
    except (dz_tools.DesignError, RuntimeError, OSError) as e:
        return None, gr.update(value=f"⚠ {e}")
    gaps = dz_tools.missing(tpl, img, _dz_texts(tpl, texts), logo)
    note = dz_tools.describe(tpl)
    if gaps:
        note += f" · still waiting for **{', '.join(gaps)}**"
    return out, gr.update(value=note)


def design_apply_ui(tpl_json, photo, logo, cutout, canvas, primary, secondary,
                    accent, ink, *texts, progress=gr.Progress()):
    progress(0.3, desc="Rendering at full size…")
    try:
        tpl = _dz_build(tpl_json, canvas, primary, secondary, accent, ink)
        img = photo if photo is None or isinstance(photo, Image.Image) \
            else Image.fromarray(photo)
        out = dz_tools.render(tpl, photo=img, texts=_dz_texts(tpl, texts), logo=logo,
                              cutout=cutout)
    except (dz_tools.DesignError, RuntimeError, OSError) as e:
        raise gr.Error(str(e)) from e
    progress(0.9, desc="Saving PNG…")
    fd, path = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".png")
    os.close(fd)
    out.save(path, "PNG")
    library.save_path(path, "design")  # auto-add to the Library
    return out, path, f"✅ {dz_tools.describe(tpl)} · {out.width}×{out.height}px PNG"


# ── Screenshot beautifier ─────────────────────────────────────────────────────
_SHOT_FIELDS = [
    "background", "color", "color2", "angle", "padding", "corner_radius", "rim",
    "shadow", "shadow_softness", "chrome", "title", "tilt_y", "tilt_x", "spin",
    "aspect", "custom_aspect", "out_size",
]
_SHOT_TEXT = {"background", "color", "color2", "chrome", "title", "aspect",
            "custom_aspect", "out_size"}


def _shot_params(*vals):
    kw = {}
    for name, value in zip(_SHOT_FIELDS, vals):
        kw[name] = ("" if value is None else str(value)) if name in _SHOT_TEXT \
            else float(value)
    return shot_tools.ShotParams(**kw)


def _shot_vis(bg, chrome, aspect):
    """Show only the controls the chosen look actually uses."""
    ramp = bg in (shot_tools.GRADIENT, shot_tools.MESH)
    tinted = bg not in (shot_tools.BLURRED, shot_tools.TRANSPARENT)
    return (
        gr.update(visible=tinted),                                   # colour
        gr.update(visible=ramp),                                     # second colour
        gr.update(visible=ramp),                                     # gradient angle
        gr.update(visible=str(chrome).startswith("browser")),        # address bar text
        gr.update(visible=aspect == shot_tools.CUSTOM_ASPECT),       # custom ratio
    )


def screenshot_preview_ui(image, *vals):
    if image is None:
        return None, gr.update(value="")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    p = _shot_params(*vals)
    try:
        out = shot_tools.preview(img, p)
        w, h = shot_tools.result_size(img.size, p)
    except ValueError as e:
        return None, gr.update(value=f"⚠ {e}")
    return out, gr.update(
        value=f"{shot_tools.describe(p)} · **{w}×{h}px** at full size")


def screenshot_preset_ui(name):
    p = shot_tools.preset(name)
    return tuple(getattr(p, f) for f in _SHOT_FIELDS)


def screenshot_apply_ui(image, *vals, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload or paste a screenshot first.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    p = _shot_params(*vals)
    progress(0.3, desc="Composing at full size…")
    try:
        out = shot_tools.apply(img, p)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    progress(0.9, desc="Saving PNG…")
    fd, path = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".png")
    os.close(fd)
    out.save(path, "PNG")
    library.save_path(path, "screenshot")  # auto-add to the Library
    return out, path, f"✅ {shot_tools.describe(p)} · {out.width}×{out.height}px PNG"


# ── Crop & frame ──────────────────────────────────────────────────────────────
_FRAME_FIELDS = [
    "rotate", "flip_h", "flip_v", "keystone_h", "keystone_v", "straighten",
    "aspect", "custom_aspect", "crop_mode", "position_x", "position_y", "zoom",
    "out_size", "border", "border_style", "border_color", "border_blur",
    "corner_radius", "shadow",
]
_FRAME_TEXT = {"aspect", "custom_aspect", "crop_mode", "out_size", "border_style",
               "border_color"}
_FRAME_BOOL = {"flip_h", "flip_v"}


def _frame_params(*vals):
    kw = {}
    for name, value in zip(_FRAME_FIELDS, vals):
        if name in _FRAME_TEXT:
            kw[name] = "" if value is None else str(value)
        elif name in _FRAME_BOOL:
            kw[name] = bool(value)
        elif name == "rotate":
            kw[name] = int(value or 0)
        else:
            kw[name] = float(value)
    return frame_tools.FrameParams(**kw)


def _frame_vis(aspect, border, border_style):
    """Only the controls that can do anything are shown."""
    has_ratio = aspect not in (frame_tools.DEFAULT_ASPECT,)
    solid = border > 0 and border_style == "solid"
    fill = (border > 0 or aspect != frame_tools.DEFAULT_ASPECT) and border_style == "blurred photo"
    return (
        gr.update(visible=aspect == frame_tools.CUSTOM_ASPECT),   # custom ratio box
        gr.update(visible=has_ratio),                             # fill / fit
        gr.update(visible=solid),                                 # border colour
        gr.update(visible=fill),                                  # blur amount
    )


def frame_preview_ui(image, *vals):
    """The framed result at preview size, plus what it will come out as."""
    if image is None:
        return None, gr.update(value="")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    p = _frame_params(*vals)
    try:
        out = frame_tools.preview(img, p)
        w, h = frame_tools.result_size(img.size, p)
    except (ValueError, OSError) as e:
        return None, gr.update(value=f"⚠ {e}")
    return out, gr.update(value=f"**{img.width}×{img.height}** → **{w}×{h}** · "
                                f"{frame_tools.describe(p)}")


def frame_preset_ui(name):
    p = frame_tools.preset(name)
    return tuple(getattr(p, f) for f in _FRAME_FIELDS)


def frame_apply_ui(image, *vals, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload a photo to crop or frame.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    p = _frame_params(*vals)
    progress(0.3, desc="Rendering at full size…")
    try:
        out = frame_tools.apply(img, p)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    progress(0.9, desc="Saving PNG…")
    fd, path = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".png")
    os.close(fd)
    out.save(path, "PNG")
    library.save_path(path, "frame")  # auto-add to the Library
    return out, path, f"✅ {frame_tools.describe(p, img.size)}"


# ── Sharpen ───────────────────────────────────────────────────────────────────
_SHARPEN_FIELDS = [
    "kind", "amount", "radius", "threshold", "halo", "luminance_only",
    "protect_shadows", "protect_highlights", "edge_sensitivity", "detail_balance",
]


def _sharpen_kind_vis(kind):
    """Only the chosen kind's own control is shown."""
    return (
        gr.update(visible=kind == "smart"),      # edge sensitivity
        gr.update(visible=kind == "texture"),    # detail balance
    )


def _sharpen_params(kind, amount, radius, threshold, halo, luminance_only,
                    protect_shadows, protect_highlights, edge_sensitivity, detail_balance):
    return sharpen_tools.SharpenParams(
        kind=kind, amount=float(amount), radius=float(radius), threshold=float(threshold),
        halo=float(halo), luminance_only=bool(luminance_only),
        protect_shadows=float(protect_shadows), protect_highlights=float(protect_highlights),
        edge_sensitivity=float(edge_sensitivity), detail_balance=float(detail_balance),
    )


def _sharpen_split(vals):
    """(params, mask, preview centre) from the flat control list."""
    n = len(_SHARPEN_FIELDS)
    p = _sharpen_params(*vals[:n])
    center = (float(vals[-2]) / 100.0, float(vals[-1]) / 100.0)
    return p, _adjust_mask(*vals[n:-2]), center


def sharpen_preview_ui(image, *vals):
    """Live before/after at 1:1 — sharpening can only be judged at full size,
    so this is a real crop of the photo, never a downscale."""
    if image is None:
        return None, gr.update()
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    p, m, center = _sharpen_split(vals)
    before, after = sharpen_tools.preview_pair(img, p, m, center=center)
    note = (f"Showing a **1:1 crop** — {before.width}×{before.height} of "
            f"{img.width}×{img.height}. Move the crop with the sliders above."
            if before.size != img.size else
            f"Showing the whole photo at 1:1 — {img.width}×{img.height}.")
    return (before, after), gr.update(value=note)


def sharpen_preset_ui(name):
    p = sharpen_tools.preset(name)
    return tuple(getattr(p, f) for f in _SHARPEN_FIELDS)


def sharpen_apply_ui(image, *vals, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload a photo to sharpen.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    p, m, _center = _sharpen_split(vals)
    if m.shape == "painted" and m.painted is None:
        raise gr.Error("Paint over the area to sharpen first (or set the region to whole).")
    if m.shape == "faces" and not m.faces:
        raise gr.Error("No faces were found in this photo — pick another region.")
    progress(0.2, desc="Sharpening at full size…")
    out = sharpen_tools.apply(img, p, m)
    progress(0.9, desc="Saving PNG…")
    fd, path = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".png")
    os.close(fd)
    out.save(path, "PNG")
    library.save_path(path, "sharpen")  # auto-add to the Library
    return ((img, out), path,
            f"✅ {sharpen_tools.describe(p, m, img.size)} · {out.width}×{out.height}px PNG")


# ── Blur toolbox ──────────────────────────────────────────────────────────────
_BLUR_KIND_INFO = {
    "gaussian": "Soft, natural blur.",
    "box": "Flat average — harsher, cheap-camera look.",
    "motion": "Streaks along a direction, like camera shake or a moving subject.",
    "spin": "Rotation blur around a centre point.",
    "zoom": "Radial streaks out of a centre point.",
    "lens": "Disc-shaped bokeh; add highlight bloom for bright discs.",
    "pixelate": "Mosaic squares — the privacy blur.",
    "surface": "Edge-preserving smoothing: softens skin and noise, keeps edges.",
}


def _blur_kind_vis(kind):
    """Show only the controls the chosen blur kind uses."""
    return (
        gr.update(visible=kind == "motion"),
        gr.update(visible=kind in ("spin", "zoom")),
        gr.update(visible=kind in ("spin", "zoom")),
        gr.update(visible=kind == "lens"),
        gr.update(visible=kind == "surface"),
        _BLUR_KIND_INFO.get(kind, ""),
    )


def _region_vis(shape):
    """Show only the region controls the chosen shape uses, in the order
    (centre x, centre y, width, height, band tilt, roundness, feather,
    outside, editor). A band defaults to affecting the outside — that's
    tilt-shift for blur and a graduated filter for colour."""
    box = shape in ("rectangle", "ellipse")
    return (
        gr.update(visible=box or shape == "band"),    # centre x
        gr.update(visible=box or shape == "band"),    # centre y
        gr.update(visible=box),                       # width
        gr.update(visible=box or shape == "band",
                  label="Band thickness (%)" if shape == "band" else "Height (%)"),
        gr.update(visible=shape == "band"),           # band tilt
        gr.update(visible=shape == "rectangle"),      # roundness
        gr.update(visible=shape != "whole"),          # feather
        gr.update(visible=shape != "whole", value=(shape == "band")),   # outside
        gr.update(visible=shape == "faces"),          # face padding
        gr.update(visible=shape == "painted"),        # editor
    )


def _blur_shape_vis(shape):
    """_region_vis plus the blur-only "graded edge" toggle, which sits right
    after "outside" in the Blur tab's output list."""
    v = _region_vis(shape)
    return v[:8] + (gr.update(visible=shape != "whole"),) + v[8:]


def _blur_params(kind, strength, angle, cx, cy, highlights, threshold):
    return blur.BlurParams(kind=kind, strength=float(strength), angle=float(angle),
                           center_x=float(cx), center_y=float(cy),
                           highlights=float(highlights), threshold=float(threshold))


def _mask_params(shape, x, y, w, h, mangle, roundness, feather, outside, progressive,
                 face_pad, faces, editor, dmap=None, focus=70.0, dof=25.0):
    painted = None
    if shape == "painted":
        _bg, painted = _mask_from_editor(editor)
    return blur.MaskParams(shape=shape, x=float(x), y=float(y), w=float(w), h=float(h),
                           angle=float(mangle), roundness=float(roundness),
                           feather=float(feather), outside=bool(outside),
                           progressive=bool(progressive), painted=painted,
                           faces=list(faces or []), face_pad=float(face_pad),
                           depth=dmap, focus=float(focus), dof=float(dof))


def _blur_split(vals):
    """The flat control list → (BlurParams, MaskParams)."""
    return _blur_params(*vals[:7]), _mask_params(*vals[7:])


def blur_preview_ui(image, *vals):
    """Live before/after at preview size (cheap: strength is relative, so it
    looks like the export)."""
    if image is None:
        return None
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    bp, mp = _blur_split(vals)
    before, after = blur.preview_pair(img, bp, mp)
    return (before, after)


def blur_apply_ui(image, *vals, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload an image to blur.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    bp, mp = _blur_split(vals)
    if mp.shape == "painted" and mp.painted is None:
        raise gr.Error("Paint over the area to blur first (or pick another region shape).")
    if mp.shape == "faces" and not mp.faces:
        raise gr.Error("No faces were found in this photo — pick another region.")
    if mp.shape == blur.DEPTH and mp.depth is None:
        raise gr.Error("The depth map isn't ready. Re-pick the depth region, or "
                       'install onnxruntime with: pip install -e ".[onnx]"')
    progress(0.2, desc="Blurring at full size…")
    out = blur.apply(img, bp, mp)
    progress(0.9, desc="Saving PNG…")
    fd, path = tempfile.mkstemp(dir=_ensure_export_dir(), suffix=".png")
    os.close(fd)
    out.save(path, "PNG")
    library.save_path(path, "blur")  # auto-add to the Library
    return (img, out), path, f"✅ {blur.describe(bp, mp, img.size)} · {out.width}×{out.height}px PNG"


def blur_on_image(image):
    """New input → load it into the paint editor (for painted masks)."""
    return gr.update(value=image) if image is not None else gr.update(value=None)


_MODEL_CHOICES = [(f"{s.name}  (×{s.scale}) — {s.notes}", s.name) for s in MODELS.values()]
_DEBLUR_CHOICES = [(f"{s.name} — {s.notes}", s.name) for s in DEBLUR_MODELS.values()]
_FACE_CHOICES = [(f"{s.name} — {s.notes}", s.name) for s in FACE_MODELS.values()]


def _available_devices() -> "list[str]":
    """Only devices this machine can actually run — offering cuda/mps that
    isn't installed just hands the user an error toast."""
    import torch

    devs = ["auto", "cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


_DEVICES = _available_devices()


def _dml_available() -> bool:
    """True when ONNX Runtime can reach a GPU through DirectML (AMD/Intel/NVIDIA
    on Windows). Used to default the video ONNX toggle on where torch is
    CPU-only but the GPU is still reachable this way."""
    try:
        import onnxruntime as ort

        return "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False

# One-click starting points for the Upscale tab. Each tunes the model + denoise
# + sharpen for a use-case; restoration always uses SIDD (denoise) since GoPro
# garbages noisy photos. Users can tweak anything after applying a preset.
UPSCALE_PRESETS: dict[str, dict] = {
    "📷 Photo": dict(
        model="realesrgan-x2plus", sharpen=0.3, restore=False, strength=1.0,
        hint="Balanced general-purpose. ×2 keeps already-good photos natural.",
    ),
    "🙂 Faces": dict(
        model="realesrgan-x2plus", sharpen=0.5, restore=True, strength=0.5,
        hint="Gentle ×2 + light denoise so skin stays natural (no plastic look).",
    ),
    "💧 Soft skin": dict(
        model="realesrgan-x2plus", sharpen=0.0, restore=True, strength=0.7,
        hint="Smoother, softer look — more denoise, no sharpening. Good for portraits.",
    ),
    "🖼️ Portrait": dict(
        model="4x-remacri", sharpen=0.2, restore=True, strength=0.4,
        hint="Remacri ×4 for natural skin & hair, with light denoise — great for people.",
    ),
    "📱 Phone snap": dict(
        model="realesrgan-x2plus", sharpen=0.4, restore=True, strength=0.4,
        hint="Cleans the mild noise/compression in everyday phone photos, then ×2.",
    ),
    "🕰️ Old / vintage": dict(
        model="realesrgan-x2plus", sharpen=0.4, restore=True, strength=0.8,
        hint="Strong denoise to clean grain in old/faded photos, gentle ×2.",
    ),
    "🌙 Low-light / noisy": dict(
        model="realesrgan-x2plus", sharpen=0.2, restore=True, strength=1.0,
        hint="Strong denoise first, gentle ×2, low sharpen so grain isn't amplified.",
    ),
    "🎨 Anime / art": dict(
        model="realesrgan-x4plus-anime", sharpen=0.0, restore=False, strength=1.0,
        hint="Anime model at ×4, no sharpen (line art needs none).",
    ),
    "🌿 Nature": dict(
        model="realesrgan-x4plus", sharpen=0.7, restore=False, strength=1.0,
        hint="×4 for maximum texture/detail in foliage & landscapes, crisper edges.",
    ),
    "🏙️ Max detail": dict(
        model="realesrgan-x4plus", sharpen=1.1, restore=False, strength=1.0,
        hint="×4 + strong sharpening for hard edges (architecture, products). Clean sources only.",
    ),
}


def apply_preset(name):
    """Return component updates for the chosen preset, plus button-variant
    updates so the active preset is highlighted. Restoration is always the safe
    SIDD denoiser."""
    p = UPSCALE_PRESETS[name]
    controls = (
        gr.update(value=p["model"]),                 # model
        gr.update(value=p["sharpen"]),               # sharpen
        gr.update(value=p["restore"]),               # deblur (Restore first)
        gr.update(value="nafnet-sidd-width64"),      # deblur_model
        gr.update(value=p["strength"]),              # restore_strength
        f"**{name}** — {p['hint']}",                 # preset_info
    )
    # highlight the active preset with an accent OUTLINE (a CSS class), leaving
    # the real Enhance button as the only filled/primary button on screen.
    highlights = tuple(
        gr.update(elem_classes=["preset-active"] if pn == name else [])
        for pn in UPSCALE_PRESETS
    )
    return controls + highlights


# -- Library (everything you export, saved automatically) --------------------

def refresh_library():
    """Reload the Library tab from disk — newest first. Returns updates for
    (gallery, video picker, video preview, count message)."""
    imgs, vids = library.list_items()
    vid_choices = [(os.path.basename(v), v) for v in vids]
    first_vid = vids[0] if vids else None
    n = len(imgs) + len(vids)
    if n:
        msg = (
            f"**{n}** item{'s' if n != 1 else ''} in your library · "
            f"{len(imgs)} image/GIF · {len(vids)} video"
            f"{'s' if len(vids) != 1 else ''}. Newest first."
        )
    else:
        msg = ("Your library is empty — export anything (an upscale, a GIF, a "
               "converted file…) and it'll appear here automatically.")
    return (
        imgs,
        gr.update(choices=vid_choices, value=first_vid),
        first_vid,
        msg,
    )


def open_library_folder():
    """Open the library folder in the OS file manager. This is a local app, so
    it opens on the machine running it — i.e. the user's own computer."""
    import subprocess
    import sys

    path = str(library.ensure_dir())
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", path])
        elif sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]  # noqa: F821 (Windows-only)
        else:
            subprocess.run(["xdg-open", path])
    except (OSError, FileNotFoundError):
        pass


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

# Hover-magnifier (loupe) over result images. A document-level mousemove draws a
# zoomed circular lens; only active while body has `loupe-on` (toggled by the
# 🔍 button) and only over images inside a `.loupe` block. Injected in <head>.
_MAGNIFIER_HEAD = """
<script>
(function(){
  const ZOOM = 2.5, R = 110;
  let lens = null;
  function lensEl(){
    if(!lens){ lens = document.createElement('div'); lens.className='mag-lens';
      (document.body || document.documentElement).appendChild(lens); }
    return lens;
  }
  function hide(){ if(lens) lens.style.display='none'; }
  function drawnRect(img){
    // The bitmap rarely fills the <img> box: Gradio letterboxes with
    // object-fit contain/cover/scale-down. Compute the actually-painted
    // rectangle so the lens zooms what's under the cursor, undistorted.
    const r = img.getBoundingClientRect();
    const nw = img.naturalWidth, nh = img.naturalHeight;
    if(!nw || !nh) return r;
    const fit = getComputedStyle(img).objectFit;
    if(fit !== 'contain' && fit !== 'cover' && fit !== 'scale-down') return r;
    let s = (fit === 'cover') ? Math.max(r.width/nw, r.height/nh)
                              : Math.min(r.width/nw, r.height/nh);
    if(fit === 'scale-down') s = Math.min(s, 1);
    const w = nw*s, h = nh*s;
    return { left: r.left + (r.width - w)/2, top: r.top + (r.height - h)/2,
             width: w, height: h };
  }
  function onMove(e){
    if(!document.body.classList.contains('loupe-on')){ hide(); return; }
    const img = e.target;
    if(!(img && img.tagName==='IMG' && img.closest('.loupe') && img.src)){ hide(); return; }
    const r = drawnRect(img);
    const x = e.clientX - r.left, y = e.clientY - r.top;
    if(x<0||y<0||x>r.width||y>r.height){ hide(); return; }
    const L = lensEl();
    L.style.display='block';
    L.style.left = e.clientX+'px';
    L.style.top  = e.clientY+'px';
    L.style.backgroundImage = 'url("'+img.src+'")';
    L.style.backgroundSize = (r.width*ZOOM)+'px '+(r.height*ZOOM)+'px';
    L.style.backgroundPosition = (-(x*ZOOM - R))+'px '+(-(y*ZOOM - R))+'px';
  }
  function init(){ document.addEventListener('mousemove', onMove, {passive:true}); }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
</script>
"""

_CSS = """
/* (No font @import here — THEME's GoogleFont already loads Manrope, and it
   falls back to system-ui when offline.) */

/* Hover magnifier (loupe) */
.mag-lens { position: fixed; pointer-events: none; display: none;
    width: 220px; height: 220px; border-radius: 50%;
    /* the lens is appended to <body>, outside the .gradio-container scope that
       defines --ac — so give the accent a literal fallback */
    border: 3px solid var(--ac, #0F766E); background-color: #000; background-repeat: no-repeat;
    box-shadow: 0 6px 24px rgba(0,0,0,.45); transform: translate(-50%,-50%);
    z-index: 99999; }
body.loupe-on .loupe img { cursor: crosshair; }
body.loupe-on #mag-btn { background: var(--ac) !important; color: #fff !important;
    border-color: var(--ac) !important; }

/* Custom CSS uses Gradio theme vars (--body-text-color etc.) so it adapts to
   both light and dark automatically. --ac is the accent (teal), brighter in dark. */
.gradio-container { --ac: #0F766E; --ac-weak: rgba(13,148,136,.10); }
.dark .gradio-container, .dark { --ac: #2DD4BF; --ac-weak: rgba(45,212,191,.13); }

/* gradio-app carries the .dark scope, so fill the viewport with IT (html/body
   sit outside the scope and would otherwise show a strip behind the app). The
   html/body fallback covers light mode; gradio-app (100vh) covers dark. */
html, body { background: var(--body-background-fill) !important; }
gradio-app { display: block; min-height: 100vh;
    background: var(--body-background-fill) !important; }
/* Flat, calm surface — no decorative glow or dot texture (matches Upscayl /
   Krea / upscale.media). Just the theme's neutral background. */
.gradio-container { max-width: 100% !important; padding: 6px 44px 64px !important;
    position: relative;
    background: var(--body-background-fill) !important; }

/* Cheap transitions on interactive controls ONLY — color/border, no box-shadow
   or transform on every .block (that caused heavy repaints / ~20fps jank). */
button, .drop, .item, .dropdown-arrow {
    transition: background-color .18s ease, border-color .18s ease, color .18s ease; }
/* transform/box-shadow only on the single button being hovered (cheap). */
.gradio-container button.primary {
    transition: background-color .18s ease, transform .12s ease, box-shadow .2s ease; }
.gradio-container button.primary:hover { transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(13,148,136,.28); }
.gradio-container button.primary:active { transform: translateY(0); box-shadow: none; }

/* Entrance motion kept to a single subtle hero fade (opacity only). Sections,
   tab bodies and accordion bodies render statically — no movement on every tab
   switch or accordion open. */
@keyframes fadeUp { from { opacity: 0; } to { opacity: 1; } }
#hero { will-change: opacity; animation: fadeUp .4s ease both; }
.label-wrap .icon { transition: transform .25s cubic-bezier(.22,.61,.36,1) !important; }

/* Dropdown popover: opacity-only fade so it never animates its POSITION while
   Gradio is still deciding to place it above/below the box (that transform was
   the "jump up then drop down" flicker). Keep z-index/shadow/solid-bg so it
   still reads as a floating layer. */
@keyframes ddOpen { from { opacity: 0; } to { opacity: 1; } }
ul.options, .options { animation: ddOpen .12s ease-out; z-index: 200 !important;
    box-shadow: 0 8px 28px rgba(28,25,23,.16) !important;
    background: var(--block-background-fill) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 12px !important; padding: 5px !important;
    max-height: 340px !important; }
/* Roomy, rounded option rows with a clear hover / keyboard-active state and an
   accent tint + weight on the currently-selected value. */
ul.options .item, .options .item { transition: background-color .12s ease;
    padding: 9px 12px !important; border-radius: 8px !important;
    line-height: 1.4; margin: 1px 0; }
ul.options .item:hover, .options .item:hover,
ul.options .item.active, .options .item.active {
    background: var(--ac-weak) !important; }
ul.options .item.selected, .options .item.selected {
    color: var(--ac) !important; font-weight: 600; }
ul.options::-webkit-scrollbar { width: 8px; }
ul.options::-webkit-scrollbar-thumb { background: var(--border-color-primary);
    border-radius: 8px; }
ul.options::-webkit-scrollbar-track { background: transparent; }
.dropdown-arrow { transition: transform .25s cubic-bezier(.22,.61,.36,1); }

/* --- Smooth, purposeful micro-interactions (opacity / tiny transform only,
   nothing that loops) — content eases in on transitions, controls give quiet
   hover + focus feedback. --- */
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
/* Tab content and accordion bodies ease in instead of snapping. */
.tabitem { animation: fadeIn .28s ease; }
[data-testid="accordion-content"] { animation: fadeIn .22s ease; }
/* Gentle hover lift on secondary + preset buttons (primary already lifts). */
.gradio-container button.secondary { transition: background-color .18s ease,
    border-color .18s ease, color .18s ease, transform .12s ease, box-shadow .18s ease; }
.gradio-container button.secondary:hover { transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(28,25,23,.08); }
.gradio-container button.secondary:active { transform: translateY(0); box-shadow: none; }
/* Calm accent focus halo on text / number / color inputs. */
.gradio-container input:focus, .gradio-container textarea:focus {
    box-shadow: 0 0 0 3px var(--ac-weak) !important; outline: none !important; }

@media (prefers-reduced-motion: reduce) {
    #hero, .tabitem, [data-testid="accordion-content"],
    ul.options, .options { animation: none; }
    .gradio-container button.secondary:hover { transform: none; box-shadow: none; } }

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
.pill .dot { width: 7px; height: 7px; border-radius: 999px; background: #22C55E; }

/* top-right utility cluster: Light/Dark + Magnifier toggles, side by side, no
   overlap. The whole row floats (not just one button), so both stay together. */
#topbar { position: absolute; top: 18px; right: 40px; z-index: 50;
    display: flex; flex-wrap: nowrap; gap: 8px; align-items: center;
    width: auto !important; min-width: 0 !important; flex: none !important; }
#topbar button { width: auto !important; min-width: 0 !important;
    flex: 0 0 auto !important; white-space: nowrap; }
/* the gear is icon-only — keep it a tidy square with a slightly larger glyph */
#settings-btn { font-size: 1.05rem !important; line-height: 1;
    padding-left: 11px !important; padding-right: 11px !important; }

/* compact, left-aligned button row (e.g. the Library toolbar) */
.toolbar { gap: 8px; }
.toolbar button { flex: 0 0 auto !important; width: auto !important;
    min-width: 0 !important; }

/* quick-preset chips: content-sized buttons that wrap onto as many lines as
   they need, instead of stretching to fill fixed-count rows */
.preset-row { flex-wrap: wrap; gap: 8px; row-gap: 8px; }
.preset-row button { flex: 0 0 auto !important; width: auto !important;
    min-width: 0 !important; white-space: nowrap; }

/* On long tabs the output column sticks while the settings column scrolls, so
   the result/preview is always in view. */
.sticky-col { position: sticky; top: 16px; align-self: flex-start; }

/* Active magnifier chip: white text fails contrast on the bright dark-mode
   accent — use the same dark ink as primary buttons there. */
.dark body.loupe-on #mag-btn, body.loupe-on .dark #mag-btn {
    color: #06231F !important; }

/* Narrow viewports: the absolute top-right cluster would overlap the hero —
   let it flow in the layout instead. */
@media (max-width: 720px) {
    #topbar { position: static; justify-content: flex-end; margin-top: 4px; } }

/* tab bar: accent the selected tab.
   Gradio measures this strip and moves whatever doesn't fit into a "…" menu —
   and those tabs are NOT rendered in the bar at all, so no amount of
   flex-wrap brings them back. With a dozen-plus tools the only fix is to make
   the buttons narrower, so the whole toolset stays one click away. Measured:
   0.78rem is the largest type that keeps all 18 tools inline down to 1200px. */
.tabitem { padding-top: 28px !important; }
.tab-container button { padding: 0 6px !important; font-size: 0.78rem !important; }

/* section heads: accent eyebrow w/ icon + underlined title */
.sec-head { margin-bottom: 8px; }
.sec-head .eyebrow { display: flex; align-items: center; gap: 6px;
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

/* "Tips" lists inside the collapsed Tips accordions — quiet text, no nested box. */
.notes ul { margin: 2px 0 0; padding-left: 20px; }
.notes li { margin: 3px 0; font-size: 0.88rem; line-height: 1.5;
    color: var(--body-text-color-subdued); }
.notes li strong { color: var(--body-text-color); font-weight: 600; }

/* active quick-preset: a calm accent outline, NOT a second filled button — so
   the real Enhance button stays the only primary action on screen. */
button.preset-active, .preset-active > button {
    border-color: var(--ac) !important; color: var(--ac) !important;
    box-shadow: inset 0 0 0 1px var(--ac) !important; font-weight: 700 !important; }

/* Long file paths in `code` spans must wrap, not get clipped at the block edge
   (e.g. the Settings "Where your files live" paths in a half-width column).
   Gradio's own `.md :not(pre)>code` uses word-break:normal + display:inline-flex
   at higher specificity, so override forcefully so long path tokens break. */
.gradio-container .md :not(pre) > code, .gradio-container :not(pre) > code,
.gradio-container kbd {
    white-space: normal !important; overflow-wrap: anywhere !important;
    word-break: break-word !important; display: inline !important;
    max-width: 100%; }
/* Fenced blocks keep their line breaks (the bare `code` selector above used to
   catch pre > code too, collapsing multi-line commands onto one line). */
.gradio-container pre { overflow-wrap: anywhere; max-width: 100%; }
.gradio-container pre > code { white-space: pre-wrap !important; display: block !important; }

/* Markdown prose must wrap and not be clipped at the block edge — this was
   shaving the first letter off wrapped lines (e.g. the About text). overflow
   visible + a hair of side padding keeps glyphs fully inside. */
.gradio-container .md, .gradio-container .prose { overflow: visible; }
.gradio-container .md p, .gradio-container .prose p,
.gradio-container .md li, .gradio-container .prose li {
    overflow-wrap: break-word; word-break: break-word; padding-inline: 2px; }

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
ICON_LIGHT = _svg('<circle cx="12" cy="12" r="4.5"/><path d="M12 2v2.5M12 19.5V22'
                  'M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8'
                  'M19.1 4.9l-1.8 1.8M6.7 17.3l-1.8 1.8"/>')
ICON_WM = _svg('<rect x="3" y="4.5" width="18" height="15" rx="2"/>'
               '<path d="M8 15.5h8M8 12h5"/>')
ICON_CROP = _svg('<path d="M6.5 2v15.5H22"/><path d="M2 6.5h15.5V22"/>')
# a window: title bar, two dots, and the content area below it
ICON_SHOT = _svg('<rect x="2.5" y="4" width="19" height="16" rx="2.5"/>'
                 '<path d="M2.5 8.5h19"/><circle cx="5.8" cy="6.25" r=".85"/>'
                 '<circle cx="8.6" cy="6.25" r=".85"/>')
# a page with a heading rule and a picture block: a laid-out design
ICON_DESIGN = _svg('<rect x="3.5" y="2.5" width="17" height="19" rx="2"/>'
                   '<path d="M7 7h10"/><path d="M7 10.5h6"/>'
                   '<rect x="7" y="14" width="10" height="4.5" rx="1"/>')
ICON_SHARP = _svg('<path d="M12 3.5 20.5 20.5 12 16 3.5 20.5z"/>')
ICON_FX = _svg('<rect x="2.5" y="5" width="19" height="14" rx="2"/>'
               '<path d="M2.5 9h3M2.5 15h3M18.5 9h3M18.5 15h3M9 5v14M15 5v14"/>')
ICON_BLUR = _svg('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/>'
                 '<path d="M12 3v2M12 19v2M3 12h2M19 12h2"/>')
ICON_STEAM = _svg('<rect x="1.5" y="8" width="3.4" height="8" rx=".8"/>'
                  '<rect x="5.8" y="8" width="3.4" height="8" rx=".8"/>'
                  '<rect x="10.1" y="8" width="3.4" height="8" rx=".8"/>'
                  '<rect x="14.4" y="8" width="3.4" height="8" rx=".8"/>'
                  '<rect x="18.7" y="8" width="3.4" height="8" rx=".8"/>')
ICON_LIBRARY = _svg('<rect x="3" y="3" width="7" height="7" rx="1.5"/>'
                    '<rect x="14" y="3" width="7" height="7" rx="1.5"/>'
                    '<rect x="3" y="14" width="7" height="7" rx="1.5"/>'
                    '<rect x="14" y="14" width="7" height="7" rx="1.5"/>')
ICON_BATCH = _svg('<rect x="8" y="8" width="12" height="12" rx="2"/>'
                  '<path d="M4 16V6a2 2 0 0 1 2-2h10"/>')
ICON_SETTINGS = _svg('<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 '
                     '0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 '
                     '0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 '
                     '1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 '
                     '1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 '
                     '0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 '
                     '0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 '
                     '0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 '
                     '1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 '
                     '2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 '
                     '1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>')

# Step-by-step Windows install guide, shown in the Settings tab.
WINDOWS_GUIDE = """\
Get Upscaler running on a Windows PC — about 10 minutes, all local.

**1 · Install Python**
Download **Python 3.12** from [python.org](https://www.python.org/downloads/windows/),
run the installer, and **tick "Add python.exe to PATH"** before you click Install.
*(Anything from 3.9–3.12 works; 3.12 is the safe pick for the AI libraries.)*

**2 · Install ffmpeg** *(only needed for the Video tab)*
Open **PowerShell** and run:
```powershell
winget install Gyan.FFmpeg
```
Or skip this and let the app install one for you by adding the `video` extra in step 4
(use `".[gui,video]"`).

**3 · Get Upscaler**
Download the project as a ZIP and unzip it (or `git clone` it). Then open
**PowerShell inside that folder**: in File Explorer, Shift-right-click the folder
→ *"Open PowerShell window here"*.

**4 · Create an environment and install**
```powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -e ".[gui]"
```
*If activation is blocked, run `Set-ExecutionPolicy -Scope Process RemoteSigned`
once, then re-run the activate line.*

**5 · Start it**
```powershell
python app.py
```
Open **http://127.0.0.1:7860** in your browser. Press **Ctrl+C** in PowerShell to stop.

---

**Make it faster with your GPU** *(optional)*
- **NVIDIA:** install a CUDA build of PyTorch from
  [pytorch.org/get-started](https://pytorch.org/get-started/locally/), then
  re-run the app — the Device setting "auto" will pick up the GPU.
- **AMD Radeon:** follow the full **`docs/SETUP-WINDOWS-AMD.md`** guide included
  in this project (WSL2 + ROCm, with a DirectML fallback).
- **No GPU?** It still runs on the CPU — slower, but fine for images.

**Good to know**
- Everything runs on your machine; nothing is ever uploaded.
- Your exports are saved to `C:\\Users\\<you>\\.upscaler\\library` (see the **Library** tab).
- Next time: open PowerShell in the folder, run `.venv\\Scripts\\Activate.ps1`, then `python app.py`.
"""


def save_settings(device, model, output_dir):
    """Persist the Settings-tab preferences to ~/.upscaler/config.json AND apply
    them to the live controls (returned as updates), so no restart is needed."""
    out_dir = (output_dir or "").strip()
    ok = config.save(device=device, model=model, output_dir=out_dir)
    if ok:
        status = "✅ Saved — applied now, and used as the defaults from here on."
    else:
        status = "⚠ Couldn't write the settings file — check the folder's permissions."
        return (status,) + (gr.update(),) * 6
    return (
        status,
        gr.update(value=model),    # Upscale tab model
        gr.update(value=device),   # Upscale tab device
        gr.update(value=model),    # Batch model
        gr.update(value=device),   # Batch device
        gr.update(value=device),   # Video device
        gr.update(value=out_dir),  # Lian Li save-to folder
    )


# -- Model Manager (Settings) ------------------------------------------------

def _mm_rows():
    """(dataframe rows, total-usage markdown) for the Model Manager table."""
    rows = []
    for s in manage.list_specs():
        status = "✓ downloaded" if s.present else "— not downloaded"
        size = manage.human_size(s.size_bytes) if s.present else "—"
        rows.append([s.group, s.name, s.filename, status, size])
    total = f"**Total on disk:** {manage.human_size(manage.total_bytes())}"
    return rows, total


def _mm_refresh():
    rows, total = _mm_rows()
    return rows, total


def _mm_download(filename):
    if not filename:
        return _mm_rows() + ("Pick a model to download first.",)
    status = manage.download_one(filename)
    rows, total = _mm_rows()
    return rows, total, status


def _mm_remove(filename):
    if not filename:
        return _mm_rows() + ("Pick a model to remove first.",)
    status = manage.remove_one(filename)
    rows, total = _mm_rows()
    return rows, total, status
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
    _purge_old_exports()
    device_name = resolve_device("auto").type
    # Saved preferences seed the defaults (guarded against stale/invalid values).
    cfg = config.load()
    _cfg_model = cfg["model"] if cfg["model"] in MODELS else "realesrgan-x2plus"
    _cfg_device = cfg["device"] if cfg["device"] in _DEVICES else "auto"
    # fill_width: Gradio otherwise centres the whole app inside a fixed max
    # width, which left ~130px of dead gutter on each side at 1200px and was
    # the real reason the tab bar ran out of room.
    with gr.Blocks(title="Upscaler", fill_width=True) as demo:
        gr.HTML(
            '<div id="hero">'
            f'<div class="brandrow"><span class="logo">{ICON_LOGO}</span>'
            '<span class="brand">Upscaler</span></div>'
            '<div class="sub">Enlarge, restore and colorize photos with AI, then '
            "develop them: color and light, film effects, sharpen, blur, crop, "
            "watermark, beautify a screenshot, drop it into a design template, "
            "strip the GPS location, hit a size limit — or batch a whole folder. "
            "Every tool runs on your own machine; nothing is ever uploaded.</div>"
            f'<span class="pill"><span class="dot"></span>Running locally · {device_name}</span>'
            "</div>"
        )
        with gr.Row(elem_id="topbar"):
            theme_btn = gr.Button(
                "◐ Light / Dark", elem_id="theme-toggle", size="sm", variant="secondary"
            )
            mag_btn = gr.Button(
                "🔍 Magnifier", elem_id="mag-btn", size="sm", variant="secondary",
            )
            settings_btn = gr.Button(
                "⚙", elem_id="settings-btn", size="sm", variant="secondary",
            )
        theme_btn.click(None, js=_TOGGLE_THEME_JS)
        # JS-only toggle: enables the hover loupe over result images
        mag_btn.click(None, js="() => document.body.classList.toggle('loupe-on')")

        with gr.Tabs() as main_tabs:
            # ---- Tab: Upscale & Enhance ----
            with gr.Tab("Upscale"):
                gr.HTML(_section_head(
                    "Enhance", "Upscale & Enhance",
                    "Enlarge and sharpen any image with AI, plus optional cleanup "
                    "for blur and noise. ×2 keeps already-good photos looking "
                    "natural; ×4 adds the most detail but can over-process clean "
                    "images. Start with a preset, then fine-tune.",
                    icon=ICON_AI,
                ))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        inp = gr.Image(
                            label="Input", type="pil",
                            sources=["upload", "clipboard"], height=300,
                            elem_classes="drop", buttons=["download", "fullscreen"],
                        )
                        gr.Markdown("**Quick presets** — a starting point; tweak anything after.")
                        # One wrapping chip row: buttons size to their label and
                        # flow onto as many lines as needed (fixed-count rows
                        # wrap unevenly at narrow widths).
                        with gr.Row(elem_classes="preset-row"):
                            _preset_buttons = [
                                gr.Button(_pname, size="sm", variant="secondary")
                                for _pname in UPSCALE_PRESETS
                            ]
                        preset_info = gr.Markdown()
                        out_size = gr.Dropdown(
                            list(_SIZE_PRESETS) + list(fit.TARGET_PRESETS)
                            + [_EXACT_CUSTOM],
                            value="Model default (×2/×4)",
                            label="Output size", filterable=True,
                            info="Top entries shrink the longest edge and keep the "
                            "shape. The ones with two numbers (3440×1440, phones, "
                            "tablets) land on that exact size — the image is cropped "
                            "to fit the new shape first.",
                        )
                        custom_size = gr.Textbox(
                            value="", visible=False, label="Custom size",
                            placeholder="3440x1440",
                            info="Width × height in pixels. The image is cropped to "
                            "this shape, then enlarged and fitted to it exactly.",
                        )
                        # Presets only — picking one just snaps the slider below,
                        # which is what actually feeds the job.
                        crop_anchor = gr.Dropdown(
                            list(fit.ANCHORS), value="center", visible=False,
                            label="Keep which part", filterable=False,
                            info="Which part of the photo to keep when the crop has "
                            "to cut something — top is usually right for portraits.",
                        )
                        crop_position = gr.Slider(
                            0, 100, value=50, step=1, label="Crop position",
                            info="Slide to choose which part survives the crop — "
                            "0% = top/left edge, 100% = bottom/right edge.",
                            visible=False,
                        )
                        # buttons=[] is this gradio's spelling of "no download
                        # button" — the preview is a throwaway visual aid.
                        crop_preview = gr.Image(
                            label="Crop preview — the bright region is kept",
                            interactive=False, visible=False, buttons=[],
                        )
                        model = gr.Dropdown(
                            _MODEL_CHOICES, value=_cfg_model,
                            label="Upscale model", filterable=True,
                            info="Picks the AI that enlarges your image. Use ×2 for "
                            "already-good photos, ×4 for small or soft ones, or the "
                            "anime model for drawings and line art. Type to filter.",
                        )
                        with gr.Accordion("Which model for what? (best → worst)", open=False):
                            gr.Markdown(
                                "* **Everyday photos** — UltraSharp › ×4 default › "
                                "NMKD-Siax › NMKD-Superscale\n"
                                "* **Portraits / skin** — Remacri › NMKD-Superscale › "
                                "×2 default\n"
                                "* **JPEG / compressed** — UltraSharp › NMKD-Siax › "
                                "×4 default\n"
                                "* **Anime / line art** — Anime (×4) › UltraSharp\n"
                                "* **Already sharp (be gentle)** — ×2 default › "
                                "NMKD-Superscale\n"
                                "* **Max texture & detail** — UltraSharp › ×4 default "
                                "› Remacri\n\n"
                                "*Starting points — results vary by image, so try a "
                                "couple. UltraSharp/Remacri are non-commercial.*",
                                elem_classes="notes",
                            )
                        sharpen = gr.Slider(
                            0.0, 3.0, value=0.0, step=0.1,
                            label="Sharpen edges — 0 = off",
                            info="Crispens edges after enlarging. Keep it low — too "
                            "much adds bright halos (glowing outlines) around edges.",
                        )
                        with gr.Accordion("Clean up — deblur / denoise", open=False):
                            deblur = gr.Checkbox(
                                value=False, label="Clean up before upscaling",
                                info="Tick to clean up the photo (deblur / denoise) "
                                "before enlarging. To only clean it up without "
                                "enlarging, use the Clean-up-only button below.",
                            )
                            deblur_model = gr.Dropdown(
                                _DEBLUR_CHOICES, value="nafnet-sidd-width64",
                                label="Clean-up model", filterable=False,
                                info="SIDD is the safe default — it cleans grain and "
                                "noise. GoPro fixes motion blur only and will wreck "
                                "noisy photos, so use it only for genuine motion blur.",
                            )
                            restore_strength = gr.Slider(
                                0.0, 1.0, value=1.0, step=0.05,
                                label="Clean-up strength",
                                info="How strongly the cleanup is applied. 1 = full "
                                "effect; lower blends the original back in to keep "
                                "more fine detail (and a little noise).",
                            )
                            fbcnn = gr.Checkbox(
                                value=False, label="Remove JPEG artifacts (FBCNN)",
                                info="De-blocks heavily-compressed JPEGs — runs before "
                                "the deblur/denoise above. Independent: tick either, "
                                "both, or neither. Needs the optional \"face\" packages.",
                            )
                            restore_btn = gr.Button(
                                "✨ Clean up only (deblur / denoise · no upscale)",
                                variant="secondary",
                            )
                        with gr.Accordion("Restore faces", open=False):
                            face = gr.Checkbox(
                                value=False, label="Enhance faces",
                                info="After upscaling, detect faces and restore them "
                                "— a big improvement on photos of people. "
                                "Needs the optional \"face\" packages.",
                            )
                            face_model = gr.Dropdown(
                                _FACE_CHOICES, value=DEFAULT_FACE_MODEL,
                                label="Face model", filterable=False,
                                info="GFPGAN is the gentle, natural default. "
                                "CodeFormer is stronger on badly degraded faces and "
                                "lets you trade fidelity vs. quality below.",
                            )
                            face_strength = gr.Slider(
                                0.0, 1.0, value=0.8, step=0.05,
                                label="Face strength",
                                info="How strongly faces are restored. 1 = full "
                                "restoration; lower blends the original face back in "
                                "to keep more likeness.",
                            )
                            face_fidelity = gr.Slider(
                                0.0, 1.0, value=0.5, step=0.05,
                                label="CodeFormer fidelity",
                                info="Only used by CodeFormer. Higher = truer to the "
                                "original face (safer); lower = stronger, freer "
                                "restoration. Ignored by GFPGAN.",
                            )
                        with gr.Accordion("Advanced", open=False):
                            device = gr.Dropdown(
                                _DEVICES, value=_cfg_device, label="Device",
                                filterable=False,
                                info="Where the work runs. \"auto\" uses your "
                                "graphics card (GPU) if it can, otherwise your "
                                "processor (CPU).",
                            )
                            onnx = gr.Checkbox(
                                value=False, label="Alternative speed engine (ONNX)",
                                info="Runs without PyTorch — often faster on a CPU. "
                                "The first run exports the model, so it takes a "
                                "moment.",
                            )
                            tile = gr.Slider(
                                0, 1024, value=512, step=64,
                                label="Tile size (0 = off)",
                                info="Splits big images into chunks so they use less "
                                "memory. Lower this if you hit out-of-memory errors; "
                                "0 turns it off.",
                            )
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "* **Start with a Quick preset**, then fine-tune — "
                                "×2 suits everyday photos, ×4 can over-process clean "
                                "ones.\n"
                                "* **Keep Sharpen near 0** — past ~1.0 you get halos "
                                "(glowing edges).\n"
                                "* **Turn on \"Clean up before upscaling\" only for blurry or "
                                "noisy photos**, and lower the strength (~0.5) for "
                                "faces.\n"
                                "* **Out-of-memory error?** Lower the Tile size "
                                "(under Advanced) first.",
                                elem_classes="notes",
                            )
                        with gr.Row():
                            run = gr.Button(
                                "Enhance", variant="primary", size="lg", scale=3
                            )
                            enh_cancel = gr.Button("✕ Cancel", variant="stop", scale=1)
                            clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1, elem_classes="sticky-col"):
                        out = gr.ImageSlider(
                            label="Before / after — drag the divider to compare",
                            # max_height (not height): a fixed height= crops the
                            # block via overflow:hidden, and gradio 6.15's slider
                            # anchors its two <img> layers differently when
                            # cropped (in-flow base is top-anchored, clipped
                            # layer is absolutely centered), tearing the seam
                            # vertically. max_height scales the image instead,
                            # keeping both layers in identical boxes.
                            type="pil", max_height=300,
                            buttons=["download", "fullscreen"],
                            elem_classes=["loupe"],
                        )
                        info = gr.Markdown()

            # ---- Tab: Colorize (DDColor) ----
            with gr.Tab("Colorize"):
                gr.HTML(_section_head(
                    "Color", "Colorize Photos",
                    "Bring black-and-white or faded photos to life. DDColor predicts "
                    "natural color and keeps every bit of the original detail.",
                    icon=ICON_AI,
                ))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        col_in = gr.Image(
                            label="Input", type="pil",
                            sources=["upload", "clipboard"], height=300,
                            elem_classes="drop", buttons=["download", "fullscreen"],
                        )
                        col_model = gr.Dropdown(
                            _COLORIZE_CHOICES, value=DEFAULT_COLORIZE_MODEL,
                            label="Model", filterable=False,
                            info="DDColor predicts color from the brightness of your "
                            "photo, so detail is never lost.",
                        )
                        col_strength = gr.Slider(
                            0.0, 1.0, value=1.0, step=0.05, label="Color strength",
                            info="1 = full color; lower keeps it subtler. 0 returns the "
                            "original in grayscale.",
                        )
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "* **Any photo works** — black-and-white, sepia, or "
                                "faded.\n"
                                "* **Detail is preserved** — only color is added.\n"
                                "* **Needs the \"face\" packages** (spandrel).\n"
                                "* **First run downloads ~870MB**, then it's cached.",
                                elem_classes="notes",
                            )
                        with gr.Row():
                            col_btn = gr.Button("Colorize", variant="primary",
                                                size="lg", scale=3)
                            col_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1):
                        col_out = gr.ImageSlider(
                            # max_height, not height — see `out` slider above.
                            label="Before / after", type="pil", max_height=300,
                            elem_classes=["loupe"],
                        )
                        col_info = gr.Markdown()

            # ---- Tab: Remove objects / inpaint (LaMa) ----
            with gr.Tab("Objects"):
                gr.HTML(_section_head(
                    "Erase", "Remove Objects",
                    "Paint over anything you want gone — a photobomber, a sign, a "
                    "blemish — and LaMa fills the gap from the surroundings.",
                    icon=ICON_AI,
                ))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        ip_editor = gr.ImageEditor(
                            label="Paint over the object to remove", type="pil",
                            height=360, sources=["upload", "clipboard"],
                            layers=False, transforms=(),
                            brush=gr.Brush(
                                colors=["#ffffff"], color_mode="fixed", default_size=24
                            ),
                        )
                        ip_model = gr.Dropdown(
                            _INPAINT_CHOICES, value=DEFAULT_INPAINT_MODEL,
                            label="Model", filterable=False,
                            info="Big-LaMa fills the painted region from the "
                            "surrounding image.",
                        )
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "* **Cover the whole object**, plus a little margin.\n"
                                "* **Use a bigger brush** for bigger objects.\n"
                                "* **Best on clutter / backgrounds** — wires, signs, "
                                "blemishes, photobombers.\n"
                                "* **First run downloads ~196MB**, then it's cached.",
                                elem_classes="notes",
                            )
                        with gr.Row():
                            ip_btn = gr.Button("Remove object", variant="primary",
                                               size="lg", scale=3)
                            ip_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1):
                        ip_out = gr.ImageSlider(
                            # max_height, not height — see `out` slider above.
                            label="Before / after", type="pil", max_height=360,
                            elem_classes=["loupe"],
                        )
                        ip_info = gr.Markdown()

            # ---- Tab: Remove background ----
            with gr.Tab("Remove BG"):
                gr.HTML(_section_head(
                    "Cut-out", "Remove Background",
                    "Lift the subject cleanly off its background with AI and save a "
                    "transparent PNG. It drops straight into the Lian Li tab "
                    "as a sticker.",
                    icon=ICON_AI,
                ))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        bg_in = gr.Image(
                            label="Input", type="pil",
                            sources=["upload", "clipboard"], height=300,
                            elem_classes="drop", buttons=["download", "fullscreen"],
                        )
                        bg_model = gr.Dropdown(
                            _BG_CHOICES, value=background.DEFAULT_BG_MODEL,
                            label="Model", filterable=False,
                            info="The AI that finds your subject. u2net is the best "
                            "all-rounder; u2netp is lighter and faster but a bit "
                            "less precise.",
                        )
                        bg_feather = gr.Slider(
                            0, 10, value=1, step=1, label="Edge feather (px)",
                            info="Softens the cut-out edge so it blends in. 0 = a "
                            "hard, crisp edge.",
                        )
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "* **Try u2net first** — best all-rounder; u2netp "
                                "is lighter and faster.\n"
                                "* **1–2px of edge feather** looks most natural; use "
                                "0 for a crisp, hard edge.\n"
                                "* **The result is a transparent PNG** — the "
                                "checkerboard just shows where it's see-through.\n"
                                "* **Pairs with the Lian Li tab** — drop the cut-out "
                                "in as a sticker.",
                                elem_classes="notes",
                            )
                        with gr.Row():
                            bg_btn = gr.Button("Remove background", variant="primary",
                                               size="lg", scale=3)
                            bg_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1):
                        bg_preview = gr.Image(
                            label="Cut-out (checkerboard = transparent)",
                            height=300, buttons=["fullscreen"], elem_classes=["loupe"],
                        )
                        bg_file = gr.File(label="Download transparent PNG")
                        bg_info = gr.Markdown()

            # ---- Tab: Color & light (no AI) ----
            with gr.Tab("Color & Light"):
                gr.HTML(_section_head(
                    "Develop", "Color & Light",
                    "Exposure, contrast, highlights and shadows, white balance, "
                    "vibrance, black & white — the everyday photo adjustments, with "
                    "one-click Auto, a shelf of looks, and the option to apply them "
                    "to just a shape, a graduated band or wherever you paint.",
                    icon=ICON_LIGHT,
                ))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        ad_in = gr.Image(
                            label="Input", type="pil", image_mode=None,
                            sources=["upload", "clipboard"], height=300,
                            elem_classes="drop", buttons=["download", "fullscreen"],
                        )
                        with gr.Row():
                            ad_preset = gr.Dropdown(
                                adjust.PRESET_NAMES, value=adjust.PRESET_NONE,
                                label="Look", filterable=False, scale=3,
                                info="A starting point — every slider stays editable "
                                "afterwards. 'None' resets them all.",
                            )
                            ad_auto = gr.Button("✨ Auto", variant="secondary", scale=1)
                        with gr.Accordion("Light", open=True):
                            ad_exposure = gr.Slider(
                                -100, 100, value=0, step=1, label="Exposure",
                                info="Overall brightness, like changing the shutter "
                                "speed. ±100 is two stops.",
                            )
                            ad_contrast = gr.Slider(
                                -100, 100, value=0, step=1, label="Contrast",
                                info="Pushes lights and darks apart.",
                            )
                            with gr.Row():
                                ad_highlights = gr.Slider(
                                    -100, 100, value=0, step=1, label="Highlights",
                                    info="Negative rescues blown-out brights; positive "
                                    "lifts them.",
                                )
                                ad_shadows = gr.Slider(
                                    -100, 100, value=0, step=1, label="Shadows",
                                    info="Positive opens up dark areas; negative "
                                    "deepens them.",
                                )
                            with gr.Row():
                                ad_black = gr.Slider(
                                    0, 50, value=0, step=0.5, label="Black point",
                                    info="Where black starts — raise it for deeper "
                                    "blacks, or to cut haze.",
                                )
                                ad_white = gr.Slider(
                                    50, 100, value=100, step=0.5, label="White point",
                                    info="Where white starts — lower it for a brighter, "
                                    "punchier picture.",
                                )
                            with gr.Row():
                                ad_gamma = gr.Slider(
                                    0.2, 3, value=1, step=0.01, label="Midtones (gamma)",
                                    info="Brightens or darkens the middle tones without "
                                    "moving black or white.",
                                )
                                ad_clarity = gr.Slider(
                                    -100, 100, value=0, step=1, label="Clarity",
                                    info="Local contrast — positive adds punch and "
                                    "texture, negative softens.",
                                )
                        with gr.Accordion("Color", open=True):
                            with gr.Row():
                                ad_temp = gr.Slider(
                                    -100, 100, value=0, step=1, label="Temperature",
                                    info="Cooler (blue) to warmer (orange).",
                                )
                                ad_tint = gr.Slider(
                                    -100, 100, value=0, step=1, label="Tint",
                                    info="Green to magenta — fixes the cast fluorescent "
                                    "light leaves behind.",
                                )
                            with gr.Row():
                                ad_vibrance = gr.Slider(
                                    -100, 100, value=0, step=1, label="Vibrance",
                                    info="Boosts the muted colors and leaves the already "
                                    "vivid ones (and skin) alone.",
                                )
                                ad_saturation = gr.Slider(
                                    -100, 100, value=0, step=1, label="Saturation",
                                    info="Boosts every color equally. -100 is greyscale.",
                                )
                            ad_hue = gr.Slider(
                                -180, 180, value=0, step=1, label="Hue shift (°)",
                                info="Rotates every color around the color wheel.",
                            )
                        with gr.Accordion("Black & white", open=False):
                            ad_mono = gr.Checkbox(
                                value=False, label="Convert to black & white",
                                info="Uses the mix below, so you can decide how bright "
                                "each original color comes out.",
                            )
                            with gr.Row():
                                ad_mr = gr.Slider(0, 100, value=30, step=1, label="Red mix",
                                                  info="How bright reds become.")
                                ad_mg = gr.Slider(0, 100, value=59, step=1, label="Green mix",
                                                  info="How bright greens become.")
                                ad_mb = gr.Slider(0, 100, value=11, step=1, label="Blue mix",
                                                  info="How bright blues become — lower it "
                                                  "to darken skies.")
                            with gr.Row():
                                ad_tone = gr.ColorPicker(
                                    value="#d8b070", label="Tone color",
                                    info="The tint for a sepia or cyanotype look.",
                                )
                                ad_tone_strength = gr.Slider(
                                    0, 100, value=0, step=1, label="Tone strength",
                                    info="How strongly the tone color is mixed in. "
                                    "0 = plain black & white.",
                                )
                        with gr.Accordion("Where to apply", open=False):
                            ad_shape = gr.Radio(
                                blur.SHAPES, value="whole", label="Region",
                                info="whole = the entire photo · rectangle / ellipse = a "
                                "shape you position · band = a straight strip, which with "
                                "'outside' is a graduated filter for skies · painted = "
                                "wherever you brush · faces = every face found "
                                "automatically.",
                            )
                            with gr.Row():
                                ad_x = gr.Slider(0, 100, value=50, step=1, label="Centre X (%)",
                                                 visible=False, info="Shape position.")
                                ad_y = gr.Slider(0, 100, value=50, step=1, label="Centre Y (%)",
                                                 visible=False, info="Shape position.")
                            with gr.Row():
                                ad_w = gr.Slider(1, 100, value=50, step=1, label="Width (%)",
                                                 visible=False, info="Shape size.")
                                ad_h = gr.Slider(1, 100, value=50, step=1, label="Height (%)",
                                                 visible=False, info="Shape size, or how "
                                                 "thick the band is.")
                            ad_mangle = gr.Slider(-90, 90, value=0, step=1, label="Band tilt (°)",
                                                  visible=False, info="Rotate the strip.")
                            ad_round = gr.Slider(0, 100, value=0, step=1,
                                                 label="Corner roundness (%)", visible=False,
                                                 info="0 = sharp corners, 100 = a pill.")
                            ad_feather = gr.Slider(0, 50, value=15, step=0.5, label="Feather (%)",
                                                   visible=False,
                                                   info="How softly the adjustment fades out "
                                                   "at the edge of the region.")
                            ad_outside = gr.Checkbox(value=False, label="Apply outside the shape",
                                                     visible=False,
                                                     info="Adjust everything except the shape.")
                            ad_facepad = gr.Slider(
                                -25, 100, value=25, step=1, label="Face padding (%)",
                                visible=False,
                                info="Grows the oval around each detected face — more "
                                "covers hair and chin, less keeps it tight.",
                            )
                            ad_faces_note = gr.Markdown(visible=False, elem_classes="notes")
                            ad_faces_state = gr.State([])
                            ad_editor = gr.ImageEditor(
                                label="Paint where to adjust", type="pil", height=360,
                                sources=[], layers=False, transforms=(), visible=False,
                                brush=gr.Brush(colors=["#ffffff"], color_mode="fixed",
                                               default_size=40),
                            )
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "* **Start with Auto**, then fine-tune. It sets the "
                                "black and white points, midtones and white balance "
                                "from the photo itself.\n"
                                "* **Vibrance before saturation** for people — it "
                                "leaves skin tones alone.\n"
                                "* **Brighten just the faces** with region = faces — "
                                "it finds them, you lift the exposure.\n"
                                "* **Darken a bright sky** with region = band, "
                                "'apply outside' off, a big feather, and negative "
                                "exposure. That's a graduated filter.\n"
                                "* **Rescue a backlit photo** with Shadows up and "
                                "Highlights down, rather than exposure.\n"
                                "* **Black & white:** drop the blue mix to darken a "
                                "sky, raise the red mix to brighten skin.\n"
                                "* **Stack it:** apply, then 'Use as input' to adjust "
                                "a second region differently.",
                                elem_classes="notes",
                            )
                        with gr.Row():
                            ad_btn = gr.Button("Apply (full size)", variant="primary",
                                               size="lg", scale=3)
                            ad_use = gr.Button("↪ Use as input", variant="secondary", scale=2)
                            ad_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1, elem_classes="sticky-col"):
                        ad_preview = gr.ImageSlider(
                            # max_height, not height — see `out` slider above.
                            label="Live preview — before / after (drag the divider)",
                            type="pil", max_height=340, elem_classes=["loupe"],
                        )
                        ad_out = gr.ImageSlider(
                            label="Result at full size — before / after", type="pil",
                            max_height=340, elem_classes=["loupe"], interactive=False,
                        )
                        ad_file = gr.File(label="Download PNG")
                        ad_info = gr.Markdown()

            # ---- Tab: Effects & film looks (no AI) ----
            with gr.Tab("Effects"):
                gr.HTML(_section_head(
                    "Looks", "Effects & film looks",
                    "Grain, halation, light leaks, vignettes, duotone, halftone, "
                    "dithering, scanlines and glitch — stack as many as you like, or "
                    "start from a look. Everything is sized relative to the photo, so "
                    "it looks the same at any resolution.",
                    icon=ICON_FX,
                ))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        fx_in = gr.Image(
                            label="Input", type="pil", image_mode=None,
                            sources=["upload", "clipboard"], height=300,
                            elem_classes="drop", buttons=["download", "fullscreen"],
                        )
                        fx_preset = gr.Dropdown(
                            effects.PRESET_NAMES, value=effects.PRESET_NONE,
                            label="Look", filterable=False,
                            info="A ready-made stack — every slider below stays editable "
                            "afterwards. 'None' clears them all.",
                        )
                        with gr.Accordion("Film", open=True):
                            with gr.Row():
                                fx_grain = gr.Slider(
                                    0, 100, value=0, step=1, label="Grain",
                                    info="Film grain, strongest in the midtones.",
                                )
                                fx_grain_size = gr.Slider(
                                    1, 6, value=1, step=0.1, label="Grain size",
                                    info="1 is fine, per-pixel grain; higher clumps it "
                                    "into coarser, older-film specks.",
                                )
                            fx_halation = gr.Slider(
                                0, 100, value=0, step=1, label="Halation / glow",
                                info="Light bleeding out of bright areas, the way it "
                                "does on film.",
                            )
                            with gr.Row():
                                fx_hal_thresh = gr.Slider(
                                    0, 99, value=65, step=1, label="Glow threshold",
                                    info="How bright a pixel must be before it glows. "
                                    "Lower spreads the glow further into the picture.",
                                )
                                fx_hal_radius = gr.Slider(
                                    0.2, 10, value=2, step=0.1, label="Glow radius (%)",
                                    info="How far the glow spreads, relative to the photo.",
                                )
                            fx_hal_color = gr.ColorPicker(
                                value="#ff5522", label="Glow color",
                                info="Classic film halation is warm orange-red; a pale "
                                "cream gives a dreamy white glow.",
                            )
                            fx_leak = gr.Slider(
                                0, 100, value=0, step=1, label="Light leak",
                                info="A colored wash across the frame, like light "
                                "catching the film.",
                            )
                            with gr.Row():
                                fx_leak_angle = gr.Slider(
                                    0, 360, value=45, step=1, label="Leak direction (°)",
                                    info="Which side the light comes from.",
                                )
                                fx_leak_soft = gr.Slider(
                                    1, 100, value=60, step=1, label="Leak softness",
                                    info="Low keeps it to one edge; high washes the "
                                    "whole frame.",
                                )
                            fx_leak_color = gr.ColorPicker(value="#ff8a3d", label="Leak color",
                                                           info="The color of the wash.")
                        with gr.Accordion("Lens", open=False):
                            fx_vignette = gr.Slider(
                                -100, 100, value=0, step=1, label="Vignette",
                                info="Positive darkens the corners, negative brightens "
                                "them.",
                            )
                            with gr.Row():
                                fx_vig_radius = gr.Slider(
                                    0, 99, value=60, step=1, label="Vignette size (%)",
                                    info="How much of the middle stays untouched.",
                                )
                                fx_vig_feather = gr.Slider(
                                    1, 100, value=50, step=1, label="Vignette softness",
                                    info="How gradually it fades in toward the corners.",
                                )
                            fx_aberration = gr.Slider(
                                0, 100, value=0, step=1, label="Chromatic aberration",
                                info="Red and blue fringing toward the corners, like a "
                                "cheap lens.",
                            )
                        with gr.Accordion("Print", open=False):
                            fx_duotone = gr.Slider(
                                0, 100, value=0, step=1, label="Duotone",
                                info="Recolors the photo between two colors by "
                                "brightness.",
                            )
                            with gr.Row():
                                fx_duo_dark = gr.ColorPicker(value="#1b2a4a", label="Shadow color",
                                                             info="What the dark areas become.")
                                fx_duo_light = gr.ColorPicker(value="#ffd9a0", label="Highlight color",
                                                              info="What the bright areas become.")
                            fx_posterize = gr.Slider(
                                0, 32, value=0, step=1, label="Posterize levels",
                                info="Flattens the photo into this many brightness steps "
                                "per color. 0 = off.",
                            )
                            with gr.Row():
                                fx_dither = gr.Slider(
                                    0, 100, value=0, step=1, label="Dither",
                                    info="Ordered dithering — retro computer graphics.",
                                )
                                fx_dither_levels = gr.Slider(
                                    2, 16, value=4, step=1, label="Dither levels",
                                    info="Colors per channel. 2 gives the classic 8-color "
                                    "look; pair with black & white for true 1-bit.",
                                )
                            fx_halftone = gr.Slider(
                                0, 100, value=0, step=1, label="Halftone",
                                info="A dot screen, like newspaper or comic printing.",
                            )
                            with gr.Row():
                                fx_ht_cell = gr.Slider(
                                    0.2, 5, value=1, step=0.1, label="Dot size (%)",
                                    info="Bigger dots read as a comic; small ones as "
                                    "newsprint.",
                                )
                                fx_ht_angle = gr.Slider(
                                    0, 90, value=45, step=1, label="Screen angle (°)",
                                    info="The angle of the dot grid. 45 is traditional.",
                                )
                        with gr.Accordion("Screen", open=False):
                            with gr.Row():
                                fx_scanlines = gr.Slider(
                                    0, 100, value=0, step=1, label="Scanlines",
                                    info="Dark horizontal lines, like an old CRT.",
                                )
                                fx_scan_spacing = gr.Slider(
                                    1, 20, value=3, step=0.5, label="Line spacing",
                                    info="How far apart the lines sit, relative to the "
                                    "photo.",
                                )
                            with gr.Row():
                                fx_glitch = gr.Slider(
                                    0, 100, value=0, step=1, label="Glitch",
                                    info="Displaced bands and torn color channels, like "
                                    "damaged tape.",
                                )
                                fx_glitch_seed = gr.Slider(
                                    0, 999, value=7, step=1, label="Glitch seed",
                                    info="Change it for a different random tear; the same "
                                    "seed always gives the same one.",
                                )
                        with gr.Accordion("Where to apply", open=False):
                            fx_shape = gr.Radio(
                                blur.SHAPES, value="whole", label="Region",
                                info="whole = the entire photo · rectangle / ellipse = a "
                                "shape you position · band = a straight strip · painted = "
                                "wherever you brush · faces = every face found "
                                "automatically.",
                            )
                            with gr.Row():
                                fx_x = gr.Slider(0, 100, value=50, step=1, label="Centre X (%)",
                                                 visible=False, info="Shape position.")
                                fx_y = gr.Slider(0, 100, value=50, step=1, label="Centre Y (%)",
                                                 visible=False, info="Shape position.")
                            with gr.Row():
                                fx_w = gr.Slider(1, 100, value=50, step=1, label="Width (%)",
                                                 visible=False, info="Shape size.")
                                fx_h = gr.Slider(1, 100, value=50, step=1, label="Height (%)",
                                                 visible=False, info="Shape size, or how "
                                                 "thick the band is.")
                            fx_mangle = gr.Slider(-90, 90, value=0, step=1, label="Band tilt (°)",
                                                  visible=False, info="Rotate the strip.")
                            fx_round = gr.Slider(0, 100, value=0, step=1,
                                                 label="Corner roundness (%)", visible=False,
                                                 info="0 = sharp corners, 100 = a pill.")
                            fx_feather = gr.Slider(0, 50, value=15, step=0.5, label="Feather (%)",
                                                   visible=False,
                                                   info="How softly the effects fade out at "
                                                   "the edge of the region.")
                            fx_outside = gr.Checkbox(value=False, label="Apply outside the shape",
                                                     visible=False,
                                                     info="Affect everything except the shape.")
                            fx_facepad = gr.Slider(
                                -25, 100, value=25, step=1, label="Face padding (%)",
                                visible=False,
                                info="Grows the oval around each detected face.",
                            )
                            fx_faces_note = gr.Markdown(visible=False, elem_classes="notes")
                            fx_faces_state = gr.State([])
                            fx_editor = gr.ImageEditor(
                                label="Paint where to apply", type="pil", height=360,
                                sources=[], layers=False, transforms=(), visible=False,
                                brush=gr.Brush(colors=["#ffffff"], color_mode="fixed",
                                               default_size=40),
                            )
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "* **Start from a look**, then dial the sliders back — "
                                "most effects read best at half the strength you first "
                                "reach for.\n"
                                "* **Grain size matters more than amount** for an old-film "
                                "feel: 2–3 with a modest amount beats fine grain turned "
                                "up.\n"
                                "* **Halation needs highlights** — if nothing glows, lower "
                                "the glow threshold.\n"
                                "* **Grade first, then add effects**: set the mood in "
                                "Color & Light, then 'Use as input' here.\n"
                                "* **For a true 1-bit look**, make it black & white in "
                                "Color & Light first, then dither with 2 levels.\n"
                                "* **Effects can be local too** — a halftone in just a "
                                "painted area, or grain everywhere except the faces.",
                                elem_classes="notes",
                            )
                        with gr.Row():
                            fx_btn = gr.Button("Apply (full size)", variant="primary",
                                               size="lg", scale=3)
                            fx_use = gr.Button("↪ Use as input", variant="secondary", scale=2)
                            fx_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1, elem_classes="sticky-col"):
                        fx_preview = gr.ImageSlider(
                            # max_height, not height — see `out` slider above.
                            label="Live preview — before / after (drag the divider)",
                            type="pil", max_height=340, elem_classes=["loupe"],
                        )
                        fx_out = gr.ImageSlider(
                            label="Result at full size — before / after", type="pil",
                            max_height=340, elem_classes=["loupe"], interactive=False,
                        )
                        fx_file = gr.File(label="Download PNG")
                        fx_info = gr.Markdown()

            # ---- Tab: Blur toolbox (no AI) ----
            with gr.Tab("Blur"):
                gr.HTML(_section_head(
                    "Blur", "Blur toolbox",
                    "Eight kinds of blur — soft, motion, spin, zoom, lens bokeh, "
                    "pixelate, edge-keeping surface blur — over the whole photo or "
                    "just a shape, a tilt-shift band or wherever you paint, with "
                    "feathered edges and a live before / after.",
                    icon=ICON_BLUR,
                ))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        bl_in = gr.Image(
                            label="Input", type="pil", image_mode=None,
                            sources=["upload", "clipboard"], height=300,
                            elem_classes="drop", buttons=["download", "fullscreen"],
                        )
                        bl_kind = gr.Radio(
                            blur.KINDS, value="gaussian", label="Blur type",
                            info=_BLUR_KIND_INFO["gaussian"],
                        )
                        bl_strength = gr.Slider(
                            0, 100, value=30, step=1, label="Strength",
                            info="Relative to the picture: 100 = a radius of 10% of "
                            "its short side, so it looks the same at any resolution.",
                        )
                        bl_angle = gr.Slider(
                            0, 180, value=0, step=1, label="Direction (°)", visible=False,
                            info="Which way the streaks run — 0 = horizontal, 90 = vertical.",
                        )
                        with gr.Row():
                            bl_cx = gr.Slider(0, 100, value=50, step=1, label="Centre X (%)",
                                              visible=False, info="Where the spin / zoom radiates from.")
                            bl_cy = gr.Slider(0, 100, value=50, step=1, label="Centre Y (%)",
                                              visible=False, info="Where the spin / zoom radiates from.")
                        bl_highlights = gr.Slider(
                            0, 100, value=0, step=1, label="Highlight bloom", visible=False,
                            info="Lifts bright points into glowing discs, like a real lens.",
                        )
                        bl_threshold = gr.Slider(
                            0, 100, value=25, step=1, label="Edge protection", visible=False,
                            info="How strong an edge must be to stay sharp — lower keeps "
                            "more detail, higher smooths more.",
                        )
                        with gr.Accordion("Where to blur", open=True):
                            bl_shape = gr.Radio(
                                blur.BLUR_SHAPES, value="whole", label="Region",
                                info="whole = everything · rectangle / ellipse = a shape "
                                "you position · band = a straight strip (tilt-shift) · "
                                "painted = wherever you brush · faces = every face found "
                                "automatically, for privacy · depth = a real depth of "
                                "field, sharp at one distance and soft beyond it.",
                            )
                            with gr.Row():
                                bl_x = gr.Slider(0, 100, value=50, step=1, label="Centre X (%)",
                                                 visible=False, info="Shape position.")
                                bl_y = gr.Slider(0, 100, value=50, step=1, label="Centre Y (%)",
                                                 visible=False, info="Shape position.")
                            with gr.Row():
                                bl_w = gr.Slider(1, 100, value=50, step=1, label="Width (%)",
                                                 visible=False, info="Shape size.")
                                bl_h = gr.Slider(1, 100, value=50, step=1, label="Height (%)",
                                                 visible=False, info="Shape size, or how thick the band is.")
                            bl_mangle = gr.Slider(
                                -90, 90, value=0, step=1, label="Band tilt (°)", visible=False,
                                info="Rotate the strip — 0 = horizontal.",
                            )
                            bl_round = gr.Slider(
                                0, 100, value=0, step=1, label="Corner roundness (%)", visible=False,
                                info="0 = sharp corners, 100 = a pill.",
                            )
                            bl_feather = gr.Slider(
                                0, 50, value=10, step=0.5, label="Feather (%)", visible=False,
                                info="How soft the edge of the region is, relative to the "
                                "picture's short side.",
                            )
                            with gr.Row():
                                bl_outside = gr.Checkbox(
                                    value=False, label="Blur outside the shape", visible=False,
                                    info="Keep the shape sharp and blur everything else "
                                    "(tilt-shift, focus on a subject).",
                                )
                                bl_progressive = gr.Checkbox(
                                    value=True, label="Graded edge", visible=False,
                                    info="Ramp the blur up through the feather (half → full) "
                                    "instead of cross-fading one blur — smoother tilt-shift.",
                                )
                            bl_facepad = gr.Slider(
                                -25, 100, value=25, step=1, label="Face padding (%)",
                                visible=False,
                                info="Grows the oval around each detected face — more "
                                "covers hair and chin, less keeps it tight.",
                            )
                            bl_faces_note = gr.Markdown(visible=False, elem_classes="notes")
                            bl_faces_state = gr.State([])
                            bl_depth_view = gr.Image(
                                label="Depth map — click where you want it sharp",
                                visible=False, height=220, elem_classes=["loupe"],
                                interactive=False,
                            )
                            bl_depth_note = gr.Markdown(visible=False, elem_classes="notes")
                            bl_depth_state = gr.State(None)
                            bl_focus = gr.Slider(
                                0, 100, value=70, step=0.1, label="Focus distance",
                                visible=False,
                                info="Which distance stays sharp: 100 is the nearest "
                                "thing in the frame, 0 the farthest. Clicking the depth "
                                "map sets this for you.",
                            )
                            bl_dof = gr.Slider(
                                0, 100, value=25, step=1, label="Depth of field",
                                visible=False,
                                info="How deep the sharp zone runs. Small is a wide "
                                "aperture with only your subject sharp; large keeps "
                                "most of the scene in focus.",
                            )
                            bl_editor = gr.ImageEditor(
                                label="Paint where to blur", type="pil", height=360,
                                sources=[], layers=False, transforms=(), visible=False,
                                brush=gr.Brush(colors=["#ffffff"], color_mode="fixed",
                                               default_size=40),
                            )
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "* **Hide every face at once:** region = faces, "
                                "pixelate, strength 45+. It finds them for you.\n"
                                "* **Portrait mode:** region = faces with 'blur outside' "
                                "and lens blur — the face stays sharp, the background "
                                "goes soft.\n"
                                "* **A real depth of field:** region = depth, then click "
                                "your subject on the depth map. Unlike the face trick "
                                "this softens things by how far away they are, so a "
                                "distant wall blurs more than a nearby one.\n"
                                "* **Hide a plate or a sign:** pixelate + ellipse, "
                                "strength 50+, a little feather.\n"
                                "* **Tilt-shift / miniature look:** gaussian or lens + "
                                "band, tick 'blur outside', feather 15–25, graded edge on.\n"
                                "* **Make the subject pop:** lens with some highlight "
                                "bloom + ellipse around the subject, 'blur outside'.\n"
                                "* **Speed:** motion blur along the direction of travel, "
                                "with the subject painted out (painted region + 'blur "
                                "outside').\n"
                                "* **Smooth skin or noise without mush:** surface blur, "
                                "strength 15–30, then raise edge protection until edges "
                                "come back.\n"
                                "* **Stack effects:** apply, then 'Use result as input' "
                                "and blur another region.",
                                elem_classes="notes",
                            )
                        with gr.Row():
                            bl_btn = gr.Button("Apply blur (full size)", variant="primary",
                                               size="lg", scale=3)
                            bl_use = gr.Button("↪ Use as input", variant="secondary", scale=2)
                            bl_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1, elem_classes="sticky-col"):
                        bl_preview = gr.ImageSlider(
                            # max_height, not height — see `out` slider above.
                            label="Live preview — before / after (drag the divider)",
                            type="pil", max_height=340, elem_classes=["loupe"],
                        )
                        bl_out = gr.ImageSlider(
                            label="Result at full size — before / after", type="pil",
                            max_height=340, elem_classes=["loupe"],
                            # It's also read by "Use as input", which would otherwise
                            # make Gradio render it as an upload dropzone when empty.
                            interactive=False,
                        )
                        bl_file = gr.File(label="Download PNG")
                        bl_info = gr.Markdown()

            # ---- Tab: Watermark (no AI) ----
            with gr.Tab("Watermark"):
                gr.HTML(_section_head(
                    "Credit", "Watermark",
                    "Sign your work with text or a logo — in a corner, tiled across the "
                    "whole frame for a proof copy, or **behind the subject**, so a "
                    "headline passes behind the person in the photo.",
                    icon=ICON_WM,
                ))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        wm_in = gr.Image(
                            label="Input", type="pil", image_mode=None,
                            sources=["upload", "clipboard"], height=280,
                            elem_classes="drop", buttons=["download", "fullscreen"],
                        )
                        wm_preset = gr.Dropdown(
                            wm_tools.PRESET_NAMES, value=wm_tools.PRESET_NONE,
                            label="Preset", filterable=False,
                            info="A starting point — everything stays editable. 'None' "
                            "clears the watermark.",
                        )
                        wm_kind = gr.Radio(
                            wm_tools.KINDS, value="text", label="Mark",
                            info="Your name or a caption, or an image such as a logo.",
                        )
                        wm_text = gr.Textbox(
                            value="© Your Name", label="Text", lines=2,
                            info="What to stamp on the photo.",
                        )
                        with gr.Row():
                            wm_font = gr.Dropdown(
                                wm_tools.FONT_NAMES, value=wm_tools.DEFAULT_FONT,
                                label="Font", filterable=True, info="The typeface.",
                            )
                            wm_size = gr.Slider(
                                0.5, 30, value=4, step=0.1, label="Text size (%)",
                                info="As a share of the photo's short side, so it looks "
                                "the same on every picture.",
                            )
                        with gr.Row():
                            wm_color = gr.ColorPicker(value="#ffffff", label="Text color")
                            wm_outline = gr.ColorPicker(value="#000000", label="Outline color",
                                                        info="Keeps white text readable on "
                                                        "a bright sky.")
                        with gr.Row():
                            wm_outline_w = gr.Slider(
                                0, 30, value=8, step=0.5, label="Outline width (%)",
                                info="Thickness of the outline, relative to the text.",
                            )
                            wm_shadow = gr.Slider(
                                0, 100, value=45, step=1, label="Shadow",
                                info="A soft drop shadow behind the text.",
                            )
                        wm_logo = gr.Image(
                            label="Logo image (a transparent PNG works best)", type="pil",
                            image_mode="RGBA", sources=["upload", "clipboard"], height=140,
                            visible=False,
                        )
                        wm_logo_scale = gr.Slider(
                            1, 100, value=18, step=0.5, label="Logo size (%)", visible=False,
                            info="Width of the logo as a share of the photo's width.",
                        )
                        wm_position = gr.Dropdown(
                            wm_tools.POSITIONS, value=wm_tools.DEFAULT_POSITION,
                            label="Position", filterable=False,
                            info="Where the mark sits — or 'tiled' to repeat it across "
                            "the whole frame, which survives being cropped out.",
                        )
                        with gr.Row():
                            wm_margin = gr.Slider(
                                0, 25, value=3, step=0.5, label="Margin (%)",
                                info="Distance from the edge.",
                            )
                            wm_opacity = gr.Slider(
                                0, 100, value=70, step=1, label="Opacity",
                                info="How strongly the mark shows.",
                            )
                        wm_rotation = gr.Slider(
                            -180, 180, value=0, step=1, label="Rotation (°)",
                            info="Tilt the mark.",
                        )
                        wm_behind = gr.Checkbox(
                            value=False, label="Put it behind the subject",
                            info="Cuts the subject out and lays the text on the "
                            "background, so a headline passes behind a person. Works "
                            "with any position, tiled included.",
                        )
                        wm_behind_note = gr.Markdown(visible=False, elem_classes="notes")
                        wm_cutout_state = gr.State(None)
                        with gr.Row():
                            wm_cut_feather = gr.Slider(
                                0, 10, value=1, step=1, label="Cut-out edge softness (px)",
                                visible=False,
                                info="Softens the edge where the subject meets the text. "
                                "1 or 2 hides a ragged cut-out.",
                            )
                            wm_subject_shadow = gr.Slider(
                                0, 100, value=35, step=1, label="Subject shadow",
                                visible=False,
                                info="A soft shadow from the subject onto the text. "
                                "Without it the text reads as pasted under a sticker.",
                            )
                        with gr.Row():
                            wm_tile_gap = gr.Slider(
                                2, 50, value=14, step=0.5, label="Tile spacing (%)",
                                visible=False, info="Gap between repeats.",
                            )
                            wm_tile_angle = gr.Slider(
                                -90, 90, value=30, step=1, label="Tile angle (°)",
                                visible=False, info="Angle of the repeating pattern.",
                            )
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "* **Keep the outline on** — white text vanishes on a "
                                "bright sky without it.\n"
                                "* **'tiled' is the proof watermark**: it can't be "
                                "cropped off, so use a low opacity and let it sit over "
                                "the whole picture.\n"
                                "* **A transparent PNG logo** composites cleanly; the "
                                "Remove BG tab will make you one.\n"
                                "* **Sizes are relative**, so the same settings suit a "
                                "phone snap and a 4K frame — batch a whole folder from "
                                "the command line with `upscaler watermark`.\n"
                                "* **Watermark last**, after cropping and colour, or the "
                                "crop may cut your signature off.\n"
                                "* **Behind the subject:** a big word, centred, no "
                                "outline, opacity 100. Works best on a photo with one "
                                "clear subject. Needs the background-removal model, "
                                "which the `[onnx]` extra installs.",
                                elem_classes="notes",
                            )
                        with gr.Row():
                            wm_btn = gr.Button("Apply (full size)", variant="primary",
                                               size="lg", scale=3)
                            wm_use = gr.Button("↪ Use as input", variant="secondary", scale=2)
                            wm_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1, elem_classes="sticky-col"):
                        wm_preview = gr.Image(
                            label="Preview", height=380, buttons=["fullscreen"],
                            elem_classes=["loupe"],
                        )
                        wm_note = gr.Markdown(elem_classes="notes")
                        wm_out = gr.Image(
                            label="Result at full size", height=340, type="pil",
                            buttons=["download", "fullscreen"], elem_classes=["loupe"],
                            interactive=False,
                        )
                        wm_file = gr.File(label="Download PNG")
                        wm_info = gr.Markdown()

            # ---- Tab: Design templates (no AI, except the cut-out one) ----
            with gr.Tab("Design"):
                # The first template, resolved here so every control below opens
                # already showing it — a load event would leave the tab blank
                # until the browser had been round-tripped.
                _dz0 = dz_tools.built_in(dz_tools.BUILT_IN_NAMES[0])
                _dz0_slots = _dz0.text_slots()
                _dz0_copy = {}
                for _layer in _dz0.layers:
                    if _layer.kind == dz_tools.TEXT:
                        _dz0_copy.setdefault(_layer.slot, _layer.text)
                gr.HTML(_section_head(
                    "Compose", "Design Templates",
                    "Start from a finished layout — a YouTube thumbnail, a quote card, "
                    "an event poster, a title slide — and fill in the words and the "
                    "photo. The template carries the design; a palette restyles the "
                    "whole thing in one click.",
                    icon=ICON_DESIGN,
                ))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        dz_template = gr.Dropdown(
                            dz_tools.BUILT_IN_NAMES, value=dz_tools.BUILT_IN_NAMES[0],
                            label="Template", filterable=False,
                            info="Ten layouts for the things people actually make.",
                        )
                        dz_note = gr.Markdown(f"*{_dz0.note}*", elem_classes="notes")
                        dz_photo = gr.Image(
                            label="Photo", type="pil", image_mode=None,
                            sources=["upload", "clipboard"], height=200,
                            elem_classes="drop", buttons=["download", "fullscreen"],
                            visible=_dz0.wants_photo(),
                        )
                        dz_cut_note = gr.Markdown(visible=False, elem_classes="notes")
                        dz_cutout_state = gr.State(None)
                        dz_logo = gr.Image(
                            label="Logo (a transparent PNG works best)", type="pil",
                            image_mode="RGBA", sources=["upload", "clipboard"],
                            height=120, visible=_dz0.wants_logo(),
                        )
                        gr.HTML('<div class="col-label">Your words</div>')
                        dz_texts = [
                            gr.Textbox(
                                value=(_dz0_copy.get(_dz0_slots[i], "")
                                       if i < len(_dz0_slots) else ""),
                                label=(_dz0_slots[i].replace("_", " ").title()
                                       if i < len(_dz0_slots) else f"Text {i + 1}"),
                                lines=1, visible=i < len(_dz0_slots),
                            )
                            for i in range(DZ_SLOTS)
                        ]
                        gr.HTML('<div class="col-label">Colour</div>')
                        dz_palette = gr.Dropdown(
                            dz_tools.PALETTE_NAMES, value=None, label="Palette",
                            filterable=False,
                            info="Restyles the whole design. The four colours below "
                            "stay editable afterwards.",
                        )
                        with gr.Row():
                            dz_primary = gr.ColorPicker(value=_dz0.palette.primary,
                                                        label="Ground")
                            dz_secondary = gr.ColorPicker(value=_dz0.palette.secondary,
                                                          label="Second")
                        with gr.Row():
                            dz_accent = gr.ColorPicker(value=_dz0.palette.accent,
                                                       label="Accent")
                            dz_ink = gr.ColorPicker(value=_dz0.palette.ink, label="Ink")
                        dz_canvas = gr.Dropdown(
                            dz_tools.CANVAS_NAMES, value=_dz0.canvas,
                            label="Canvas", filterable=False,
                            info="The finished size. Every measurement is a share of "
                            "the canvas, so a template holds together at any of them.",
                        )
                        with gr.Accordion("Template JSON (advanced)", open=False):
                            gr.Markdown(
                                "The template as plain JSON — layers, positions, "
                                "colours. Edit it and press **Load** to see the "
                                "change, or copy it somewhere as your own template.",
                                elem_classes="notes",
                            )
                            dz_json = gr.Code(value=dz_tools.to_json(_dz0),
                                              language="json", lines=14,
                                              label="Template")
                            dz_load = gr.Button("Load this template", size="sm")
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "* **The words shrink to fit.** A long headline wraps "
                                "and sizes itself down rather than running off the "
                                "edge — so type what you mean and leave it.\n"
                                "* **Palette first, then tweak.** Pick a palette for "
                                "the mood, then nudge the four colours.\n"
                                "* **The canvas is separate from the template.** A "
                                "quote card renders just as well at story size.\n"
                                "* **Subject spotlight** puts the word behind the "
                                "person, so it wants a photo with one clear subject "
                                "and the background-removal model from `[onnx]`.\n"
                                "* **Make your own** by editing the JSON above — or "
                                "from the command line with "
                                "`upscaler design \"Quote card\" --save mine.json`.",
                                elem_classes="notes",
                            )
                        with gr.Row():
                            dz_btn = gr.Button("Render (full size)", variant="primary",
                                               size="lg", scale=3)
                            dz_clear = gr.Button("↺ Clear", variant="secondary",
                                                 scale=1)
                    with gr.Column(scale=1, elem_classes="sticky-col"):
                        dz_preview = gr.Image(
                            label="Preview", height=420, buttons=["fullscreen"],
                            elem_classes=["loupe"],
                        )
                        dz_desc = gr.Markdown(elem_classes="notes")
                        dz_out = gr.Image(
                            label="Result at full size", height=340, type="pil",
                            buttons=["download", "fullscreen"], elem_classes=["loupe"],
                            interactive=False,
                        )
                        dz_file = gr.File(label="Download PNG")
                        dz_info = gr.Markdown()

            # ---- Tab: Screenshot beautifier (no AI) ----
            with gr.Tab("Screenshot"):
                gr.HTML(_section_head(
                    "Present", "Screenshot Beautifier",
                    "Turn a raw screen capture into something you can put in a README, "
                    "a landing page or a post: padding and rounded corners on a colour "
                    "field, a soft shadow underneath, and — if you want it — a window "
                    "or browser frame, leaned back in 3D.",
                    icon=ICON_SHOT,
                ))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        shot_in = gr.Image(
                            label="Screenshot (paste with ⌘V / Ctrl+V)", type="pil",
                            image_mode=None, sources=["upload", "clipboard"], height=280,
                            elem_classes="drop", buttons=["download", "fullscreen"],
                        )
                        shot_preset = gr.Dropdown(
                            shot_tools.PRESET_NAMES, value="Indigo mesh",
                            label="Look", filterable=False,
                            info="A starting point — everything below stays editable.",
                        )
                        gr.HTML('<div class="col-label">Background</div>')
                        shot_bg = gr.Radio(
                            [(shot_tools.BACKGROUND_LABELS[v], v)
                             for v in shot_tools.BACKGROUNDS],
                            value=shot_tools.GRADIENT, label="Behind the shot",
                            info="Mesh is the soft multi-colour wash; transparent gives "
                            "you a PNG to drop on your own page.",
                        )
                        with gr.Row():
                            shot_color = gr.ColorPicker(value="#6366f1", label="Colour")
                            shot_color2 = gr.ColorPicker(value="#a855f7", label="Second colour")
                        shot_angle = gr.Slider(
                            0, 360, value=135, step=5, label="Gradient angle (°)",
                            info="Which way the colours run.",
                        )
                        gr.HTML('<div class="col-label">The shot</div>')
                        with gr.Row():
                            shot_padding = gr.Slider(
                                0, 40, value=9, step=0.5, label="Padding (%)",
                                info="Space around the shot, as a share of its short "
                                "side — so it looks the same on any capture.",
                            )
                            shot_radius = gr.Slider(
                                0, 12, value=2, step=0.1, label="Corner radius (%)",
                                info="0 keeps the square corners.",
                            )
                        with gr.Row():
                            shot_shadow = gr.Slider(
                                0, 100, value=55, step=1, label="Shadow",
                                info="How dark the shadow under the shot is.",
                            )
                            shot_soft = gr.Slider(
                                0, 100, value=50, step=1, label="Shadow softness",
                                info="Low is a tight contact shadow, high a wide pool.",
                            )
                        shot_rim = gr.Slider(
                            0, 100, value=0, step=1, label="Edge highlight",
                            info="A hairline around the shot. A shadow shows nothing "
                            "when a dark screenshot sits on a dark background — this "
                            "is what separates them.",
                        )
                        gr.HTML('<div class="col-label">Window frame</div>')
                        shot_chrome = gr.Dropdown(
                            [(shot_tools.CHROME_LABELS[v], v) for v in shot_tools.CHROMES],
                            value=shot_tools.NO_CHROME, label="Chrome", filterable=False,
                            info="Wraps the shot in a title bar, so it reads as an app "
                            "rather than a crop.",
                        )
                        shot_title = gr.Textbox(
                            value="", label="Address bar", visible=False,
                            placeholder="example.com",
                            info="What the browser frame's address bar reads.",
                        )
                        gr.HTML('<div class="col-label">In space</div>')
                        with gr.Row():
                            shot_tilt = gr.Slider(
                                -shot_tools.MAX_TILT, shot_tools.MAX_TILT, value=0, step=1,
                                label="Tilt (°)",
                                info="Leans the right edge away from you; negative "
                                "leans the left.",
                            )
                            shot_pitch = gr.Slider(
                                -shot_tools.MAX_TILT, shot_tools.MAX_TILT, value=0, step=1,
                                label="Pitch (°)",
                                info="Leans the top edge away; negative the bottom.",
                            )
                        shot_spin = gr.Slider(
                            -shot_tools.MAX_SPIN, shot_tools.MAX_SPIN, value=0, step=0.5,
                            label="Spin (°)", info="Rotates it flat on the page. A "
                            "degree or two with a tilt sells the 3D.",
                        )
                        gr.HTML('<div class="col-label">Canvas</div>')
                        with gr.Row():
                            shot_aspect = gr.Dropdown(
                                shot_tools.ASPECT_NAMES, value=shot_tools.AUTO_ASPECT,
                                label="Shape", filterable=False,
                                info="Auto follows the shot. A fixed shape grows the "
                                "background to fit — it never crops the shot.",
                            )
                            shot_custom = gr.Textbox(
                                value="", label="Custom ratio", visible=False,
                                placeholder="16:10", info="Like 16:10 or 1200x800.",
                            )
                        shot_size = gr.Textbox(
                            value="", label="Exact size (optional)",
                            placeholder="1600x900",
                            info="Land on exact pixels, e.g. 1600x900 for a GitHub "
                            "social card.",
                        )
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "* **Paste straight in.** ⌘⇧4 on a Mac or Win+Shift+S "
                                "on Windows puts the capture on the clipboard; click "
                                "the box above and press ⌘V / Ctrl+V.\n"
                                "* **Dark screenshot on a dark background?** Turn up "
                                "*Edge highlight* — a black shadow on near-black shows "
                                "nothing, and a lit edge is what a real window has.\n"
                                "* **Tilt plus a degree or two of spin** reads as a "
                                "product shot; tilt alone can look like a mistake.\n"
                                "* **Transparent** gives you a PNG with the shadow "
                                "still attached, to drop on your own background.\n"
                                "* **1600×900** is the size GitHub and most social "
                                "cards want — 'README hero' sets it for you.\n"
                                "* **Batch a folder** from the command line with "
                                "`upscaler screenshot ./shots --preset \"Indigo mesh\"`.",
                                elem_classes="notes",
                            )
                        with gr.Row():
                            shot_btn = gr.Button("Apply (full size)", variant="primary",
                                               size="lg", scale=3)
                            shot_use = gr.Button("↪ Use as input", variant="secondary",
                                               scale=2)
                            shot_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1, elem_classes="sticky-col"):
                        shot_preview = gr.Image(
                            label="Preview", height=380, buttons=["fullscreen"],
                            elem_classes=["loupe"],
                        )
                        shot_note = gr.Markdown(elem_classes="notes")
                        shot_out = gr.Image(
                            label="Result at full size", height=340, type="pil",
                            buttons=["download", "fullscreen"], elem_classes=["loupe"],
                            interactive=False,
                        )
                        shot_file = gr.File(label="Download PNG")
                        shot_info = gr.Markdown()

            # ---- Tab: Crop & frame (no AI) ----
            with gr.Tab("Crop"):
                gr.HTML(_section_head(
                    "Geometry", "Crop & Frame",
                    "Crop to any shape, straighten a tilted horizon, fix leaning "
                    "verticals, land on an exact pixel size, and add a border, "
                    "rounded corners or a drop shadow. Nothing here is guesswork — "
                    "a straighten or a lean is trimmed back to real pixels, never "
                    "leaving empty corners.",
                    icon=ICON_CROP,
                ))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        fr_in = gr.Image(
                            label="Input", type="pil", image_mode=None,
                            sources=["upload", "clipboard"], height=300,
                            elem_classes="drop", buttons=["download", "fullscreen"],
                        )
                        fr_preset = gr.Dropdown(
                            frame_tools.PRESET_NAMES, value=frame_tools.PRESET_NONE,
                            label="Preset", filterable=False,
                            info="Common shapes and frames. Everything stays editable "
                            "afterwards; 'None' clears it.",
                        )
                        with gr.Accordion("Crop", open=True):
                            fr_aspect = gr.Dropdown(
                                list(frame_tools.ASPECTS), value=frame_tools.DEFAULT_ASPECT,
                                label="Shape", filterable=False,
                                info="The aspect ratio to crop to. 'Original' keeps the "
                                "photo's own shape.",
                            )
                            fr_custom = gr.Textbox(
                                label="Custom ratio", value="", visible=False,
                                placeholder="16:10",
                                info="Any ratio — 16:10, 5/4, or a size like 1200x800.",
                            )
                            fr_mode = gr.Radio(
                                frame_tools.CROP_MODES, value="fill", label="How to fit",
                                visible=False,
                                info="fill crops the photo to the shape · fit keeps the "
                                "whole photo and fills the margin instead, so nothing is "
                                "cut off.",
                            )
                            with gr.Row():
                                fr_px = gr.Slider(0, 100, value=50, step=1, label="Position X (%)",
                                                  info="Which part survives a crop that "
                                                  "trims the sides.")
                                fr_py = gr.Slider(0, 100, value=50, step=1, label="Position Y (%)",
                                                  info="Which part survives a crop that "
                                                  "trims top and bottom.")
                            fr_zoom = gr.Slider(
                                1, frame_tools.MAX_ZOOM, value=1, step=0.05, label="Zoom",
                                info="Crop in tighter than the largest box that fits.",
                            )
                        with gr.Accordion("Straighten & rotate", open=False):
                            fr_straighten = gr.Slider(
                                -frame_tools.MAX_STRAIGHTEN, frame_tools.MAX_STRAIGHTEN,
                                value=0, step=0.1, label="Straighten (°)",
                                info="Level a tilted horizon. The frame is trimmed to "
                                "the largest rectangle with no empty corners.",
                            )
                            fr_rotate = gr.Radio(
                                frame_tools.ROTATIONS, value=0, label="Rotate (°)",
                                info="Quarter turns, clockwise.",
                            )
                            with gr.Row():
                                fr_fliph = gr.Checkbox(value=False, label="Mirror left ↔ right")
                                fr_flipv = gr.Checkbox(value=False, label="Flip top ↕ bottom")
                            with gr.Row():
                                fr_kh = gr.Slider(
                                    -100, 100, value=0, step=1, label="Lean horizontally",
                                    info="Straightens converging horizontals — a wall "
                                    "shot from an angle.",
                                )
                                fr_kv = gr.Slider(
                                    -100, 100, value=0, step=1, label="Lean vertically",
                                    info="Straightens converging verticals — a building "
                                    "shot from below.",
                                )
                        with gr.Accordion("Frame", open=False):
                            fr_border = gr.Slider(
                                0, 40, value=0, step=0.5, label="Border (%)",
                                info="A margin around the photo, as a share of its short "
                                "side.",
                            )
                            fr_style = gr.Radio(
                                frame_tools.BORDER_STYLES, value="solid", label="Fill",
                                info="A flat color, or a zoomed blurred copy of the photo "
                                "itself.",
                            )
                            fr_color = gr.ColorPicker(value="#ffffff", label="Border color",
                                                      info="The color of the margin.")
                            fr_blur = gr.Slider(
                                0, 100, value=60, step=1, label="Fill blur", visible=False,
                                info="How soft the blurred-photo fill is.",
                            )
                            fr_radius = gr.Slider(
                                0, 50, value=0, step=0.5, label="Rounded corners (%)",
                                info="Rounds the photo's corners. Without a border the "
                                "corners come out transparent.",
                            )
                            fr_shadow = gr.Slider(
                                0, 100, value=0, step=1, label="Drop shadow",
                                info="A soft shadow under the photo. It needs a border "
                                "to fall on.",
                            )
                        fr_size = gr.Textbox(
                            label="Exact output size (optional)", value="",
                            placeholder="1920x1080",
                            info="Land on an exact pixel size. The photo is cropped to "
                            "that shape first, then resampled.",
                        )
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "* **'fit' instead of 'fill'** when a crop would cut "
                                "something important — the whole photo goes in and the "
                                "margin gets a blurred copy of it.\n"
                                "* **Straighten before cropping**: the crop is taken "
                                "from the levelled frame, so you don't lose it twice.\n"
                                "* **Leaning verticals** on a building shot from below: "
                                "pull 'lean vertically' until the edges are parallel.\n"
                                "* **A drop shadow needs a border** — that's the space "
                                "it falls on.\n"
                                "* **For a wallpaper**, pick the shape and type the "
                                "exact size; upscale first if the photo is small.\n"
                                "* **Rounded corners with no border** give a transparent "
                                "PNG you can drop onto anything.",
                                elem_classes="notes",
                            )
                        with gr.Row():
                            fr_btn = gr.Button("Apply (full size)", variant="primary",
                                               size="lg", scale=3)
                            fr_use = gr.Button("↪ Use as input", variant="secondary", scale=2)
                            fr_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1, elem_classes="sticky-col"):
                        fr_preview = gr.Image(
                            label="Preview", height=380, buttons=["fullscreen"],
                            elem_classes=["loupe"],
                        )
                        fr_note = gr.Markdown(elem_classes="notes")
                        fr_out = gr.Image(
                            label="Result at full size", height=340,
                            buttons=["download", "fullscreen"], elem_classes=["loupe"],
                            type="pil", interactive=False,
                        )
                        fr_file = gr.File(label="Download PNG")
                        fr_info = gr.Markdown()

            # ---- Tab: Sharpen (no AI) ----
            with gr.Tab("Sharpen"):
                gr.HTML(_section_head(
                    "Detail", "Sharpen",
                    "Four ways to bring out detail — the classic unsharp mask, a "
                    "high-pass overlay, an edge-aware pass that leaves skin and sky "
                    "alone, and a two-scale texture pass — with halo control so edges "
                    "get crisp instead of outlined.",
                    icon=ICON_SHARP,
                ))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        sh_in = gr.Image(
                            label="Input", type="pil", image_mode=None,
                            sources=["upload", "clipboard"], height=300,
                            elem_classes="drop", buttons=["download", "fullscreen"],
                        )
                        sh_preset = gr.Dropdown(
                            sharpen_tools.PRESET_NAMES, value=sharpen_tools.PRESET_NONE,
                            label="Preset", filterable=False,
                            info="A starting point — every control stays editable "
                            "afterwards. 'None' turns sharpening off.",
                        )
                        sh_kind = gr.Radio(
                            sharpen_tools.KINDS, value="unsharp", label="Method",
                            info="unsharp = the classic · high-pass = lifts edges "
                            "without changing overall tone · smart = only where there "
                            "are real edges, so noise and skin are left alone · "
                            "texture = two scales at once, for fine detail plus a "
                            "little structure.",
                        )
                        sh_amount = gr.Slider(
                            0, 300, value=80, step=1, label="Amount",
                            info="How hard to push. 100 is a classic full-strength "
                            "unsharp mask; 0 turns it off.",
                        )
                        with gr.Row():
                            sh_radius = gr.Slider(
                                sharpen_tools.MIN_RADIUS, sharpen_tools.MAX_RADIUS,
                                value=1.0, step=0.1, label="Radius (px)",
                                info="The width of the edges you're lifting. Around 1 "
                                "suits most photos; go wider only for a very soft shot.",
                            )
                            sh_threshold = gr.Slider(
                                0, 100, value=4, step=1, label="Threshold",
                                info="Leaves flat areas alone, so grain and noise "
                                "aren't amplified along with the detail.",
                            )
                        sh_halo = gr.Slider(
                            0, 100, value=35, step=1, label="Halo limit",
                            info="Caps the bright and dark rim an edge may gain. Lower "
                            "keeps it honest; high lets edges outline themselves.",
                        )
                        sh_edge = gr.Slider(
                            0, 100, value=50, step=1, label="Edge sensitivity",
                            visible=False,
                            info="How strictly the smart pass sticks to real edges. "
                            "Higher protects more of the flat areas.",
                        )
                        sh_balance = gr.Slider(
                            0, 100, value=50, step=1, label="Fine ↔ structure",
                            visible=False,
                            info="Low favours fine grit, high favours broader "
                            "structure.",
                        )
                        with gr.Accordion("Protect", open=False):
                            sh_lum = gr.Checkbox(
                                value=True, label="Sharpen brightness only",
                                info="Leaves color untouched, which avoids the colored "
                                "fringes per-channel sharpening leaves on edges.",
                            )
                            with gr.Row():
                                sh_shadows = gr.Slider(
                                    0, 100, value=0, step=1, label="Protect shadows",
                                    info="Holds back in the darkest areas, where "
                                    "sharpening mostly finds noise.",
                                )
                                sh_highlights = gr.Slider(
                                    0, 100, value=0, step=1, label="Protect highlights",
                                    info="Holds back in the brightest areas, where a "
                                    "halo shows most.",
                                )
                        with gr.Accordion("Where to sharpen", open=False):
                            sh_shape = gr.Radio(
                                blur.SHAPES, value="whole", label="Region",
                                info="whole = the entire photo · rectangle / ellipse = a "
                                "shape you position · band = a straight strip · painted = "
                                "wherever you brush · faces = every face found "
                                "automatically.",
                            )
                            with gr.Row():
                                sh_x = gr.Slider(0, 100, value=50, step=1, label="Centre X (%)",
                                                 visible=False, info="Shape position.")
                                sh_y = gr.Slider(0, 100, value=50, step=1, label="Centre Y (%)",
                                                 visible=False, info="Shape position.")
                            with gr.Row():
                                sh_w = gr.Slider(1, 100, value=50, step=1, label="Width (%)",
                                                 visible=False, info="Shape size.")
                                sh_h = gr.Slider(1, 100, value=50, step=1, label="Height (%)",
                                                 visible=False, info="Shape size, or how "
                                                 "thick the band is.")
                            sh_mangle = gr.Slider(-90, 90, value=0, step=1, label="Band tilt (°)",
                                                  visible=False, info="Rotate the strip.")
                            sh_round = gr.Slider(0, 100, value=0, step=1,
                                                 label="Corner roundness (%)", visible=False,
                                                 info="0 = sharp corners, 100 = a pill.")
                            sh_feather = gr.Slider(0, 50, value=15, step=0.5, label="Feather (%)",
                                                   visible=False,
                                                   info="How softly the sharpening fades out "
                                                   "at the edge of the region.")
                            sh_outside = gr.Checkbox(value=False, label="Sharpen outside the shape",
                                                     visible=False,
                                                     info="Sharpen everything except the shape.")
                            sh_facepad = gr.Slider(
                                -25, 100, value=25, step=1, label="Face padding (%)",
                                visible=False,
                                info="Grows the oval around each detected face.",
                            )
                            sh_faces_note = gr.Markdown(visible=False, elem_classes="notes")
                            sh_faces_state = gr.State([])
                            sh_editor = gr.ImageEditor(
                                label="Paint where to sharpen", type="pil", height=360,
                                sources=[], layers=False, transforms=(), visible=False,
                                brush=gr.Brush(colors=["#ffffff"], color_mode="fixed",
                                               default_size=40),
                            )
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "* **Judge it at 1:1** — the preview is a real crop, not "
                                "a shrunken copy, because shrinking a photo hides "
                                "exactly the artefacts you're looking for.\n"
                                "* **Radius first, then amount.** Around 1px suits most "
                                "photos; a wide radius plus a big amount is what makes "
                                "pictures look crunchy.\n"
                                "* **Watch the halo limit** — if edges grow a bright "
                                "outline, lower it rather than lowering the amount.\n"
                                "* **Raise the threshold on a noisy or high-ISO shot**, "
                                "or use the smart method, so the noise stays put.\n"
                                "* **Sharpen last**, after upscaling and color work.\n"
                                "* **Portraits:** smart method, protect highlights, and "
                                "region = faces if you only want the eyes and lips to "
                                "come up.",
                                elem_classes="notes",
                            )
                        with gr.Row():
                            sh_btn = gr.Button("Apply (full size)", variant="primary",
                                               size="lg", scale=3)
                            sh_use = gr.Button("↪ Use as input", variant="secondary", scale=2)
                            sh_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1, elem_classes="sticky-col"):
                        with gr.Row():
                            sh_cx = gr.Slider(0, 100, value=50, step=1, label="Preview X (%)",
                                              info="Which part of the photo the 1:1 crop "
                                              "shows.")
                            sh_cy = gr.Slider(0, 100, value=50, step=1, label="Preview Y (%)",
                                              info="Which part of the photo the 1:1 crop "
                                              "shows.")
                        sh_preview = gr.ImageSlider(
                            # max_height, not height — see `out` slider above.
                            label="Live preview at 1:1 — before / after (drag the divider)",
                            type="pil", max_height=340, elem_classes=["loupe"],
                        )
                        sh_crop_note = gr.Markdown(elem_classes="notes")
                        sh_out = gr.ImageSlider(
                            label="Result at full size — before / after", type="pil",
                            max_height=340, elem_classes=["loupe"], interactive=False,
                        )
                        sh_file = gr.File(label="Download PNG")
                        sh_info = gr.Markdown()

            # ---- Tab: Video upscaler (frame-by-frame) ----
            with gr.Tab("Video"):
                gr.HTML(_section_head(
                    "Video", "Video Upscaler",
                    "Upscale a clip frame by frame on your own machine, keeping the "
                    "original audio. ×2 is faster and steadier between frames than "
                    "×4. Longer clips take a while, and ffmpeg must be installed.",
                    icon=ICON_AI,
                ))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        vid_in = gr.Video(label="Input video", sources=["upload"])
                        vid_model = gr.Dropdown(
                            _MODEL_CHOICES, value="realesrgan-x2plus",
                            label="Upscale model", filterable=True,
                            info="The AI that enlarges each frame. ×2 is recommended "
                            "for video — it's faster and flickers less between frames "
                            "than ×4. Type to filter.",
                        )
                        vid_size = gr.Dropdown(
                            list(_SIZE_PRESETS), value="Model default (×2/×4)",
                            label="Output size", filterable=False,
                            info="After enlarging, shrinks the longest edge of every "
                            "frame to this size (e.g. 4K = 3840px).",
                        )
                        vid_sharpen = gr.Slider(
                            0.0, 3.0, value=0.0, step=0.1,
                            label="Sharpen per frame — 0 = off",
                            info="Crispens edges on each frame. Go easy on video — "
                            "sharpening can amplify flicker between frames.",
                        )
                        vid_smooth = gr.Dropdown(
                            ["Off", "30", "48", "60", "120"], value="Off",
                            label="Smooth motion (interpolate to fps)", filterable=False,
                            info="Invents in-between frames so playback looks "
                            "smoother. Higher target fps = smoother motion but much "
                            "slower to render.",
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
                                    info="Skip everything before this point.",
                                )
                                vid_end = gr.Number(
                                    value=0, label="End (seconds)", minimum=0,
                                    info="Stop here (0 or the clip length = play to "
                                    "the end). Lower it to render less.",
                                )
                        with gr.Accordion("Advanced", open=False):
                            vid_device = gr.Dropdown(
                                _DEVICES, value=_cfg_device, label="Device",
                                filterable=False,
                                info="Where the work runs. \"auto\" uses your "
                                "graphics card (GPU) if it can, otherwise your "
                                "processor (CPU).",
                            )
                            vid_tile = gr.Slider(
                                0, 1024, value=512, step=64,
                                label="Tile size (0 = off)",
                                info="Splits big frames into chunks to use less "
                                "memory. Lower this if a render crashes with an "
                                "out-of-memory error; 0 turns it off.",
                            )
                            # Default on when it's the only road to the GPU:
                            # torch resolves to CPU but DirectML can see a GPU
                            # (the typical AMD-on-Windows setup) — video on CPU
                            # is painfully slow and nobody finds this toggle.
                            _vid_onnx_default = device_name == "cpu" and _dml_available()
                            vid_onnx = gr.Checkbox(
                                value=_vid_onnx_default,
                                label="Alternative speed engine (ONNX)",
                                info="Runs frames without PyTorch. With the "
                                "DirectML runtime installed this uses your "
                                "graphics card (AMD included) — much faster than "
                                "CPU. The first run exports the model."
                                + (" Turned on for you: your GPU is reachable "
                                   "via DirectML but not via PyTorch."
                                   if _vid_onnx_default else ""),
                            )
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "* **Use the ×2 model for video** — faster, and "
                                "flickers less between frames than ×4.\n"
                                "* **Test on a short trim first** (open Trim) before "
                                "the whole clip — long videos take a while.\n"
                                "* **Keep per-frame Sharpen very low** — it can "
                                "amplify shimmer between frames.\n"
                                "* **Needs ffmpeg installed**; your audio is kept "
                                "automatically.",
                                elem_classes="notes",
                            )
                        with gr.Row():
                            vid_btn = gr.Button(
                                "Upscale video", variant="primary", size="lg", scale=3
                            )
                            vid_cancel = gr.Button("✕ Cancel", variant="stop", scale=1)
                            vid_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1, elem_classes="sticky-col"):
                        vid_out = gr.Video(label="Result", buttons=["download"])
                        vid_compare = gr.ImageSlider(
                            label="Before / after (drag to compare)",
                            # max_height, not height — see `out` slider above.
                            type="pil", max_height=220,
                            buttons=["download", "fullscreen"],
                            elem_classes=["loupe"],
                        )
                        vid_scrub = gr.Slider(
                            0, 1, value=0, step=0.1, visible=False,
                            label="Compare at (seconds)",
                            info="Move through the clip to check any moment "
                            "against the source, not just the first frame.",
                        )
                        vid_info = gr.Markdown()

            # ---- Tab: Convert & documents (format / PDF) ----
            with gr.Tab("Convert"):
                gr.HTML(_section_head(
                    "Convert", "Convert & Documents",
                    "Change an image's format, combine several images into a PDF, "
                    "fit a photo under a file-size limit, strip the metadata that "
                    "records where a photo was taken, or build and split PDFs. Pick a "
                    "task below to begin.",
                    icon=ICON_CONVERT,
                ))
                method = gr.Dropdown(
                    _CONVERT_METHODS, value=_CONVERT_METHODS[0],
                    label="What do you want to do?", filterable=False,
                    info="Pick your task: change an image's format, squeeze one under "
                    "a size limit, remove metadata such as GPS location, build a PDF "
                    "from images, or split a PDF back into images.",
                )
                with gr.Accordion("Tips", open=False):
                    gr.Markdown(
                        "* **The options below change** to match the task you pick "
                        "here.\n"
                        "* **PNG and TIFF keep full quality**; JPEG, WebP, AVIF and "
                        "HEIC are smaller but lossy.\n"
                        "* **Quality 90 is a great balance** for lossy formats "
                        "(ignored for PNG/TIFF).\n"
                        "* **PDF pages: 150 DPI is fine on screen** — use 300 only "
                        "if you'll print them.",
                        elem_classes="notes",
                    )

                # -- Method C: remove metadata --
                with gr.Column(visible=False) as grp_meta:
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            # A File, not an Image: Gradio re-encodes an Image on
                            # upload, which would strip the metadata before we
                            # could show anyone what was in it.
                            md_in = gr.File(
                                label="Photo to check (the original file, not a copy)",
                                file_count="single",
                                file_types=["image"], elem_classes="drop",
                            )
                            md_preview = gr.Image(label="Preview", height=220,
                                                  buttons=["fullscreen"],
                                                  elem_classes=["loupe"])
                            md_mode = gr.Radio(
                                md_tools.MODES, value=md_tools.REMOVE_ALL,
                                label="What to remove",
                                info="Everything, or just the location, or everything "
                                "except your copyright and artist name.",
                            )
                            md_keep_rot = gr.Checkbox(
                                value=True, label="Keep the photo upright",
                                info="Phones store some photos sideways plus a tag "
                                "saying to rotate them. This keeps that one tag, which "
                                "says nothing about you, so the picture doesn't end up "
                                "on its side.",
                            )
                            md_btn = gr.Button("Remove metadata", variant="primary",
                                               size="lg", visible=False)
                            with gr.Accordion("What is this?", open=False):
                                gr.Markdown(
                                    "Your camera or phone writes a block of data next "
                                    "to the pixels, and it travels with the file: **where "
                                    "the photo was taken**, when, the camera and its "
                                    "serial number, and sometimes your name.\n\n"
                                    "Big platforms strip it when you upload. Forums, "
                                    "email attachments, file transfers and your own "
                                    "website do not.\n\n"
                                    "**Cleaning a JPEG or PNG here is lossless** — the "
                                    "metadata is cut out and the compressed picture is "
                                    "copied through untouched, so it costs no quality "
                                    "at all.",
                                    elem_classes="notes",
                                )
                        with gr.Column(scale=1):
                            md_report = gr.Markdown()
                            md_file = gr.File(label="Download the cleaned file")
                            md_result = gr.Markdown()

                # -- Method B: fit a file-size budget --
                with gr.Column(visible=False) as grp_budget:
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            opt_in = gr.Image(
                                label="Image", type="pil", image_mode=None,
                                sources=["upload", "clipboard"], height=300,
                                elem_classes="drop", buttons=["download", "fullscreen"],
                            )
                            opt_target = gr.Textbox(
                                value="500 KB", label="Target size",
                                placeholder="500 KB",
                                info="The most the file may weigh — 500KB, 2MB, 1.5 MB. "
                                "Handy for email limits, forum uploads and Steam.",
                            )
                            opt_fmt = gr.Dropdown(
                                opt_tools.BUDGET_FORMATS, value=opt_tools.AUTO,
                                label="Format", filterable=False,
                                info="auto picks WebP, which carries the same picture in "
                                "about half a JPEG's bytes. Choose JPEG if whatever you "
                                "are uploading to won't take WebP.",
                            )
                            with gr.Row():
                                opt_minq = gr.Slider(
                                    1, 95, value=40, step=1, label="Quality floor",
                                    info="How far quality may drop before it starts "
                                    "shrinking the picture instead.",
                                )
                                opt_maxedge = gr.Number(
                                    value=0, label="Max width/height (px)", precision=0,
                                    info="Cap the long edge first. 0 leaves the size "
                                    "alone.",
                                )
                            opt_resize = gr.Checkbox(
                                value=True, label="Shrink the picture if it still won't fit",
                                info="Off keeps the original dimensions no matter what, "
                                "which may mean the budget can't be met.",
                            )
                            with gr.Accordion("Tips", open=False):
                                gr.Markdown(
                                    "* **Quality is searched, not guessed** — it tries "
                                    "the highest setting that still fits, because how "
                                    "big a photo encodes depends on what's in it.\n"
                                    "* **auto means WebP.** Pick JPEG only if the site "
                                    "you're uploading to refuses it.\n"
                                    "* **Metadata is always stripped**, which drops the "
                                    "GPS coordinates along with the bytes.\n"
                                    "* **If it can't reach the target**, lower the "
                                    "quality floor or allow shrinking.",
                                    elem_classes="notes",
                                )
                            opt_btn = gr.Button("Fit the budget", variant="primary",
                                                size="lg")
                        with gr.Column(scale=1):
                            opt_out = gr.ImageSlider(
                                label="Before / after — drag to compare", type="pil",
                                max_height=340, elem_classes=["loupe"], interactive=False,
                            )
                            opt_file = gr.File(label="Download")
                            opt_info = gr.Markdown()

                # -- Method A: change image format --
                with gr.Column(visible=True) as grp_format:
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            conv_in = gr.Image(
                                label="Image", type="pil", image_mode=None,
                                sources=["upload", "clipboard"], height=300,
                                elem_classes="drop", buttons=["download", "fullscreen"],
                            )
                            conv_fmt = gr.Dropdown(
                                list(FORMATS), value="PNG", label="Convert to",
                                filterable=False,
                                info="The file type to save. PNG and TIFF keep full "
                                "quality; JPEG, WebP, AVIF and HEIC make smaller "
                                "files but are lossy (some quality is thrown away).",
                            )
                            conv_quality = gr.Slider(
                                1, 100, value=90, step=1, label="Quality (lossy)",
                                info="Only matters for lossy formats. Higher looks "
                                "better but makes a bigger file; ignored for "
                                "PNG/TIFF.",
                            )
                            conv_lossless = gr.Checkbox(
                                value=False, label="Lossless WebP",
                                info="Saves WebP with no quality loss at all — a "
                                "bigger file, but nothing is thrown away.",
                            )
                            conv_btn = gr.Button("Convert", variant="primary", size="lg")
                        with gr.Column(scale=1):
                            conv_file = gr.File(label="Download converted file")
                            conv_info = gr.Markdown()

                # -- Method B: images -> PDF --
                with gr.Column(visible=False) as grp_topdf:
                    with gr.Row(equal_height=False):
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
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            pdf_in = gr.File(
                                label="PDF", file_count="single",
                                file_types=[".pdf"], elem_classes="drop",
                            )
                            pdf_dpi = gr.Slider(
                                72, 300, value=150, step=1, label="Render DPI",
                                info="How much detail each PDF page is rendered at. "
                                "Higher = sharper, larger PNGs. 150 is a good "
                                "default; use 300 for print quality.",
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

                opt_btn.click(
                    optimize_ui,
                    [opt_in, opt_target, opt_fmt, opt_minq, opt_resize, opt_maxedge],
                    [opt_out, opt_file, opt_info], show_progress_on=[opt_out],
                )
                md_in.change(metadata_inspect_ui, md_in,
                             [md_preview, md_report, md_btn], show_progress="hidden")
                md_btn.click(metadata_strip_ui, [md_in, md_mode, md_keep_rot],
                             [md_file, md_result], show_progress_on=[md_file])
                method.change(
                    _switch_method, method,
                    [grp_format, grp_budget, grp_meta, grp_topdf, grp_frompdf],
                    show_progress="hidden",  # instant visibility toggle, no flash
                )

            # ---- Tab: Batch (one operation over many images) ----
            with gr.Tab("Batch"):
                gr.HTML(_section_head(
                    "Batch", "Batch Processing",
                    "Drop a whole stack of images, pick one operation, and run it "
                    "over all of them at once. Results download as a ZIP and are "
                    "saved to your Library.",
                    icon=ICON_BATCH,
                ))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        batch_in = gr.File(
                            label="Images (drop as many as you like)",
                            file_count="multiple", file_types=["image"],
                            elem_classes="drop",
                        )
                        batch_op = gr.Radio(
                            _BATCH_OPS, value="Upscale", label="Operation",
                            info="What to do to every image you dropped above. "
                            "Recipe runs a whole saved chain of edits.",
                        )
                        with gr.Column(visible=True) as batch_grp_up:
                            batch_model = gr.Dropdown(
                                _MODEL_CHOICES, value=_cfg_model,
                                label="Upscale model", filterable=True,
                            )
                            batch_size = gr.Dropdown(
                                list(_SIZE_PRESETS), value="Model default (×2/×4)",
                                label="Output size", filterable=False,
                            )
                            batch_sharpen = gr.Slider(
                                0.0, 3.0, value=0.0, step=0.1,
                                label="Sharpen edges — 0 = off",
                            )
                        with gr.Column(visible=False) as batch_grp_conv:
                            batch_fmt = gr.Dropdown(
                                list(FORMATS), value="PNG", label="Convert to",
                                filterable=False,
                            )
                            batch_quality = gr.Slider(
                                1, 100, value=90, step=1, label="Quality (lossy)",
                            )
                        with gr.Column(visible=False) as batch_grp_bg:
                            batch_bg_model = gr.Dropdown(
                                _BG_CHOICES, value=background.DEFAULT_BG_MODEL,
                                label="Model", filterable=False,
                            )
                            batch_feather = gr.Slider(
                                0, 10, value=1, step=1, label="Edge feather (px)",
                            )
                        with gr.Column(visible=False) as batch_grp_recipe:
                            batch_recipe = gr.Dropdown(
                                recipe_tools.BUILT_IN_NAMES,
                                value=recipe_tools.BUILT_IN_NAMES[0],
                                label="Recipe", filterable=False,
                                info="A saved chain of edits. Picking one loads it "
                                "below, where you can change it.",
                            )
                            batch_recipe_note = gr.Markdown(elem_classes="notes")
                            with gr.Accordion("The recipe itself (editable)", open=False):
                                batch_recipe_json = gr.Code(
                                    value=recipe_tools.to_json(
                                        recipe_tools.built_in(recipe_tools.BUILT_IN_NAMES[0])),
                                    language="json", label="Recipe JSON", lines=14,
                                )
                                gr.Markdown(
                                    "Steps run top to bottom. Each names a tool — "
                                    f"{', '.join(recipe_tools.TOOLS)} — and the settings "
                                    "that tool uses, so anything you can do in a tab you "
                                    "can put here. `region` restricts a step to a shape, "
                                    "every detected face, or depth.\n\n"
                                    "Save one to a file and run it over a folder from "
                                    "the terminal with `upscaler recipe my.json ./folder`.",
                                    elem_classes="notes",
                                )
                        with gr.Accordion("Advanced", open=False):
                            batch_device = gr.Dropdown(
                                _DEVICES, value=_cfg_device, label="Device",
                                filterable=False,
                            )
                            batch_tile = gr.Slider(
                                0, 1024, value=512, step=64,
                                label="Tile size (0 = off)",
                            )
                        with gr.Row():
                            batch_run = gr.Button("Process all", variant="primary",
                                                  size="lg", scale=3)
                            batch_cancel = gr.Button("✕ Cancel", variant="stop", scale=1)
                    with gr.Column(scale=1, elem_classes="sticky-col"):
                        batch_gallery = gr.Gallery(
                            label="Results", columns=3, height=420,
                            object_fit="cover", buttons=["download", "fullscreen"],
                            elem_classes=["loupe"],
                        )
                        batch_zip = gr.File(label="Download all (ZIP)")
                        batch_info = gr.Markdown()
                batch_op.change(
                    _switch_batch_op, batch_op,
                    [batch_grp_up, batch_grp_conv, batch_grp_bg, batch_grp_recipe],
                    show_progress="hidden",  # instant visibility toggle, no flash
                )
                batch_recipe.input(recipe_load_built_in, batch_recipe,
                                   [batch_recipe_json, batch_recipe_note],
                                   show_progress="hidden")
                batch_recipe_json.change(recipe_show_ui,
                                         [batch_recipe, batch_recipe_json],
                                         [batch_recipe_json, batch_recipe_note],
                                         show_progress="hidden",
                                         trigger_mode="always_last")
                batch_evt = batch_run.click(
                    batch_process,
                    [batch_in, batch_op, batch_model, batch_size, batch_sharpen,
                     batch_fmt, batch_quality, batch_bg_model, batch_feather,
                     batch_device, batch_tile, batch_recipe, batch_recipe_json],
                    [batch_gallery, batch_zip, batch_info],
                    show_progress_on=[batch_gallery],
                )
                batch_cancel.click(lambda: _BATCH_CANCEL.set(), None, None,
                                   cancels=[batch_evt])

            # ---- Tab: Lian Li 8.8" Screen builder ----
            with gr.Tab("Lian Li"):
                gr.HTML(_section_head(
                    "Panel", "Lian Li 8.8″ Screen",
                    "Compose media at the panel's exact size (1920×480 or 480×1920, "
                    "4:1) so L-Connect 3 never has to resample it. Fit any photo, "
                    "GIF or video into the frame — the dimmed area is what gets "
                    "cropped — then export a PNG, looping GIF or MP4.",
                    icon=ICON_PANEL,
                ))
                with gr.Row(equal_height=False):
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
                                label="Upscale model", filterable=True,
                                info="The AI that enlarges your source before it's "
                                "fitted. ×2 is plenty for 1920×480; use ×4 for very "
                                "small sources, or the anime model for line art.",
                            )
                            pn_enhance = gr.Button("✨ Enhance source", variant="secondary")
                        pn_orient = gr.Radio(
                            list(panel.ORIENTATIONS), value="Landscape · 1920×480",
                            label="Orientation",
                            info="Match how your panel is mounted — wide (Landscape) "
                            "or tall (Portrait).",
                        )
                        pn_fit = gr.Radio(
                            panel.FITS, value="cover", label="Fit",
                            info="How your media fills the 4:1 frame: cover fills and "
                            "crops, contain adds bars (letterboxes), stretch "
                            "distorts, manual lets you zoom and pan freely.",
                        )
                        with gr.Row():
                            pn_offx = gr.Slider(
                                -100, 100, value=0, step=1, label="Pan X (%)",
                                info="Slide left/right to choose which part survives "
                                "the crop (cover and manual fit).",
                            )
                            pn_offy = gr.Slider(
                                -100, 100, value=0, step=1, label="Pan Y (%)",
                                info="Slide up/down to choose which part survives "
                                "the crop (cover and manual fit).",
                            )
                        pn_zoom = gr.Slider(
                            0.1, 5, value=1, step=0.01, label="Zoom (manual fit)",
                            info="Zooms the image in or out — only used when Fit is "
                            "set to manual.",
                        )
                        with gr.Accordion("Background (fills letterbox gaps)", open=False):
                            pn_bgtype = gr.Radio(
                                ["solid", "gradient"], value="solid",
                                label="Type",
                                info="What fills any empty space (letterbox bars): "
                                "one solid color, or a two-color gradient.",
                            )
                            with gr.Row():
                                pn_bgcol = gr.ColorPicker(
                                    value="#000000", label="Color / Stop A",
                                    info="The fill color — or the first color of "
                                    "the gradient.",
                                )
                                pn_bgcol2 = gr.ColorPicker(
                                    value="#333333", label="Stop B",
                                    info="The second gradient color (only used when "
                                    "Type is gradient).",
                                )
                            pn_bgang = gr.Slider(
                                0, 360, value=90, step=1, label="Gradient angle",
                                info="Direction the gradient blends, in degrees "
                                "(90 = top to bottom).",
                            )
                        # Overlays: up to N_TEXT text layers + N_STICKER stickers.
                        # Each slot's components are collected (in field order) so
                        # the preview/export handlers can rebuild the overlay list.
                        _text_slots = []
                        with gr.Accordion("Text overlays", open=True):
                            for _t in range(N_TEXT):
                                with gr.Accordion(f"Text {_t + 1}", open=(_t == 0)):
                                    t_en = gr.Checkbox(value=(_t == 0), label="Show this text",
                                                       info="Tick to show this text layer.")
                                    t_content = gr.Textbox(label="Text", lines=2,
                                                           placeholder="(your text)",
                                                           info="The words to display — leave "
                                                           "empty to hide this layer.")
                                    with gr.Row():
                                        t_font = gr.Dropdown(panel.FONT_NAMES,
                                                             value=panel.DEFAULT_FONT,
                                                             label="Font", filterable=True,
                                                             info="The typeface.")
                                        t_size = gr.Slider(16, 900, value=180, step=2,
                                                           label="Size (px)",
                                                           info="Text height in pixels.")
                                    with gr.Row():
                                        t_color = gr.ColorPicker(value="#ffffff", label="Color",
                                                                 info="The text color.")
                                        t_align = gr.Radio(["left", "center", "right"],
                                                           value="center", label="Align",
                                                           info="Line up left, center or right.")
                                    with gr.Row():
                                        t_x = gr.Slider(-100, 100, value=0, step=1, label="X (%)",
                                                        info="Nudge left/right.")
                                        t_y = gr.Slider(-100, 100, value=0, step=1, label="Y (%)",
                                                        info="Nudge up/down.")
                                    t_rot = gr.Slider(-180, 180, value=0, step=1,
                                                      label="Rotation (°)",
                                                      info="Tilt the text (0 = straight).")
                                    with gr.Row():
                                        t_stroke = gr.ColorPicker(value="#000000", label="Stroke",
                                                                  info="Color of the outline "
                                                                  "around the text.")
                                        t_strokew = gr.Slider(0, 40, value=0, step=1,
                                                              label="Stroke width",
                                                              info="Outline thickness — "
                                                              "0 = no outline.")
                                    t_motion = gr.Dropdown(
                                        ["none", "scroll-left", "scroll-right",
                                         "scroll-up", "scroll-down", "fade", "typewriter"],
                                        value="none", label="Motion", filterable=False,
                                        info="Animates in the exported GIF/MP4 — the "
                                        "editor preview shows the starting frame.")
                                    with gr.Row():
                                        t_speed = gr.Slider(10, 600, value=120, step=10,
                                                            label="Scroll speed (px/s)",
                                                            info="How fast scrolling text "
                                                            "moves (snapped to loop cleanly).")
                                        t_cps = gr.Slider(1, 40, value=10, step=1,
                                                          label="Type speed (chars/s)",
                                                          info="Typewriter reveal rate.")
                                _text_slots.append([t_en, t_content, t_font, t_size, t_color,
                                                    t_align, t_x, t_y, t_rot, t_stroke, t_strokew,
                                                    t_motion, t_speed, t_cps])
                        _sticker_slots = []
                        with gr.Accordion("Stickers (image overlays)", open=False):
                            for _s in range(N_STICKER):
                                with gr.Accordion(f"Sticker {_s + 1}", open=False):
                                    s_en = gr.Checkbox(value=False, label="Show this sticker",
                                                       info="Tick to show this image sticker.")
                                    # image_mode="RGBA" preserves transparency —
                                    # without it Gradio drops alpha and PNG cut-outs
                                    # composite as opaque black.
                                    s_img = gr.Image(label="Sticker image (a see-through PNG — "
                                                     "e.g. a Remove-BG cut-out — works best)",
                                                     type="pil", image_mode="RGBA",
                                                     sources=["upload", "clipboard"], height=120)
                                    with gr.Row():
                                        s_scale = gr.Slider(2, 100, value=40, step=1,
                                                            label="Size (% of panel height)",
                                                            info="Sticker size, relative to the "
                                                            "panel's height.")
                                        s_op = gr.Slider(0, 1, value=1, step=0.01, label="Opacity",
                                                         info="How see-through it is — "
                                                         "1 = solid, 0 = invisible.")
                                    with gr.Row():
                                        s_x = gr.Slider(-100, 100, value=0, step=1, label="X (%)",
                                                        info="Nudge left/right.")
                                        s_y = gr.Slider(-100, 100, value=0, step=1, label="Y (%)",
                                                        info="Nudge up/down.")
                                    s_rot = gr.Slider(-180, 180, value=0, step=1,
                                                      label="Rotation (°)",
                                                      info="Tilt the sticker (0 = straight).")
                                _sticker_slots.append([s_en, s_img, s_scale, s_x, s_y, s_rot, s_op])
                        _clock_slots = []
                        with gr.Accordion("Clock / date", open=False):
                            gr.Markdown(
                                "*The time is **baked in at export** — a clip replays "
                                "the moments captured when you exported, it isn't a live "
                                "wall-clock in L-Connect.*", elem_classes="notes",
                            )
                            for _k in range(N_CLOCK):
                                k_en = gr.Checkbox(value=False, label="Show a clock / date",
                                                   info="Tick to overlay the time/date.")
                                k_tmpl = gr.Dropdown(
                                    ["%H:%M:%S", "%H:%M", "%I:%M %p", "%a %d %b",
                                     "%Y-%m-%d", "%d/%m %H:%M"],
                                    value="%H:%M:%S", label="Format", allow_custom_value=True,
                                    info="strftime template — %H hour %M min %S sec, "
                                    "%a day %d date %b month %Y year. Type your own too.")
                                with gr.Row():
                                    k_font = gr.Dropdown(panel.FONT_NAMES,
                                                         value=panel.DEFAULT_FONT,
                                                         label="Font", filterable=True)
                                    k_size = gr.Slider(16, 900, value=180, step=2,
                                                       label="Size (px)")
                                with gr.Row():
                                    k_color = gr.ColorPicker(value="#ffffff", label="Color")
                                    k_align = gr.Radio(["left", "center", "right"],
                                                       value="center", label="Align")
                                with gr.Row():
                                    k_x = gr.Slider(-100, 100, value=0, step=1, label="X (%)")
                                    k_y = gr.Slider(-100, 100, value=0, step=1, label="Y (%)")
                                k_rot = gr.Slider(-180, 180, value=0, step=1,
                                                  label="Rotation (°)")
                                with gr.Row():
                                    k_stroke = gr.ColorPicker(value="#000000", label="Stroke")
                                    k_strokew = gr.Slider(0, 40, value=0, step=1,
                                                          label="Stroke width")
                                _clock_slots.append([k_en, k_tmpl, k_font, k_size, k_color,
                                                     k_align, k_x, k_y, k_rot, k_stroke, k_strokew])
                        _overlay_inputs = [c for slot in _text_slots for c in slot] + \
                                          [c for slot in _sticker_slots for c in slot] + \
                                          [c for slot in _clock_slots for c in slot]
                        with gr.Accordion("Layouts (save / share)", open=False):
                            gr.Markdown(
                                "*Saves your composition + overlays (stickers are "
                                "embedded, so the file is self-contained) — **not** the "
                                "source media. Re-upload your image/video after "
                                "loading a layout.*", elem_classes="notes",
                            )
                            pn_layout_save = gr.Button("💾 Save layout (.json)",
                                                       variant="secondary", size="sm")
                            pn_layout_file = gr.File(label="Download layout")
                            pn_layout_upload = gr.File(label="Load a layout (.json)",
                                                       file_types=[".json"], height=90)
                        with gr.Group(visible=False) as pn_anim_group:
                            gr.Markdown("**Animation** — for GIF / MP4 export.")
                            with gr.Row():
                                pn_start = gr.Number(value=0, label="Trim start (s)", minimum=0,
                                                     info="Skip everything before this point.")
                                pn_end = gr.Number(value=0, label="Trim end (s)", minimum=0,
                                                   info="Stop here (0 = play to the end).")
                            with gr.Row():
                                pn_fps = gr.Dropdown(
                                    ["10", "12", "15", "24", "25", "30", "48", "50", "60"],
                                    value="30", label="FPS (≤ 60)", filterable=False,
                                    info="Frames per second — higher is smoother but a "
                                    "bigger file.",
                                )
                                pn_loop = gr.Checkbox(value=True, label="Loop",
                                                      info="Make the GIF / MP4 repeat forever.")
                            pn_colors = gr.Slider(
                                2, 256, value=128, step=1, label="GIF colors",
                                info="How many colors the GIF uses — more is richer but "
                                "a bigger file (GIF only).",
                            )
                            pn_loopmode = gr.Radio(
                                panel.LOOP_STYLES, value="normal", label="Loop style",
                                info="How it loops: normal restarts, boomerang plays "
                                "forward then back, crossfade blends the end into the "
                                "start — boomerang and crossfade both hide the seam.",
                            )
                        pn_fmt = gr.Radio(
                            ["PNG", "JPG", "GIF", "MP4"], value="GIF",
                            label="Export format",
                            info="PNG / JPG = a still image · GIF / MP4 = an "
                            "animation.",
                        )
                        pn_outdir = gr.Textbox(
                            value=cfg["output_dir"],
                            label="Save a copy to folder (optional)",
                            placeholder="/path/to/your L-Connect media folder",
                            info="On export, also drops a timestamped copy into this "
                            "folder (e.g. your L-Connect media folder), on top of the "
                            "normal download.",
                        )
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "* **Cover fit suits most photos**; use contain when "
                                "you can't crop any edges.\n"
                                "* **Use Pan X / Pan Y to reframe** which part "
                                "survives a cover crop.\n"
                                "* **Enhance the source only if it's low-res** and "
                                "looks soft on the panel; ×2 is plenty.\n"
                                "* **For a looping GIF / MP4, pick crossfade or "
                                "boomerang** to hide the repeat seam.\n"
                                "* **Everything exports at the panel's exact 4:1 "
                                "size** so L-Connect 3 never resamples it.",
                                elem_classes="notes",
                            )
                        with gr.Row():
                            pn_export = gr.Button("Export", variant="primary", size="lg", scale=3)
                            pn_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1, elem_classes="sticky-col"):
                        pn_preview = gr.Image(
                            label="Preview — bright = kept, dim = cropped out",
                            height=300, buttons=["fullscreen"], elem_classes=["loupe"],
                        )
                        pn_mockup_btn = gr.Button(
                            "🖥️ See it on the screen (3D)", variant="secondary",
                            size="sm",
                        )
                        pn_mockup = gr.Image(
                            label="On the Lian Li panel — a 3D mockup",
                            height=260, buttons=["download", "fullscreen"],
                            elem_classes=["loupe"],
                        )
                        pn_file = gr.File(label="Download export")
                        pn_info = gr.Markdown()

            # ---- Tab: Steam Workshop Showcase (five tiles from one picture/clip) ----
            with gr.Tab("Steam"):
                gr.HTML(_section_head(
                    "Steam", "Workshop Showcase tiles",
                    "Turn a photo, GIF or video into the five tiles Steam shows side by "
                    "side in your profile's Workshop Showcase. Tiles are cut at Steam's "
                    "exact widths and gaps so the picture lines up across all five, and "
                    "clips export as looping animated PNGs shrunk to fit the upload limit.",
                    icon=ICON_STEAM,
                ))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        st_media = gr.File(
                            label="Image, GIF or video",
                            file_count="single",
                            file_types=["image", ".gif", ".mp4", ".mov", ".webm",
                                        ".mkv", ".m4v", ".avi"],
                            elem_classes="drop",
                        )
                        st_preset = gr.Dropdown(
                            steam.PRESETS, value=steam.PRESET_AUTO, label="Preset",
                            filterable=False,
                            info="Shapes the row from your media's aspect ratio. Auto: a "
                            "portrait clip (TikTok / Reels) repeats in every tile at full "
                            "height, anything else spans the row uncropped. Pick "
                            "another to compare; editing a control below switches to "
                            "Custom.",
                        )
                        st_fit = gr.Radio(
                            steam.FITS, value="cover", label="Fit",
                            info="How your media fills the row: cover fills and crops, "
                            "contain adds bars, stretch distorts, manual lets you zoom "
                            "and pan freely.",
                        )
                        with gr.Row():
                            st_offx = gr.Slider(
                                -100, 100, value=0, step=1, label="Pan X (%)",
                                info="Slide left/right to choose which part survives "
                                "the crop (cover and manual fit).",
                            )
                            st_offy = gr.Slider(
                                -100, 100, value=0, step=1, label="Pan Y (%)",
                                info="Slide up/down to choose which part survives "
                                "the crop (cover and manual fit).",
                            )
                        st_zoom = gr.Slider(
                            0.1, 5, value=1, step=0.01, label="Zoom (manual fit)",
                            info="Zooms the image in or out — only used when Fit is "
                            "set to manual.",
                        )
                        with gr.Row():
                            st_tilew = gr.Slider(
                                steam.MIN_TILE_W, steam.MAX_TILE_W, value=steam.DEFAULT_TILE_W,
                                step=1, label="Tile width (px)",
                                info="Pixel width of each exported tile. Steam always shows "
                                "a tile 122px wide, so 122 is pixel-for-pixel; 150 is the "
                                "size most guides use; 245 stays crisp on Retina / HiDPI "
                                "screens. The gaps scale with it.",
                            )
                            st_tileh = gr.Slider(
                                steam.MIN_TILE_H, steam.MAX_TILE_H, value=steam.DEFAULT_TILE_H,
                                step=1, label="Tile height (px)",
                                info="Pixel height of each tile — same as the width makes "
                                "square tiles (the stock look), taller makes a bigger "
                                "banner. Steam scales it with the width.",
                            )
                        st_repeat = gr.Checkbox(
                            value=False, label="Repeat the source in every tile",
                            info="Fit the whole picture into one tile and stamp it into "
                            "all five, instead of spreading one picture across the row.",
                        )
                        with gr.Row():
                            st_bg = gr.ColorPicker(
                                value="#000000", label="Background (fills letterbox gaps)",
                                info="Shows through wherever the media doesn't cover the "
                                "row (contain fit, or a zoomed-out manual fit).",
                            )
                            st_transparent = gr.Checkbox(
                                value=False, label="Transparent background",
                                info="Leave uncovered areas see-through so Steam's own "
                                "backdrop shows — great with a cut-out PNG (Remove BG) "
                                "or a contain fit. Ignores the colour.",
                            )
                        with gr.Group(visible=False) as st_anim_group:
                            gr.Markdown("**Animation** — for APNG / GIF export.")
                            with gr.Row():
                                st_start = gr.Number(value=0, label="Trim start (s)", minimum=0,
                                                     info="Skip everything before this point.")
                                st_end = gr.Number(value=0, label="Trim end (s)", minimum=0,
                                                   info="Stop here (0 = play to the end).")
                            with gr.Row():
                                st_fps = gr.Dropdown(
                                    _STEAM_FPS, value="24", label="FPS", filterable=False,
                                    info="Frames per second — higher is smoother but a "
                                    "bigger file; the budget may lower it.",
                                )
                                st_loopmode = gr.Radio(
                                    steam.LOOP_STYLES, value="normal", label="Loop style",
                                    info="normal restarts, boomerang plays forward then "
                                    "back, crossfade blends the end into the start.",
                                )
                            st_maxmb = gr.Slider(
                                0, steam.STEAM_MAX_MB, value=steam.DEFAULT_MAX_MB, step=0.5,
                                label="Size budget per tile (MB)",
                                info="Each tile is shrunk in steps (256 colours → lower "
                                "fps → shorter clip) until it fits. Steam documents 8 MB; "
                                "5 MB is the safe bet. 0 = no limit.",
                            )
                        st_fmt = gr.Radio(
                            [_STEAM_FMT_STILL, _STEAM_FMT_ANIM, _STEAM_FMT_GIF],
                            value=_STEAM_FMT_STILL, label="Export format",
                            info="PNG = five stills from the first frame · APNG = five "
                            "looping animations in full colour · GIF = the same in "
                            "256 colours (smaller files; the format most Steam guides "
                            "use).",
                        )
                        st_hexify = gr.Checkbox(
                            value=True, label="Hexify for Steam upload",
                            info="Sets each tile's last byte to 21 — the hex-editor step "
                            "the guides describe — so Steam keeps the animation instead "
                            "of flattening it. Files still open everywhere; untick only "
                            "if you want untouched files.",
                        )
                        st_outdir = gr.Textbox(
                            value=cfg["output_dir"],
                            label="Save a copy to folder (optional)",
                            placeholder="/path/to/a/folder",
                            info="On export, also drops the five tiles into this folder, "
                            "on top of the normal download.",
                        )
                        with gr.Accordion("How to upload to Steam", open=False):
                            gr.Markdown(steam.UPLOAD_GUIDE, elem_classes="notes")
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "* **Start from a preset** — Auto reads your clip's shape: "
                                "a TikTok / Reels clip gets five full-height copies, a "
                                "widescreen clip spans the row uncropped. 'Centre tile "
                                "only' floats one tile in the middle.\n"
                                "* **Wide sources suit the row** — a 16:9 clip at 122px "
                                "tall shows only a slim band; raise Tile height or pan "
                                "to the part that matters.\n"
                                "* **The black bars in the preview are Steam's gaps** "
                                "between tiles — anything under them is never shown, so "
                                "keep faces and text off the seams.\n"
                                "* **Short loops upload best**: 3–6 seconds at 15–24 fps "
                                "fits the budget with room to spare.\n"
                                "* **Boomerang or crossfade hide the repeat seam** on "
                                "clips that don't naturally loop.\n"
                                "* **Upload the tiles in order 1 → 5** — the ZIP names "
                                "them for you.",
                                elem_classes="notes",
                            )
                        with gr.Row():
                            st_export = gr.Button("Export tiles", variant="primary", size="lg", scale=3)
                            st_cancel = gr.Button("✕ Cancel", variant="stop", scale=1)
                            st_clear = gr.Button("↺ Clear", variant="secondary", scale=1)
                    with gr.Column(scale=1, elem_classes="sticky-col"):
                        st_preview = gr.Image(
                            label="Preview — framing (bright = kept, black bars = Steam's "
                            "gaps) and how it sits on your profile",
                            height=380, buttons=["fullscreen"], elem_classes=["loupe"],
                        )
                        st_gallery = gr.Gallery(
                            label="Tiles 1 → 5 (animated tiles play here)", columns=5,
                            height=200, object_fit="contain",
                            buttons=["download", "fullscreen"], elem_classes=["loupe"],
                        )
                        st_file = gr.File(label="Download all five (ZIP)")
                        st_info = gr.Markdown()

            # ---- Tab: Library (everything you export, saved automatically) ----
            with gr.Tab("Library") as lib_tab:
                gr.HTML(_section_head(
                    "Library", "Your Library",
                    "Everything you export is saved here automatically, so your "
                    "creations are easy to find and reuse. Browse images and GIFs, "
                    "preview your videos, or open the folder to manage the files.",
                    icon=ICON_LIBRARY,
                ))
                with gr.Row(elem_classes="toolbar"):
                    lib_refresh = gr.Button("🔄 Refresh", variant="secondary", size="sm")
                    lib_open = gr.Button("📂 Open folder", variant="secondary", size="sm")
                lib_count = gr.Markdown()
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        lib_gallery = gr.Gallery(
                            label="Images & GIFs", columns=4, height=560,
                            object_fit="cover", buttons=["download", "fullscreen"],
                            elem_classes=["loupe"],
                        )
                    with gr.Column(scale=2):
                        lib_video_pick = gr.Dropdown(
                            label="Your videos", filterable=False,
                        )
                        lib_video = gr.Video(label="Preview", buttons=["download"])

        # ---- Settings: its own page, opened by the ⚙ gear (hidden by default) ----
        with gr.Column(visible=False) as settings_view:
            with gr.Row(elem_classes="toolbar"):
                set_back = gr.Button("← Back to the app", variant="secondary", size="sm")
            gr.HTML(_section_head(
                "Settings", "Settings & Setup",
                "Set your defaults once, see where your files live, and follow the "
                "step-by-step guide to run Upscaler on a Windows PC.",
                icon=ICON_SETTINGS,
            ))
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Preferences", open=True):
                        set_device = gr.Dropdown(
                            _DEVICES, value=_cfg_device, label="Default device",
                            filterable=False,
                            info="Where work runs by default — \"auto\" uses your "
                            "graphics card (GPU) when it can, otherwise the CPU.",
                        )
                        set_model = gr.Dropdown(
                            _MODEL_CHOICES, value=_cfg_model,
                            label="Default upscale model", filterable=True,
                            info="The model pre-selected on the Upscale tab when "
                            "the app starts.",
                        )
                        set_outdir = gr.Textbox(
                            value=cfg["output_dir"], label="Default save-to folder",
                            placeholder="/path/to/a folder (optional)",
                            info="Pre-fills the Lian Li \"save a copy to folder\" box.",
                        )
                        set_save = gr.Button("Save preferences", variant="primary")
                        set_status = gr.Markdown()
                        gr.Markdown(
                            "*Stored in `~/.upscaler/config.json` · applied "
                            "immediately, and used as the defaults on every start.*"
                        )
                    with gr.Accordion("Where your files live", open=False):
                        gr.Markdown(
                            f"- **Library (your exports):** `{library.LIBRARY_DIR}`\n"
                            f"- **Preferences:** `{config.CONFIG_PATH}`\n\n"
                            "Everything you make is saved to the Library "
                            "automatically — browse it in the **Library** tab."
                        )
                        set_open_lib = gr.Button(
                            "📂 Open library folder", variant="secondary", size="sm"
                        )
                    with gr.Accordion("Models & downloads", open=False):
                        _mm_init_rows, _mm_init_total = _mm_rows()
                        mm_total = gr.Markdown(_mm_init_total)
                        mm_table = gr.Dataframe(
                            value=_mm_init_rows,
                            headers=["Group", "Model", "File", "Status", "Size"],
                            interactive=False, wrap=True,
                        )
                        mm_pick = gr.Dropdown(
                            [s.filename for s in manage.list_specs()],
                            label="Model file",
                            info="Pre-download a model so the first use is instant, or "
                            "remove it to reclaim disk space (it re-downloads on next "
                            "use).",
                        )
                        with gr.Row():
                            mm_dl = gr.Button("⬇ Download", variant="primary", scale=2)
                            mm_rm = gr.Button("🗑 Remove", variant="secondary", scale=2)
                            mm_refresh = gr.Button("↻ Refresh", variant="secondary",
                                                   scale=1)
                        mm_status = gr.Markdown()
                    with gr.Accordion("Diagnostics", open=False):
                        gr.Markdown(
                            "*Nothing here is uploaded — copy it into a bug report "
                            "yourself.*"
                        )
                        diag = gr.Code(
                            value=manage.system_report(), label="System report",
                            interactive=False,
                        )
                        diag_refresh = gr.Button("↻ Refresh report",
                                                 variant="secondary", size="sm")
                    with gr.Accordion("About", open=False):
                        gr.Markdown(
                            f"**Upscaler** · running locally on **{device_name}** · "
                            "powered by Real-ESRGAN + NAFNet. All processing happens "
                            "on this machine — no cloud, no API keys, your files are "
                            "never uploaded. (Model weights and the UI font are the "
                            "only things fetched from the web.)"
                        )
                with gr.Column(scale=1):
                    with gr.Accordion("Install on Windows — step by step", open=True):
                        gr.Markdown(WINDOWS_GUIDE)

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
        col_btn.click(
            colorize_ui, [col_in, col_model, col_strength],
            [col_out, col_info], show_progress_on=[col_out],
        )
        col_clear.click(lambda: (None, None, None), None, [col_in, col_out, col_info])
        ip_btn.click(
            inpaint_ui, [ip_editor, ip_model],
            [ip_out, ip_info], show_progress_on=[ip_out],
        )
        ip_clear.click(lambda: (None, None, None), None, [ip_editor, ip_out, ip_info])
        pdf_extract_btn.click(
            extract_pdf, [pdf_in, pdf_dpi], [pdf_extract_out, pdf_gallery, pdf_extract_info]
        )
        # The crop controls only mean anything for an exact-size target, and the
        # custom box only for "Custom size…". The preview goes further: it also
        # needs an image and a parseable size, which _crop_preview decides — all
        # roads below go through it so the preview can never disagree with the
        # crop the job will actually make.
        def _on_out_size(choice, image, custom, pos):
            is_exact = choice == _EXACT_CUSTOM or choice in fit.TARGET_PRESETS
            return (
                gr.update(visible=choice == _EXACT_CUSTOM),
                gr.update(visible=is_exact),
                gr.update(visible=is_exact),
                _crop_preview(image, choice, custom, pos),
            )

        out_size.change(
            _on_out_size, [out_size, inp, custom_size, crop_position],
            [custom_size, crop_anchor, crop_position, crop_preview],
            show_progress="hidden",
        )

        # The dropdown is a preset for the slider (left/top → 0, center → 50,
        # right/bottom → 100); the slider's value is what the job reads.
        def _on_crop_anchor(anchor, image, choice, custom):
            pos = {"left": 0, "top": 0, "right": 100, "bottom": 100}.get(anchor, 50)
            return pos, _crop_preview(image, choice, custom, pos)

        crop_anchor.change(
            _on_crop_anchor, [crop_anchor, inp, out_size, custom_size],
            [crop_position, crop_preview], show_progress="hidden",
        )
        # .release, not .change: re-render once when the drag ends, not on every
        # tick through the middle. Everything else that shifts the crop re-renders
        # too, so the preview never shows a stale frame.
        crop_position.release(
            _crop_preview, [inp, out_size, custom_size, crop_position],
            crop_preview, show_progress="hidden",
        )
        inp.change(
            _crop_preview, [inp, out_size, custom_size, crop_position],
            crop_preview, show_progress="hidden",
        )
        custom_size.change(
            _crop_preview, [inp, out_size, custom_size, crop_position],
            crop_preview, show_progress="hidden",
        )
        run_evt = run.click(
            enhance,
            [inp, model, device, deblur, deblur_model, restore_strength, sharpen,
             tile, onnx, out_size, face, face_strength, face_model, face_fidelity,
             fbcnn, custom_size, crop_position],
            [out, info],
            show_progress_on=[out],
        )
        # Set the cooperative flag (stops the tile loop) AND cancel the Gradio
        # event (stops the progress stream) — either alone leaves work running.
        enh_cancel.click(lambda: _ENHANCE_CANCEL.set(), None, None,
                         cancels=[run_evt])
        restore_btn.click(
            restore_only,
            [inp, deblur_model, restore_strength, sharpen, device, onnx, fbcnn],
            [out, info],
            show_progress_on=[out],
        )
        _preset_outputs = [
            model, sharpen, deblur, deblur_model, restore_strength, preset_info,
        ] + _preset_buttons
        for _btn, _name in zip(_preset_buttons, UPSCALE_PRESETS):
            _btn.click(lambda n=_name: apply_preset(n), None, _preset_outputs,
                       show_progress="hidden")
        clear.click(lambda: (None, None, None), None, [inp, out, info])
        vid_evt = vid_btn.click(
            upscale_video_ui,
            [vid_in, vid_model, vid_size, vid_sharpen, vid_smooth, vid_start,
             vid_end, vid_device, vid_tile, vid_onnx],
            [vid_out, vid_compare, vid_info, vid_scrub],
            show_progress_on=[vid_out],
        )
        vid_cancel.click(lambda: _VIDEO_CANCEL.set(), None, None,
                         cancels=[vid_evt])
        vid_scrub.release(
            video_compare_at, [vid_in, vid_out, vid_scrub, vid_start],
            vid_compare, show_progress="hidden",
        )
        vid_clear.click(
            lambda: (None, None, None, None, gr.update(visible=False)), None,
            [vid_in, vid_out, vid_compare, vid_info, vid_scrub],
        )
        vid_in.change(_on_video_change, vid_in, [vid_start, vid_end])

        # ---- Lian Li panel builder wiring ----
        # Controls that affect the static composite → live preview. Order must
        # match _panel_params: base controls, then the flat overlay-slot values.
        _pn_preview_inputs = [
            pn_media, pn_orient, pn_fit, pn_zoom, pn_offx, pn_offy, pn_bgtype,
            pn_bgcol, pn_bgcol2, pn_bgang,
        ] + _overlay_inputs
        # pn_media is a gr.File (no .input event) and must re-render on upload
        # and when "Enhance source" swaps the file in — .change covers both.
        pn_media.change(
            panel_preview_ui, _pn_preview_inputs, pn_preview,
            show_progress="hidden",
        )
        for _c in _pn_preview_inputs[1:]:
            # .input (not .change) so only USER edits re-render: programmatic
            # updates — e.g. loading a layout writes all 76 components at once —
            # would otherwise fire 76 queued re-renders. show_progress="hidden"
            # removes the loading flash on every slider tick, and
            # trigger_mode="always_last" collapses a drag storm to one render.
            _c.input(
                panel_preview_ui, _pn_preview_inputs, pn_preview,
                show_progress="hidden", trigger_mode="always_last",
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
        pn_mockup_btn.click(
            panel_mockup_ui, _pn_preview_inputs, pn_mockup,
            show_progress_on=[pn_mockup],
        )
        pn_layout_save.click(
            panel_layout_download, _pn_preview_inputs[1:], pn_layout_file,
        )
        pn_layout_upload.upload(
            panel_layout_upload, pn_layout_upload, _pn_preview_inputs[1:],
        ).then(panel_preview_ui, _pn_preview_inputs, pn_preview)

        # ---- Color & light wiring ----
        _ad_controls = [ad_exposure, ad_contrast, ad_highlights, ad_shadows, ad_black,
                        ad_white, ad_gamma, ad_clarity, ad_temp, ad_tint, ad_hue,
                        ad_saturation, ad_vibrance, ad_mono, ad_mr, ad_mg, ad_mb,
                        ad_tone, ad_tone_strength]      # order must match _ADJUST_FIELDS
        _ad_region = [ad_shape, ad_x, ad_y, ad_w, ad_h, ad_mangle, ad_round, ad_feather,
                      ad_outside, ad_facepad, ad_faces_state, ad_editor]
        _ad_inputs = [ad_in] + _ad_controls + _ad_region
        ad_shape.change(
            _region_vis, ad_shape,
            [ad_x, ad_y, ad_w, ad_h, ad_mangle, ad_round, ad_feather, ad_outside,
             ad_facepad, ad_editor],
            show_progress="hidden",
        ).then(detect_faces_ui, [ad_in, ad_shape], [ad_faces_state, ad_faces_note],
               show_progress="hidden") \
         .then(adjust_preview_ui, _ad_inputs, ad_preview, show_progress="hidden")
        ad_in.change(blur_on_image, ad_in, ad_editor, show_progress="hidden")
        ad_in.change(detect_faces_ui, [ad_in, ad_shape], [ad_faces_state, ad_faces_note],
                     show_progress="hidden") \
             .then(adjust_preview_ui, _ad_inputs, ad_preview, show_progress="hidden")
        for _c in _ad_controls + _ad_region[1:]:
            if isinstance(_c, gr.State):   # holds the detected faces; fires no events
                continue
            (_c.change if _c is ad_editor else _c.input)(
                adjust_preview_ui, _ad_inputs, ad_preview,
                show_progress="hidden", trigger_mode="always_last",
            )
        # A preset or Auto writes every control at once, which fires .change and
        # not .input — so the preview is refreshed explicitly afterwards.
        ad_preset.input(adjust_preset_ui, ad_preset, _ad_controls, show_progress="hidden") \
            .then(adjust_preview_ui, _ad_inputs, ad_preview, show_progress="hidden")
        ad_auto.click(adjust_auto_ui, _ad_inputs, _ad_controls, show_progress="hidden") \
            .then(adjust_preview_ui, _ad_inputs, ad_preview, show_progress="hidden")
        ad_btn.click(
            adjust_apply_ui, _ad_inputs, [ad_out, ad_file, ad_info], show_progress_on=[ad_out],
        )
        ad_use.click(lambda pair: (pair[1] if pair else None), ad_out, ad_in)
        ad_clear.click(lambda: (None, None, None, None, None), None,
                       [ad_in, ad_preview, ad_out, ad_file, ad_info])

        # ---- Effects wiring ----
        _fx_controls = [fx_grain, fx_grain_size, fx_halation, fx_hal_thresh, fx_hal_radius,
                        fx_hal_color, fx_leak, fx_leak_angle, fx_leak_color, fx_leak_soft,
                        fx_vignette, fx_vig_radius, fx_vig_feather, fx_aberration,
                        fx_duotone, fx_duo_dark, fx_duo_light, fx_posterize, fx_dither,
                        fx_dither_levels, fx_halftone, fx_ht_cell, fx_ht_angle,
                        fx_scanlines, fx_scan_spacing, fx_glitch, fx_glitch_seed]
        _fx_region = [fx_shape, fx_x, fx_y, fx_w, fx_h, fx_mangle, fx_round, fx_feather,
                      fx_outside, fx_facepad, fx_faces_state, fx_editor]
        _fx_inputs = [fx_in] + _fx_controls + _fx_region
        fx_shape.change(
            _region_vis, fx_shape,
            [fx_x, fx_y, fx_w, fx_h, fx_mangle, fx_round, fx_feather, fx_outside,
             fx_facepad, fx_editor],
            show_progress="hidden",
        ).then(detect_faces_ui, [fx_in, fx_shape], [fx_faces_state, fx_faces_note],
               show_progress="hidden") \
         .then(effects_preview_ui, _fx_inputs, fx_preview, show_progress="hidden")
        fx_in.change(blur_on_image, fx_in, fx_editor, show_progress="hidden")
        fx_in.change(detect_faces_ui, [fx_in, fx_shape], [fx_faces_state, fx_faces_note],
                     show_progress="hidden") \
             .then(effects_preview_ui, _fx_inputs, fx_preview, show_progress="hidden")
        for _c in _fx_controls + _fx_region[1:]:
            if isinstance(_c, gr.State):
                continue
            (_c.change if _c is fx_editor else _c.input)(
                effects_preview_ui, _fx_inputs, fx_preview,
                show_progress="hidden", trigger_mode="always_last",
            )
        # A look writes every control at once, which fires .change and not
        # .input — so the preview is refreshed explicitly afterwards.
        fx_preset.input(effects_preset_ui, fx_preset, _fx_controls, show_progress="hidden") \
            .then(effects_preview_ui, _fx_inputs, fx_preview, show_progress="hidden")
        fx_btn.click(
            effects_apply_ui, _fx_inputs, [fx_out, fx_file, fx_info], show_progress_on=[fx_out],
        )
        fx_use.click(lambda pair: (pair[1] if pair else None), fx_out, fx_in)
        fx_clear.click(lambda: (None, None, None, None, None), None,
                       [fx_in, fx_preview, fx_out, fx_file, fx_info])

        # ---- Blur toolbox wiring ----
        _bl_inputs = [bl_in, bl_kind, bl_strength, bl_angle, bl_cx, bl_cy, bl_highlights,
                      bl_threshold, bl_shape, bl_x, bl_y, bl_w, bl_h, bl_mangle, bl_round,
                      bl_feather, bl_outside, bl_progressive, bl_facepad, bl_faces_state,
                      bl_editor, bl_depth_state, bl_focus, bl_dof]
        bl_kind.change(
            _blur_kind_vis, bl_kind,
            [bl_angle, bl_cx, bl_cy, bl_highlights, bl_threshold, bl_kind],
            show_progress="hidden",
        )
        bl_shape.change(
            _blur_shape_vis, bl_shape,
            [bl_x, bl_y, bl_w, bl_h, bl_mangle, bl_round, bl_feather, bl_outside,
             bl_progressive, bl_facepad, bl_editor],
            show_progress="hidden",
        ).then(_blur_depth_vis, bl_shape, [bl_focus, bl_dof], show_progress="hidden") \
         .then(detect_faces_ui, [bl_in, bl_shape], [bl_faces_state, bl_faces_note],
               show_progress="hidden") \
         .then(estimate_depth_ui, [bl_in, bl_shape],
               [bl_depth_state, bl_depth_view, bl_depth_note]) \
         .then(blur_preview_ui, _bl_inputs, bl_preview, show_progress="hidden")
        bl_in.change(blur_on_image, bl_in, bl_editor, show_progress="hidden")
        bl_in.change(detect_faces_ui, [bl_in, bl_shape], [bl_faces_state, bl_faces_note],
                     show_progress="hidden") \
             .then(estimate_depth_ui, [bl_in, bl_shape],
                   [bl_depth_state, bl_depth_view, bl_depth_note]) \
             .then(blur_preview_ui, _bl_inputs, bl_preview, show_progress="hidden")
        bl_depth_view.select(depth_focus_from_click, bl_depth_state, bl_focus,
                             show_progress="hidden") \
            .then(blur_preview_ui, _bl_inputs, bl_preview, show_progress="hidden")
        for _c in _bl_inputs[1:]:
            if _c is bl_shape or isinstance(_c, gr.State):
                continue   # shape is handled above; State fires no events
            (_c.change if _c is bl_editor else _c.input)(
                blur_preview_ui, _bl_inputs, bl_preview,
                show_progress="hidden", trigger_mode="always_last",
            )
        bl_btn.click(
            blur_apply_ui, _bl_inputs, [bl_out, bl_file, bl_info], show_progress_on=[bl_out],
        )
        bl_use.click(lambda pair: (pair[1] if pair else None), bl_out, bl_in)
        bl_clear.click(lambda: (None, None, None, None, None), None,
                       [bl_in, bl_preview, bl_out, bl_file, bl_info])

        # ---- Watermark wiring ----
        _wm_controls = [wm_kind, wm_text, wm_font, wm_size, wm_color, wm_outline,
                        wm_outline_w, wm_shadow, wm_logo_scale, wm_position, wm_margin,
                        wm_rotation, wm_opacity, wm_tile_gap, wm_tile_angle]
        _wm_controls += [wm_behind, wm_cut_feather, wm_subject_shadow]
        # order after the three fixed inputs must match _WM_FIELDS
        _wm_inputs = [wm_in, wm_logo, wm_cutout_state] + _wm_controls
        _wm_vis_out = [wm_text, wm_font, wm_size, wm_color, wm_outline, wm_outline_w,
                       wm_shadow, wm_logo, wm_logo_scale, wm_margin, wm_tile_gap,
                       wm_tile_angle]
        for _c in (wm_kind, wm_position):
            _c.change(_wm_vis, [wm_kind, wm_position], _wm_vis_out, show_progress="hidden")
        for _c in (wm_behind, wm_cut_feather):
            _c.change(wm_cutout_ui, [wm_in, wm_behind, wm_cut_feather],
                      [wm_cutout_state, wm_behind_note]) \
                .then(watermark_preview_ui, _wm_inputs, [wm_preview, wm_note],
                      show_progress="hidden")
        wm_behind.change(_wm_behind_vis, wm_behind,
                         [wm_cut_feather, wm_subject_shadow], show_progress="hidden")
        wm_in.change(wm_cutout_ui, [wm_in, wm_behind, wm_cut_feather],
                     [wm_cutout_state, wm_behind_note]) \
            .then(watermark_preview_ui, _wm_inputs, [wm_preview, wm_note],
                  show_progress="hidden")
        for _c in [wm_logo] + _wm_controls:
            if _c in (wm_behind, wm_cut_feather):
                continue                      # handled above: cut-out first, then preview
            _c.change(watermark_preview_ui, _wm_inputs, [wm_preview, wm_note],
                      show_progress="hidden", trigger_mode="always_last")
        wm_preset.input(watermark_preset_ui, wm_preset, _wm_controls,
                        show_progress="hidden") \
            .then(_wm_vis, [wm_kind, wm_position], _wm_vis_out, show_progress="hidden") \
            .then(_wm_behind_vis, wm_behind, [wm_cut_feather, wm_subject_shadow],
                  show_progress="hidden") \
            .then(wm_cutout_ui, [wm_in, wm_behind, wm_cut_feather],
                  [wm_cutout_state, wm_behind_note]) \
            .then(watermark_preview_ui, _wm_inputs, [wm_preview, wm_note],
                  show_progress="hidden")
        wm_btn.click(watermark_apply_ui, _wm_inputs, [wm_out, wm_file, wm_info],
                     show_progress_on=[wm_out])
        wm_use.click(lambda im: im, wm_out, wm_in)
        wm_clear.click(lambda: (None, None, None, None, None), None,
                       [wm_in, wm_preview, wm_out, wm_file, wm_info])

        # ---- Design templates wiring ----
        _dz_pick_out = ([dz_json, dz_note, dz_canvas, dz_primary, dz_secondary,
                         dz_accent, dz_ink] + dz_texts + [dz_photo, dz_logo])
        _dz_inputs = ([dz_json, dz_photo, dz_logo, dz_cutout_state, dz_canvas,
                       dz_primary, dz_secondary, dz_accent, dz_ink] + dz_texts)
        _dz_live = [dz_canvas, dz_primary, dz_secondary, dz_accent, dz_ink] + dz_texts

        def _dz_after_pick(chain):
            """A new template means a new cut-out (or none) and a new preview."""
            return chain.then(
                design_cutout_ui, [dz_photo, dz_json], [dz_cutout_state, dz_cut_note],
            ).then(design_preview_ui, _dz_inputs, [dz_preview, dz_desc],
                   show_progress="hidden")

        _dz_after_pick(dz_template.input(design_pick_ui, dz_template, _dz_pick_out,
                                         show_progress="hidden"))
        _dz_after_pick(dz_load.click(design_json_ui, dz_json, _dz_pick_out,
                                     show_progress="hidden"))
        dz_palette.input(
            lambda name: tuple(getattr(dz_tools.palette(name), r) for r in dz_tools.ROLES),
            dz_palette, [dz_primary, dz_secondary, dz_accent, dz_ink],
            show_progress="hidden",
        ).then(design_preview_ui, _dz_inputs, [dz_preview, dz_desc],
               show_progress="hidden")
        dz_photo.change(design_cutout_ui, [dz_photo, dz_json],
                        [dz_cutout_state, dz_cut_note]) \
            .then(design_preview_ui, _dz_inputs, [dz_preview, dz_desc],
                  show_progress="hidden")
        for _c in [dz_logo] + _dz_live:
            _c.change(design_preview_ui, _dz_inputs, [dz_preview, dz_desc],
                      show_progress="hidden", trigger_mode="always_last")
        dz_btn.click(design_apply_ui, _dz_inputs, [dz_out, dz_file, dz_info],
                     show_progress_on=[dz_out])
        dz_clear.click(lambda: (None, None, None, None, None), None,
                       [dz_photo, dz_preview, dz_out, dz_file, dz_info])
        # The controls above already hold the first template, so the only thing
        # left to do on open is draw it — with its own placeholder copy, so the
        # tab shows a finished design rather than an empty box.
        demo.load(design_preview_ui, _dz_inputs, [dz_preview, dz_desc],
                  show_progress="hidden")

        # ---- Screenshot wiring ----
        _shot_controls = [shot_bg, shot_color, shot_color2, shot_angle, shot_padding, shot_radius,
                        shot_rim, shot_shadow, shot_soft, shot_chrome, shot_title, shot_tilt,
                        shot_pitch, shot_spin, shot_aspect, shot_custom,
                        shot_size]              # order must match _SHOT_FIELDS
        _shot_inputs = [shot_in] + _shot_controls
        _shot_vis_out = [shot_color, shot_color2, shot_angle, shot_title, shot_custom]
        for _c in (shot_bg, shot_chrome, shot_aspect):
            _c.change(_shot_vis, [shot_bg, shot_chrome, shot_aspect], _shot_vis_out,
                      show_progress="hidden")
        shot_in.change(screenshot_preview_ui, _shot_inputs, [shot_preview, shot_note],
                     show_progress="hidden")
        for _c in _shot_controls:
            _c.change(screenshot_preview_ui, _shot_inputs, [shot_preview, shot_note],
                      show_progress="hidden", trigger_mode="always_last")
        shot_preset.input(screenshot_preset_ui, shot_preset, _shot_controls,
                        show_progress="hidden") \
            .then(_shot_vis, [shot_bg, shot_chrome, shot_aspect], _shot_vis_out,
                  show_progress="hidden") \
            .then(screenshot_preview_ui, _shot_inputs, [shot_preview, shot_note],
                  show_progress="hidden")
        shot_btn.click(screenshot_apply_ui, _shot_inputs, [shot_out, shot_file, shot_info],
                     show_progress_on=[shot_out])
        shot_use.click(lambda im: im, shot_out, shot_in)
        shot_clear.click(lambda: (None, None, None, None, None), None,
                       [shot_in, shot_preview, shot_out, shot_file, shot_info])

        # ---- Crop & frame wiring ----
        _fr_controls = [fr_rotate, fr_fliph, fr_flipv, fr_kh, fr_kv, fr_straighten,
                        fr_aspect, fr_custom, fr_mode, fr_px, fr_py, fr_zoom,
                        fr_size, fr_border, fr_style, fr_color, fr_blur,
                        fr_radius, fr_shadow]      # order must match _FRAME_FIELDS
        _fr_inputs = [fr_in] + _fr_controls
        _fr_vis_out = [fr_custom, fr_mode, fr_color, fr_blur]
        for _c in (fr_aspect, fr_border, fr_style):
            _c.change(_frame_vis, [fr_aspect, fr_border, fr_style], _fr_vis_out,
                      show_progress="hidden")
        fr_in.change(frame_preview_ui, _fr_inputs, [fr_preview, fr_note],
                     show_progress="hidden")
        for _c in _fr_controls:
            _c.change(frame_preview_ui, _fr_inputs, [fr_preview, fr_note],
                      show_progress="hidden", trigger_mode="always_last")
        fr_preset.input(frame_preset_ui, fr_preset, _fr_controls, show_progress="hidden") \
            .then(_frame_vis, [fr_aspect, fr_border, fr_style], _fr_vis_out,
                  show_progress="hidden") \
            .then(frame_preview_ui, _fr_inputs, [fr_preview, fr_note], show_progress="hidden")
        fr_btn.click(frame_apply_ui, _fr_inputs, [fr_out, fr_file, fr_info],
                     show_progress_on=[fr_out])
        fr_use.click(lambda im: im, fr_out, fr_in)
        fr_clear.click(lambda: (None, None, None, None, None), None,
                       [fr_in, fr_preview, fr_out, fr_file, fr_info])

        # ---- Sharpen wiring ----
        _sh_controls = [sh_kind, sh_amount, sh_radius, sh_threshold, sh_halo, sh_lum,
                        sh_shadows, sh_highlights, sh_edge, sh_balance]
        _sh_region = [sh_shape, sh_x, sh_y, sh_w, sh_h, sh_mangle, sh_round, sh_feather,
                      sh_outside, sh_facepad, sh_faces_state, sh_editor]
        _sh_inputs = [sh_in] + _sh_controls + _sh_region + [sh_cx, sh_cy]
        _sh_preview_out = [sh_preview, sh_crop_note]
        sh_kind.change(_sharpen_kind_vis, sh_kind, [sh_edge, sh_balance],
                       show_progress="hidden")
        sh_shape.change(
            _region_vis, sh_shape,
            [sh_x, sh_y, sh_w, sh_h, sh_mangle, sh_round, sh_feather, sh_outside,
             sh_facepad, sh_editor],
            show_progress="hidden",
        ).then(detect_faces_ui, [sh_in, sh_shape], [sh_faces_state, sh_faces_note],
               show_progress="hidden") \
         .then(sharpen_preview_ui, _sh_inputs, _sh_preview_out, show_progress="hidden")
        sh_in.change(blur_on_image, sh_in, sh_editor, show_progress="hidden")
        sh_in.change(detect_faces_ui, [sh_in, sh_shape], [sh_faces_state, sh_faces_note],
                     show_progress="hidden") \
             .then(sharpen_preview_ui, _sh_inputs, _sh_preview_out, show_progress="hidden")
        for _c in _sh_controls + _sh_region[1:] + [sh_cx, sh_cy]:
            if isinstance(_c, gr.State):
                continue
            (_c.change if _c is sh_editor else _c.input)(
                sharpen_preview_ui, _sh_inputs, _sh_preview_out,
                show_progress="hidden", trigger_mode="always_last",
            )
        sh_preset.input(sharpen_preset_ui, sh_preset, _sh_controls, show_progress="hidden") \
            .then(_sharpen_kind_vis, sh_kind, [sh_edge, sh_balance], show_progress="hidden") \
            .then(sharpen_preview_ui, _sh_inputs, _sh_preview_out, show_progress="hidden")
        sh_btn.click(
            sharpen_apply_ui, _sh_inputs, [sh_out, sh_file, sh_info], show_progress_on=[sh_out],
        )
        sh_use.click(lambda pair: (pair[1] if pair else None), sh_out, sh_in)
        sh_clear.click(lambda: (None, None, None, None, None), None,
                       [sh_in, sh_preview, sh_out, sh_file, sh_info])

        # ---- Steam showcase wiring ----
        # Order after the media file must match _steam_params.
        _st_preview_inputs = [st_media, st_fit, st_zoom, st_offx, st_offy, st_bg,
                              st_transparent, st_tilew, st_tileh, st_repeat]
        # Controls a preset writes. Editing one by hand flips the preset to
        # Custom so it stops overriding you; changing the tile width or the
        # media re-applies the active preset (its height depends on both).
        _st_preset_outputs = [st_fit, st_zoom, st_offx, st_offy, st_transparent,
                              st_tileh, st_repeat]
        _st_preset_inputs = [st_media, st_preset] + _st_preview_inputs[1:]
        st_media.change(
            steam_apply_preset, _st_preset_inputs, _st_preset_outputs, show_progress="hidden",
        ).then(steam_preview_ui, _st_preview_inputs, st_preview, show_progress="hidden")
        st_preset.input(
            steam_apply_preset, _st_preset_inputs, _st_preset_outputs, show_progress="hidden",
        ).then(steam_preview_ui, _st_preview_inputs, st_preview, show_progress="hidden")
        st_tilew.input(
            steam_apply_preset, _st_preset_inputs, _st_preset_outputs, show_progress="hidden",
        ).then(steam_preview_ui, _st_preview_inputs, st_preview, show_progress="hidden",
               trigger_mode="always_last")
        for _c in _st_preview_inputs[1:]:
            if _c is st_tilew:
                continue
            _c.input(
                steam_preview_ui, _st_preview_inputs, st_preview,
                show_progress="hidden", trigger_mode="always_last",
            )
        for _c in _st_preset_outputs:
            _c.input(lambda: gr.update(value=steam.PRESET_CUSTOM), None, st_preset,
                     show_progress="hidden")
        st_media.change(
            steam_on_media, st_media, [st_end, st_anim_group, st_fmt, st_info]
        )
        st_evt = st_export.click(
            steam_export_ui,
            _st_preview_inputs + [st_fmt, st_fps, st_start, st_end, st_loopmode,
                                  st_maxmb, st_hexify, st_outdir],
            [st_gallery, st_file, st_info],
            show_progress_on=[st_gallery],
        )
        st_cancel.click(lambda: _STEAM_CANCEL.set(), None, None, cancels=[st_evt])
        st_clear.click(
            lambda: (None, None, None, None), None,
            [st_media, st_preview, st_gallery, st_file],
        )

        # ---- Library tab ----
        _lib_outputs = [lib_gallery, lib_video_pick, lib_video, lib_count]
        lib_tab.select(refresh_library, None, _lib_outputs)   # load on open
        lib_refresh.click(refresh_library, None, _lib_outputs)
        lib_video_pick.change(lambda v: v, lib_video_pick, lib_video,
                              show_progress="hidden")
        lib_open.click(open_library_folder, None, None)

        # ---- Settings page (opened by the ⚙ gear, closed by Back) ----
        settings_btn.click(
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            None, [main_tabs, settings_view],
        )
        set_back.click(
            lambda: (gr.update(visible=True), gr.update(visible=False)),
            None, [main_tabs, settings_view],
        )
        set_save.click(
            save_settings, [set_device, set_model, set_outdir],
            [set_status, model, device, batch_model, batch_device, vid_device,
             pn_outdir],
        )
        set_open_lib.click(open_library_folder, None, None)
        mm_dl.click(_mm_download, [mm_pick], [mm_table, mm_total, mm_status])
        mm_rm.click(_mm_remove, [mm_pick], [mm_table, mm_total, mm_status])
        mm_refresh.click(_mm_refresh, None, [mm_table, mm_total])
        diag_refresh.click(lambda: manage.system_report(), None, diag)
    return demo


def main(inbrowser: bool = False) -> None:
    build_demo().launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("UPSCALER_PORT", "7860")),
        theme=THEME,
        css=_CSS,
        js=_APPLY_THEME_JS,
        head=_MAGNIFIER_HEAD,
        inbrowser=inbrowser,
        # The Library reads from ~/.upscaler/library, outside the app dir — Gradio
        # won't serve files from there unless the folder is explicitly allowed.
        allowed_paths=[str(library.ensure_dir())],
    )


def gui() -> None:
    """`upscaler-gui` console script: launch and open the browser."""
    main(inbrowser=True)


if __name__ == "__main__":
    # `python app.py` (dev / LaunchAgent) keeps the old behavior: no browser
    # pop-up, so the login-time autostart stays silent.
    main()
