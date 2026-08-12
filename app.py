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
from PIL import Image

from upscaler import background, config, library, manage, panel
from upscaler.convert import FORMATS, convert, extension_for
from upscaler.document import images_to_pdf, pdf_to_images
from upscaler.deblur import Deblurrer, DeblurTooLargeError
from upscaler.engine import (CancelledError, OutputTooLargeError, Upscaler,
                             resolve_device)
from upscaler import fit
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
            custom_size="", crop_anchor="center", progress=gr.Progress()):
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
        cropped = fit.crop(src, *exact, anchor=crop_anchor or "center")
        if cropped.size != src.size:
            stages.append(f"crop to {exact[0]}:{exact[1]} ({crop_anchor or 'center'})")
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
        stages.append(f"→ {exact[0]}×{exact[1]}")
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
            stages.append(f"→ {target}px")

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


_CONVERT_METHODS = ["Change image format", "Images → PDF", "PDF → Images"]

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


def _switch_method(choice):
    """Show only the group for the selected conversion method."""
    return (
        gr.update(visible=choice == _CONVERT_METHODS[0]),
        gr.update(visible=choice == _CONVERT_METHODS[1]),
        gr.update(visible=choice == _CONVERT_METHODS[2]),
    )


# -- Batch processing (one operation over many images) -----------------------

_BATCH_OPS = ["Upscale", "Convert format", "Remove background"]


def _switch_batch_op(op):
    """Show only the settings group for the selected batch operation."""
    return (
        gr.update(visible=op == _BATCH_OPS[0]),
        gr.update(visible=op == _BATCH_OPS[1]),
        gr.update(visible=op == _BATCH_OPS[2]),
    )


def batch_process(files, op, model, out_size, sharpen, fmt, quality,
                  bg_model, feather, device, tile, progress=gr.Progress()):
    """Run one operation over many uploaded images. Returns (gallery, zip, info).

    Resilient: a file that can't be read or fails is skipped and counted, so one
    bad image never sinks the whole batch. Every result is also saved to the
    Library.
    """
    if not files:
        raise gr.Error("Add at least one image to process.")
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
button, .tab-nav button, .drop, .item, .dropdown-arrow {
    transition: background-color .18s ease, border-color .18s ease, color .18s ease; }
/* transform/box-shadow only on the single button being hovered (cheap). */
.gradio-container button.primary {
    transition: background-color .18s ease, transform .12s ease, box-shadow .2s ease; }
.gradio-container button.primary:hover { transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(13,148,136,.28); }
.gradio-container button.primary:active { transform: translateY(0); box-shadow: none; }
.tab-nav button:hover { color: var(--body-text-color) !important; }

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
    with gr.Blocks(title="Upscaler") as demo:
        gr.HTML(
            '<div id="hero">'
            f'<div class="brandrow"><span class="logo">{ICON_LOGO}</span>'
            '<span class="brand">Upscaler</span></div>'
            '<div class="sub">Enlarge, sharpen and clean up your photos with AI — '
            "then convert formats, clean up video, build PDFs or cut out "
            "backgrounds. Every tool runs on your own machine; nothing is ever "
            "uploaded.</div>"
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
                        crop_anchor = gr.Dropdown(
                            list(fit.ANCHORS), value="center", visible=False,
                            label="Keep which part", filterable=False,
                            info="Which part of the photo to keep when the crop has "
                            "to cut something — top is usually right for portraits.",
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
                            type="pil", height=300, buttons=["download", "fullscreen"],
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
                            label="Before / after", type="pil", height=300,
                            elem_classes=["loupe"],
                        )
                        col_info = gr.Markdown()

            # ---- Tab: Remove objects / inpaint (LaMa) ----
            with gr.Tab("Remove Objects"):
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
                            label="Before / after", type="pil", height=360,
                            elem_classes=["loupe"],
                        )
                        ip_info = gr.Markdown()

            # ---- Tab: Remove background ----
            with gr.Tab("Remove BG"):
                gr.HTML(_section_head(
                    "Cut-out", "Remove Background",
                    "Lift the subject cleanly off its background with AI and save a "
                    "transparent PNG. It drops straight into the Lian Li Screen tab "
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
                            type="pil", height=220, buttons=["download", "fullscreen"],
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
                    "or split a PDF back into images. Pick a task below to begin.",
                    icon=ICON_CONVERT,
                ))
                method = gr.Dropdown(
                    _CONVERT_METHODS, value=_CONVERT_METHODS[0],
                    label="What do you want to do?", filterable=False,
                    info="Pick your task: change an image's format, build a PDF from "
                    "images, or split a PDF back into images.",
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

                method.change(
                    _switch_method, method, [grp_format, grp_topdf, grp_frompdf],
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
                            info="What to do to every image you dropped above.",
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
                    [batch_grp_up, batch_grp_conv, batch_grp_bg],
                    show_progress="hidden",  # instant visibility toggle, no flash
                )
                batch_evt = batch_run.click(
                    batch_process,
                    [batch_in, batch_op, batch_model, batch_size, batch_sharpen,
                     batch_fmt, batch_quality, batch_bg_model, batch_feather,
                     batch_device, batch_tile],
                    [batch_gallery, batch_zip, batch_info],
                    show_progress_on=[batch_gallery],
                )
                batch_cancel.click(lambda: _BATCH_CANCEL.set(), None, None,
                                   cancels=[batch_evt])

            # ---- Tab: Lian Li 8.8" Screen builder ----
            with gr.Tab("Lian Li Screen"):
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
        # custom box only for "Custom size…".
        out_size.change(
            lambda choice: (
                gr.update(visible=choice == _EXACT_CUSTOM),
                gr.update(visible=choice == _EXACT_CUSTOM
                          or choice in fit.TARGET_PRESETS),
            ),
            out_size, [custom_size, crop_anchor],
        )
        run_evt = run.click(
            enhance,
            [inp, model, device, deblur, deblur_model, restore_strength, sharpen,
             tile, onnx, out_size, face, face_strength, face_model, face_fidelity,
             fbcnn, custom_size, crop_anchor],
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


if __name__ == "__main__":
    build_demo().launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("UPSCALER_PORT", "7860")),
        theme=THEME,
        css=_CSS,
        js=_APPLY_THEME_JS,
        head=_MAGNIFIER_HEAD,
        # The Library reads from ~/.upscaler/library, outside the app dir — Gradio
        # won't serve files from there unless the folder is explicitly allowed.
        allowed_paths=[str(library.ensure_dir())],
    )
