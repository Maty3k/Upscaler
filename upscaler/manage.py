"""Model-file management + a telemetry-free diagnostics report.

Powers the Settings page's "Models & downloads" and "Diagnostics" panels. Kept
free of any *eager* heavy imports (torch/gradio/onnxruntime) — it only reads the
registry and the weights helpers — so it stays cheap and easily testable. The
one device probe in :func:`system_report` imports the engine lazily.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

from upscaler import background
from upscaler.models import weights as W
from upscaler.models.registry import (
    ARTIFACT_MODELS,
    COLORIZE_MODELS,
    DEBLUR_MODELS,
    FACE_DETECTOR,
    FACE_MODELS,
    INPAINT_MODELS,
    MODELS,
)

_WEIGHT_EXTS = {".pth", ".onnx", ".pt", ".ckpt", ".safetensors"}


@dataclass(frozen=True)
class ManagedSpec:
    group: str
    name: str
    filename: str
    present: bool
    size_bytes: int
    url: str
    notes: str
    spec: object


def _grouped_specs():
    """(group, spec) for every weight the app knows how to download."""
    yield from (("Upscale", s) for s in MODELS.values())
    yield from (("Clean-up", s) for s in DEBLUR_MODELS.values())
    yield from (("Clean-up", s) for s in ARTIFACT_MODELS.values())
    yield from (("Colorize", s) for s in COLORIZE_MODELS.values())
    yield from (("Inpaint", s) for s in INPAINT_MODELS.values())
    yield from (("Faces", s) for s in FACE_MODELS.values())
    yield ("Faces", FACE_DETECTOR)
    yield from (("Background", s) for s in background.BG_MODELS.values())


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def list_specs() -> list[ManagedSpec]:
    out = []
    for group, spec in _grouped_specs():
        dest = W.WEIGHTS_DIR / spec.filename
        present = dest.is_file()
        out.append(ManagedSpec(
            group=group, name=spec.name, filename=spec.filename,
            present=present, size_bytes=_size(dest) if present else 0,
            url=spec.url, notes=spec.notes, spec=spec,
        ))
    return out


def _spec_by_filename() -> dict:
    return {spec.filename: spec for _g, spec in _grouped_specs()}


def total_bytes() -> int:
    """Total size of all weight files actually on disk (incl. orphans)."""
    if not W.WEIGHTS_DIR.is_dir():
        return 0
    total = 0
    for p in W.WEIGHTS_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in _WEIGHT_EXTS:
            total += _size(p)
    return total


def human_size(n: int) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n)} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def download_one(filename: str) -> str:
    """Pre-download a registered weight. Returns a friendly status string."""
    spec = _spec_by_filename().get(filename)
    if spec is None:
        return f"⚠ Unknown file {filename!r}."
    try:
        dest = W.ensure_weights(spec)
    except RuntimeError as e:  # network / checksum failure already friendly
        return f"⚠ {e}"
    return f"✅ Ready: {filename} ({human_size(_size(Path(dest)))})."


def remove_one(filename: str) -> str:
    """Delete a registered weight file from WEIGHTS_DIR. Refuses anything that
    isn't a known filename or escapes the weights dir."""
    if filename not in _spec_by_filename():
        return f"⚠ Refused: {filename!r} is not a known model file."
    dest = (W.WEIGHTS_DIR / filename).resolve()
    root = W.WEIGHTS_DIR.resolve()
    if not dest.is_relative_to(root):
        return "⚠ Refused: path is outside the weights folder."
    if not dest.exists():
        return f"{filename} isn't downloaded — nothing to remove."
    dest.unlink(missing_ok=True)
    return f"🗑 Removed {filename}."


# -- Diagnostics report ------------------------------------------------------

_OPTIONAL_DEPS = [
    "torch", "torchvision", "gradio", "onnxruntime", "onnx", "spandrel",
    "spandrel_extra_arches", "cv2", "pypdfium2", "pillow_heif", "imageio_ffmpeg",
    "numpy", "PIL",
]


def _has(mod: str) -> str:
    try:
        return "present" if importlib.util.find_spec(mod) else "missing"
    except (ImportError, ValueError, ModuleNotFoundError):
        return "missing"


def system_report() -> str:
    """A plain-text, copy-pasteable local system report. Never raises, makes no
    network calls, uploads nothing."""
    lines: list[str] = []

    try:
        from upscaler import __version__ as ver
    except Exception:
        ver = "unknown"
    lines.append(f"Upscaler: {ver}")
    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"Python: {platform.python_version()} ({platform.machine()})")

    lines.append("")
    lines.append("Optional packages:")
    for mod in _OPTIONAL_DEPS:
        lines.append(f"  {mod:22s} {_has(mod)}")

    lines.append("")
    lines.append(f"ffmpeg:  {shutil.which('ffmpeg') or 'not found'}")
    lines.append(f"ffprobe: {shutil.which('ffprobe') or 'not found'}")

    lines.append("")
    try:
        import torch  # noqa: F401
        from upscaler.engine import resolve_device
        lines.append(f"torch: {torch.__version__}")
        lines.append(f"  CUDA available: {torch.cuda.is_available()}")
        mps = getattr(torch.backends, "mps", None)
        lines.append(f"  MPS available:  {bool(mps and mps.is_available())}")
        lines.append(f"  resolved device: {resolve_device('auto').type}")
    except Exception as e:
        lines.append(f"torch: missing — running report without device probe ({e})")

    lines.append("")
    wdir = W.WEIGHTS_DIR
    lines.append(f"Weights dir: {wdir} (exists: {wdir.is_dir()})")
    lines.append(f"  weights on disk: {human_size(total_bytes())}")
    try:
        from upscaler import config, library
        lines.append(f"Library dir: {library.LIBRARY_DIR}")
        lines.append(f"Config file: {config.CONFIG_PATH}")
    except Exception:
        pass
    try:
        probe = wdir if wdir.exists() else (wdir.parent if wdir.parent.exists() else Path.home())
        lines.append(f"Free disk: {human_size(shutil.disk_usage(probe).free)}")
    except Exception:
        pass

    lines.append("")
    lines.append("Env overrides:")
    for var in ("UPSCALER_WEIGHTS_DIR", "UPSCALER_LIBRARY", "UPSCALER_CONFIG", "UPSCALER_PORT"):
        lines.append(f"  {var:22s} {os.environ.get(var, '(default)')}")

    return "\n".join(lines)
