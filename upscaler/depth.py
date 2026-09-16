"""Monocular depth estimation — how far away each pixel is, from one photo.

Runs Depth Anything V2 Small on onnxruntime, the same way background removal
runs U²-Net: weights fetched lazily, checksum pinned, no torch involved. The
Small model is Apache-2.0, unlike the Base and Large models of the same family,
which is why it is the one here.

The network returns **inverse depth**: a larger number means nearer, and the
scale is relative to the picture rather than in metres. That is exactly what a
depth-of-field effect wants — it only needs to know what is in front of what —
so the map is normalised to 0 (farthest) … 255 (nearest) and handed back as an
``L`` image at the photo's own size.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from upscaler.models.registry import DEFAULT_DEPTH_MODEL, DEPTH_MODELS
from upscaler.models.weights import ensure_weights

# The DPT preprocessing the model was exported with: ImageNet normalisation at
# 518×518. Straight from the published preprocessor_config.json.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ONNX_HINT = ('depth estimation needs onnxruntime. Install it with: '
             'pip install -e ".[onnx]"')

_sessions: "dict[str, object]" = {}


def _session(model: str):
    """A cached onnxruntime session for ``model``."""
    if model not in DEPTH_MODELS:
        raise ValueError(f"Unknown depth model {model!r}. "
                         f"Available: {', '.join(DEPTH_MODELS)}")
    if model in _sessions:
        return _sessions[model]
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(ONNX_HINT) from e

    path = str(ensure_weights(DEPTH_MODELS[model]))
    options = ort.SessionOptions()
    # The fp16 export of this model trips a graph-fusion bug in onnxruntime, and
    # while the two weights shipped here are not affected, a future one might
    # be; failing over to unoptimised beats failing to load at all.
    try:
        session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    except Exception:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    _sessions[model] = session
    return session


def estimate(image: Image.Image, model: str = DEFAULT_DEPTH_MODEL) -> Image.Image:
    """A depth map for ``image``, as an ``L`` image at the same size.

    255 is the nearest thing in the frame and 0 the farthest. The values are
    relative to this picture only — two photos' maps are not comparable, and
    none of it is in metres.
    """
    session = _session(model)
    spec = DEPTH_MODELS[model]
    rgb = image.convert("RGB")

    small = rgb.resize((spec.size, spec.size), Image.BICUBIC)
    arr = (np.asarray(small, dtype=np.float32) / 255.0 - _MEAN) / _STD
    inp = session.get_inputs()[0]
    dtype = np.float16 if "float16" in inp.type else np.float32
    batch = arr.transpose(2, 0, 1)[None].astype(dtype)

    raw = np.asarray(session.run(None, {inp.name: batch})[0], dtype=np.float32).squeeze()
    lo, hi = float(raw.min()), float(raw.max())
    norm = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
    return Image.fromarray((norm * 255.0).round().astype(np.uint8), "L").resize(
        rgb.size, Image.BICUBIC)


def focus_at(depth_map: Image.Image, x_pct: float, y_pct: float,
             radius: float = 2.0) -> float:
    """The focus setting (0..100) that puts the point at ``x_pct``, ``y_pct``
    in focus — what a camera does when you tap the screen.

    Reading a single pixel would be at the mercy of one noisy value, so a small
    neighbourhood is averaged. ``radius`` is a percentage of the short side.
    """
    arr = np.asarray(depth_map.convert("L"), dtype=np.float32)
    h, w = arr.shape
    cx = int(round(np.clip(x_pct, 0.0, 100.0) / 100.0 * (w - 1)))
    cy = int(round(np.clip(y_pct, 0.0, 100.0) / 100.0 * (h - 1)))
    r = max(1, int(round(max(0.0, radius) / 100.0 * min(w, h))))
    patch = arr[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
    if patch.size == 0:
        patch = arr[cy:cy + 1, cx:cx + 1]
    return round(float(np.median(patch)) / 255.0 * 100.0, 1)


def available() -> bool:
    """True when onnxruntime is installed, so callers can offer depth or not."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True
