"""Background removal via U²-Net, producing a transparent-PNG cutout.

Runs on onnxruntime (already a dependency for the ONNX backend) with weights
lazily downloaded and cached like every other model here — no heavy extra
dependency tree. The result is an RGBA image whose alpha is the foreground mask,
ready to drop onto the Lian Li panel as a sticker.

Models are the standard U²-Net ONNX exports (Apache-2.0), mirrored alongside the
official rembg releases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from upscaler.models.weights import ensure_weights


@dataclass(frozen=True)
class BGSpec:
    name: str
    url: str
    filename: str
    sha256: Optional[str] = None
    size: int = 320  # network input is size×size
    notes: str = ""


_REMBG = "https://github.com/danielgatis/rembg/releases/download/v0.0.0"

BG_MODELS: dict[str, BGSpec] = {
    "u2net": BGSpec(
        name="u2net",
        url=f"{_REMBG}/u2net.onnx",
        filename="u2net.onnx",
        size=320,
        notes="General foreground/subject cut-out. Best all-rounder (~176MB).",
    ),
    "u2netp": BGSpec(
        name="u2netp",
        url=f"{_REMBG}/u2netp.onnx",
        filename="u2netp.onnx",
        size=320,
        notes="Lighter/faster U²-Net (~4MB) — lower fidelity edges.",
    ),
}

DEFAULT_BG_MODEL = "u2net"

# ImageNet normalisation used by U²-Net's published pre-processing.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_sessions: dict[str, object] = {}


def _session(spec: BGSpec):
    sess = _sessions.get(spec.name)
    if sess is None:
        try:
            import onnxruntime as ort  # local import: only when removing a background
        except ImportError as e:
            raise RuntimeError(
                "Background removal needs onnxruntime. Install it with: "
                'pip install -e ".[onnx]" (or ".[gui]")'
            ) from e

        path = ensure_weights(spec)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        _sessions[spec.name] = sess
    return sess


def _mask(img: Image.Image, spec: BGSpec) -> Image.Image:
    """Predict the foreground alpha mask (L), upsampled back to `img` size."""
    sess = _session(spec)
    inp_name = sess.get_inputs()[0].name

    small = img.convert("RGB").resize((spec.size, spec.size), Image.LANCZOS)
    arr = np.array(small, dtype=np.float32)
    arr = arr / (arr.max() or 1.0)
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)[None]  # 1×3×H×W

    pred = sess.run(None, {inp_name: arr.astype(np.float32)})[0]
    pred = np.squeeze(pred)  # H×W
    lo, hi = float(pred.min()), float(pred.max())
    pred = (pred - lo) / (hi - lo + 1e-8)
    mask = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
    return mask.resize(img.size, Image.LANCZOS)


def remove_background(
    img: Image.Image,
    model: str = DEFAULT_BG_MODEL,
    feather: int = 0,
) -> Image.Image:
    """Return `img` as RGBA with the background made transparent.

    `feather` (px) softens the mask edge with a slight blur — nice for stickers.
    """
    spec = BG_MODELS.get(model)
    if spec is None:
        raise ValueError(
            f"Unknown background model {model!r}. Available: {', '.join(BG_MODELS)}"
        )
    mask = _mask(img, spec)
    if feather and feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(float(feather)))
    out = img.convert("RGBA")
    out.putalpha(mask)
    return out


def on_checkerboard(rgba: Image.Image, square: int = 16) -> Image.Image:
    """Composite an RGBA cutout over a checkerboard so transparency is visible
    in the (opaque) preview pane."""
    w, h = rgba.size
    yy, xx = np.mgrid[0:h, 0:w]
    checker = ((xx // square + yy // square) % 2).astype(np.uint8)
    board = np.where(checker[..., None] == 0, 205, 255).astype(np.uint8)
    board = np.repeat(board, 3, axis=2)
    bg = Image.fromarray(board, "RGB")
    bg.paste(rgba, (0, 0), rgba)
    return bg
