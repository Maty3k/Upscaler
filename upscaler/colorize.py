"""Optional DDColor colorization (the Colorize tab).

DDColor predicts the AB chroma channels and merges them with the *source*
luminance, so a black-and-white or faded photo gets plausible color while
keeping all of its original detail. It loads through the shared spandrel loader
(behind the ``[face]`` extra) and downloads its weights lazily on first use.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from upscaler.engine import load_spandrel, resolve_device
from upscaler.models.registry import COLORIZE_MODELS, DEFAULT_COLORIZE_MODEL
from upscaler.models.weights import ensure_weights


class Colorizer:
    """Colorize a grayscale/faded image with DDColor. Lazy: the ~870MB net only
    downloads/loads the first time :meth:`colorize` runs."""

    def __init__(self, model: str = DEFAULT_COLORIZE_MODEL, device: str = "auto"):
        if model not in COLORIZE_MODELS:
            raise ValueError(
                f"Unknown colorize model {model!r}. "
                f"Available: {', '.join(COLORIZE_MODELS)}"
            )
        self._spec = COLORIZE_MODELS[model]
        self._device = resolve_device(device)
        self._net = None

    def _load(self):
        if self._net is None:
            # DDColor is 1->3 channels, so skip load_spandrel's RGB channel guard.
            lm = load_spandrel(
                ensure_weights(self._spec), self._device, require_channels=None
            )
            self._net = lm.net
        return self._net

    @torch.inference_mode()
    def colorize(self, image: Image.Image, strength: float = 1.0) -> Image.Image:
        """Return a colorized RGB image the same size as ``image``.

        ``strength`` in [0, 1] dials saturation: 1 = full color, 0 = the source
        as grayscale-RGB (the original luminance is always preserved)."""
        net = self._load()
        gray = image.convert("L")
        x = torch.from_numpy(np.asarray(gray, dtype=np.float32) / 255.0)
        x = x[None, None].to(self._device)  # (1, 1, H, W) — DDColor builds RGB
        out = net(x).clamp(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        colored = Image.fromarray(np.round(out * 255.0).astype(np.uint8), "RGB")

        strength = max(0.0, min(1.0, float(strength)))
        if strength >= 0.999:
            return colored
        return Image.blend(gray.convert("RGB"), colored, strength)
