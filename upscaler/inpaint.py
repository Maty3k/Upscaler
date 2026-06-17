"""Optional LaMa object removal / inpainting (the Inpaint tab).

The user paints a mask over the object to remove; LaMa fills the masked region
from the surrounding context. Big-LaMa ships as a self-contained TorchScript
model called as ``model(image, mask)``, so this needs only the core ``torch``
dependency (no extra packages) and downloads the weights lazily on first use.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from upscaler.engine import resolve_device
from upscaler.models.registry import DEFAULT_INPAINT_MODEL, INPAINT_MODELS
from upscaler.models.weights import ensure_weights

_MOD = 8  # LaMa needs spatial dims as multiples of 8


class Inpainter:
    """Remove objects with Big-LaMa. Lazy: the ~196MB model only downloads/loads
    the first time :meth:`inpaint` runs."""

    def __init__(self, model: str = DEFAULT_INPAINT_MODEL, device: str = "auto"):
        if model not in INPAINT_MODELS:
            raise ValueError(
                f"Unknown inpaint model {model!r}. "
                f"Available: {', '.join(INPAINT_MODELS)}"
            )
        self._spec = INPAINT_MODELS[model]
        self._device = resolve_device(device)
        self._net = None

    def _load(self):
        if self._net is None:
            self._net = torch.jit.load(
                str(ensure_weights(self._spec)), map_location=self._device
            ).eval()
        return self._net

    @staticmethod
    def _pad(t: torch.Tensor) -> torch.Tensor:
        _, _, h, w = t.shape
        ph, pw = (-h) % _MOD, (-w) % _MOD
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), mode="replicate")
        return t

    @torch.inference_mode()
    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Fill the painted region of ``image``. ``mask`` is any image where
        non-black pixels mark what to remove; it's resized to match ``image``."""
        net = self._load()
        rgb = image.convert("RGB")
        w, h = rgb.size
        x = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1)[None].to(self._device)

        m = np.asarray(mask.convert("L").resize((w, h)), dtype=np.float32)
        m = (m > 127).astype(np.float32)  # binarize: painted = inpaint
        mt = torch.from_numpy(m)[None, None].to(self._device)

        out = net(self._pad(x), self._pad(mt)).clamp(0, 1)[:, :, :h, :w]
        arr = out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        return Image.fromarray(np.round(arr * 255.0).astype(np.uint8), "RGB")
