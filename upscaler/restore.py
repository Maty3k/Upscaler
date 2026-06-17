"""Optional FBCNN JPEG-artifact removal, used as a pre-upscale clean-up stage.

Heavy JPEG compression bakes in blocking and ringing — the exact thing a sharp
upscaler would otherwise enlarge and emphasise. FBCNN ("Flexible Blind CNN")
removes those artifacts. It loads through the shared spandrel loader (the arch
ships in core spandrel), behind the same optional ``[face]`` extra as the other
spandrel models, and is imported lazily with a clear message.

Like the NAFNet :class:`~upscaler.deblur.Deblurrer`, this runs whole-image (no
tiling): FBCNN works at the original resolution, before any enlargement.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from upscaler.engine import load_spandrel, resolve_device
from upscaler.models.registry import ARTIFACT_MODELS, DEFAULT_ARTIFACT_MODEL
from upscaler.models.weights import ensure_weights


class ArtifactRemover:
    """Whole-image JPEG-artifact removal (FBCNN). Lazy: the ~288MB net only
    downloads/loads the first time :meth:`restore` is called."""

    def __init__(self, model: str = DEFAULT_ARTIFACT_MODEL, device: str = "auto"):
        if model not in ARTIFACT_MODELS:
            raise ValueError(
                f"Unknown artifact model {model!r}. "
                f"Available: {', '.join(ARTIFACT_MODELS)}"
            )
        self._spec = ARTIFACT_MODELS[model]
        self._device = resolve_device(device)
        self._net = None

    def _load(self):
        if self._net is None:
            lm = load_spandrel(
                ensure_weights(self._spec), self._device, require_channels=3
            )
            self._net = lm.net
        return self._net

    @torch.inference_mode()
    def restore(self, image: Image.Image) -> Image.Image:
        """Return ``image`` with JPEG artifacts removed (same resolution)."""
        net = self._load()
        rgb = image.convert("RGB")
        x = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self._device)
        # spandrel's descriptor already clamps to [0, 1].
        y = net(x).clamp(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        return Image.fromarray(np.round(y * 255.0).astype(np.uint8), "RGB")
