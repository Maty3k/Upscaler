"""Model-based deblur stage using NAFNet (GoPro motion-deblur weights).

NAFNet's channel attention pools globally, so the network is run on the whole
image (it pads internally to a valid size) rather than tiled — tiling would
introduce seams. Deblur is meant to run at the image's native resolution,
*before* upscaling, so memory is rarely a concern.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from upscaler.engine import _load_state_dict, resolve_device
from upscaler.models.nafnet import NAFNet
from upscaler.models.registry import DeblurSpec, resolve_deblur_model
from upscaler.models.weights import ensure_weights


class Deblurrer:
    """Wraps a pretrained NAFNet deblur model for whole-image inference.

    Example
    -------
    >>> db = Deblurrer()
    >>> sharp = db.deblur(Image.open("blurry.jpg"))
    """

    def __init__(self, model: Optional[str] = None, device: str = "auto"):
        self.spec: DeblurSpec = resolve_deblur_model(model)
        self.device = resolve_device(device)

        net = NAFNet(
            img_channel=3,
            width=self.spec.width,
            middle_blk_num=self.spec.middle_blk_num,
            enc_blk_nums=self.spec.enc_blk_nums,
            dec_blk_nums=self.spec.dec_blk_nums,
        )
        net.load_state_dict(_load_state_dict(ensure_weights(self.spec)), strict=True)
        net.eval()
        net.to(self.device)
        self.net = net

    @torch.inference_mode()
    def deblur(self, image: Image.Image) -> Image.Image:
        """Deblur a PIL image, returning a same-resolution result."""
        rgb = image.convert("RGB")
        x = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = self.net(x).clamp_(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        return Image.fromarray(np.round(out * 255.0).astype(np.uint8), mode="RGB")
