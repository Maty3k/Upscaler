"""High-level enhance() pipeline: upscale, then optionally sharpen."""

from __future__ import annotations

from typing import Optional

from PIL import Image

from upscaler.engine import Upscaler
from upscaler.sharpen import unsharp_mask


def enhance(
    image: Image.Image,
    *,
    model: Optional[str] = None,
    scale: Optional[int] = 4,
    device: str = "auto",
    tile: int = 512,
    sharpen: float = 0.0,
    upscaler: Optional[Upscaler] = None,
) -> Image.Image:
    """Upscale ``image`` and optionally apply a final unsharp pass.

    Pass a prebuilt ``upscaler`` to reuse a loaded model across many images
    (avoids reloading weights per call). ``sharpen`` is the unsharp strength;
    0 disables it.
    """
    up = upscaler or Upscaler(model=model, scale=scale, device=device, tile=tile)
    result = up.upscale(image)
    if sharpen > 0:
        result = unsharp_mask(result, strength=sharpen)
    return result
