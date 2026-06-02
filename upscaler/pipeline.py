"""High-level enhance() pipeline: optional deblur, then upscale, then sharpen."""

from __future__ import annotations

from typing import Optional

from PIL import Image

from upscaler.deblur import Deblurrer
from upscaler.engine import Upscaler
from upscaler.sharpen import unsharp_mask


def enhance(
    image: Image.Image,
    *,
    model: Optional[str] = None,
    scale: Optional[int] = 4,
    device: str = "auto",
    tile: int = 512,
    deblur: bool = False,
    deblur_model: Optional[str] = None,
    sharpen: float = 0.0,
    upscaler: Optional[Upscaler] = None,
    deblurrer: Optional[Deblurrer] = None,
) -> Image.Image:
    """Enhance ``image``: optional NAFNet deblur → upscale → optional unsharp.

    Deblur runs first, at native resolution, so motion blur isn't amplified by
    the upscaler. Pass prebuilt ``upscaler``/``deblurrer`` to reuse loaded models
    across many images. ``sharpen`` is the unsharp strength (0 disables it).
    """
    result = image
    if deblur:
        db = deblurrer or Deblurrer(model=deblur_model, device=device)
        result = db.deblur(result)

    up = upscaler or Upscaler(model=model, scale=scale, device=device, tile=tile)
    result = up.upscale(result)

    if sharpen > 0:
        result = unsharp_mask(result, strength=sharpen)
    return result
