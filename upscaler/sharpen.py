"""Sharpening stages.

Phase 1 ships the classical *unsharp mask* — zero extra weights, instant, and
good for mild softness. A model-based deblur stage (e.g. NAFNet) is planned for
Phase 2 for genuinely blurry input; it will slot in alongside this with the same
(image, strength) -> image signature.
"""

from __future__ import annotations

from PIL import Image, ImageFilter


def unsharp_mask(
    image: Image.Image,
    strength: float = 1.0,
    radius: float = 2.0,
    threshold: int = 3,
) -> Image.Image:
    """Sharpen via unsharp masking.

    ``strength`` is a convenience multiplier on the standard ``percent`` amount
    (1.0 -> ~150%). ``radius`` controls the scale of detail enhanced; ``threshold``
    avoids amplifying flat-area noise.
    """
    if strength <= 0:
        return image
    percent = int(150 * strength)
    return image.convert("RGB").filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )
