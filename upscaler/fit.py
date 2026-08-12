"""Fit an image to an exact target resolution — ultrawide, phone, tablet.

The upscale models enlarge by a fixed integer factor and preserve aspect ratio,
which never lands on "3440x1440" on its own. This plans the three steps that
do: crop the source to the target's aspect ratio, enlarge past the target, then
resample down to the exact pixel count.

Cropping happens *first*, before the model runs, for two reasons. It's the only
step that removes pixels, and every pixel cropped afterwards would have been
enlarged at full cost first — an ultrawide target from a 4:3 source discards
about 40% of the frame, which on a small machine is the difference between
fitting in memory and paging. And it costs nothing in quality, because the
enlarged result is resampled down to the target anyway.

The tradeoff it does carry: the model sees the crop's new borders as image
edges, so a cropped-first result can differ very slightly at the frame edge
from one cropped afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from PIL import Image

# Where the crop sits when the source is wider/taller than the target ratio.
# "attention" isn't offered: it needs saliency detection this package doesn't
# have, and a wrong guess silently cuts someone's head off.
ANCHORS = ("center", "top", "bottom", "left", "right")

# name -> (width, height). Grouped for the UI; ordering is display order.
TARGET_PRESETS: dict[str, tuple[int, int]] = {
    # Ultrawide and super-ultrawide monitors
    "Ultrawide · 2560×1080": (2560, 1080),
    "Ultrawide QHD · 3440×1440": (3440, 1440),
    "Super ultrawide · 5120×1440": (5120, 1440),
    # Standard monitors
    "Full HD · 1920×1080": (1920, 1080),
    "QHD · 2560×1440": (2560, 1440),
    "4K UHD · 3840×2160": (3840, 2160),
    "5K · 5120×2880": (5120, 2880),
    # Phones (portrait)
    "Phone FHD+ · 1080×2400": (1080, 2400),
    "iPhone 15/16 · 1179×2556": (1179, 2556),
    "iPhone Pro Max · 1290×2796": (1290, 2796),
    # Tablets
    "iPad · 1640×2360": (1640, 2360),
    "iPad Pro 11 · 1668×2388": (1668, 2388),
    "iPad Pro 13 · 2048×2732": (2048, 2732),
}


@dataclass(frozen=True)
class FitPlan:
    """What it takes to land on an exact target."""

    crop_box: tuple[int, int, int, int]  # left, top, right, bottom in the source
    scale: int                           # model factor to run (1 = no enlarge)
    target: tuple[int, int]              # final width, height
    upscaled: tuple[int, int]            # size after the model, before resample

    @property
    def crop_size(self) -> tuple[int, int]:
        left, top, right, bottom = self.crop_box
        return right - left, bottom - top

    @property
    def downsamples(self) -> bool:
        """True when the final resample shrinks — the crisp direction."""
        return self.upscaled[0] >= self.target[0]


def crop_box_for_aspect(src_w: int, src_h: int, target_w: int, target_h: int,
                        anchor: str = "center") -> tuple[int, int, int, int]:
    """Largest box in the source matching the target's aspect ratio."""
    if min(src_w, src_h, target_w, target_h) <= 0:
        raise ValueError("sizes must be positive")
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")

    # Compare src_w/src_h against target_w/target_h without floating point, so
    # an exact-ratio source is never cropped by a rounding hair.
    if src_w * target_h > target_w * src_h:  # source is wider — trim the sides
        crop_w = max(1, round(src_h * target_w / target_h))
        crop_h = src_h
    else:                                    # source is taller — trim top/bottom
        crop_w = src_w
        crop_h = max(1, round(src_w * target_h / target_w))
    crop_w, crop_h = min(crop_w, src_w), min(crop_h, src_h)

    if anchor == "left":
        left = 0
    elif anchor == "right":
        left = src_w - crop_w
    else:
        left = (src_w - crop_w) // 2
    if anchor == "top":
        top = 0
    elif anchor == "bottom":
        top = src_h - crop_h
    else:
        top = (src_h - crop_h) // 2
    return left, top, left + crop_w, top + crop_h


def plan(src_w: int, src_h: int, target_w: int, target_h: int,
         scales: Sequence[int] = (1, 2, 4), anchor: str = "center") -> FitPlan:
    """Plan crop + model scale + resample to land exactly on the target.

    Picks the smallest offered scale whose output covers the target, so the
    final resample shrinks (supersampling, which stays sharp) rather than
    stretches. If even the largest scale falls short, that largest one is used
    and the resample has to enlarge the rest of the way.
    """
    box = crop_box_for_aspect(src_w, src_h, target_w, target_h, anchor)
    crop_w, crop_h = box[2] - box[0], box[3] - box[1]

    usable = sorted(s for s in scales if s >= 1) or [1]
    scale = usable[-1]
    for s in usable:
        if crop_w * s >= target_w and crop_h * s >= target_h:
            scale = s
            break
    return FitPlan(crop_box=box, scale=scale, target=(target_w, target_h),
                   upscaled=(crop_w * scale, crop_h * scale))


def crop(image: Image.Image, target_w: int, target_h: int,
         anchor: str = "center") -> Image.Image:
    """Crop to the target's aspect ratio. Size is unchanged if it already matches."""
    box = crop_box_for_aspect(image.width, image.height, target_w, target_h, anchor)
    return image if box == (0, 0, image.width, image.height) else image.crop(box)


def resize_exact(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resample to exactly (target_w, target_h). Assumes the ratio already matches."""
    if image.size == (target_w, target_h):
        return image
    return image.resize((target_w, target_h), Image.LANCZOS)


def parse_target(text: str) -> Optional[tuple[int, int]]:
    """Read a custom '3440x1440' (or '3440 × 1440', '3440*1440') size.

    Returns None for anything unparseable, so the caller can fall back rather
    than raise at the user over a typo.
    """
    if not text:
        return None
    cleaned = text.strip().lower().replace("×", "x").replace("*", "x")
    cleaned = cleaned.replace(" ", "").replace(",", "")
    parts = cleaned.split("x")
    if len(parts) != 2:
        return None
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if w <= 0 or h <= 0 or w > 30000 or h > 30000:
        return None
    return w, h
