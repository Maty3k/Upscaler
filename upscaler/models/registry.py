"""Registry of pretrained Real-ESRGAN models and how to build/load each one.

Weights are downloaded lazily on first use (see ``upscaler.models.weights``) and
cached under ``upscaler/weights/``. They are NOT committed to the repo: they are
large and carry their own upstream license terms.

All weights here are from the official Real-ESRGAN releases (BSD-3-Clause):
https://github.com/xinntao/Real-ESRGAN
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    scale: int
    filename: str
    num_block: int = 23
    num_feat: int = 64
    num_grow_ch: int = 32
    # SHA-256 of the .pth, verified after download when set. Pin these before a
    # release for supply-chain integrity (see scripts/print_checksums.py).
    sha256: Optional[str] = None
    notes: str = ""


_REL = "https://github.com/xinntao/Real-ESRGAN/releases/download"

MODELS: dict[str, ModelSpec] = {
    "realesrgan-x4plus": ModelSpec(
        name="realesrgan-x4plus",
        url=f"{_REL}/v0.1.0/RealESRGAN_x4plus.pth",
        filename="RealESRGAN_x4plus.pth",
        scale=4,
        num_block=23,
        notes="General-purpose 4x. Good default.",
    ),
    "realesrgan-x2plus": ModelSpec(
        name="realesrgan-x2plus",
        url=f"{_REL}/v0.2.1/RealESRGAN_x2plus.pth",
        filename="RealESRGAN_x2plus.pth",
        scale=2,
        num_block=23,
        notes="General-purpose 2x.",
    ),
    "realesrgan-x4plus-anime": ModelSpec(
        name="realesrgan-x4plus-anime",
        url=f"{_REL}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        filename="RealESRGAN_x4plus_anime_6B.pth",
        scale=4,
        num_block=6,
        notes="Lighter 6-block model tuned for anime / illustration / line art.",
    ),
}

# Default model chosen for a requested integer scale factor.
_DEFAULT_FOR_SCALE = {2: "realesrgan-x2plus", 4: "realesrgan-x4plus"}


def resolve_model(model: Optional[str] = None, scale: Optional[int] = None) -> ModelSpec:
    """Pick a ModelSpec from an explicit name, or fall back to one for ``scale``."""
    if model:
        if model not in MODELS:
            raise ValueError(
                f"Unknown model {model!r}. Available: {', '.join(sorted(MODELS))}"
            )
        return MODELS[model]
    scale = scale or 4
    if scale not in _DEFAULT_FOR_SCALE:
        raise ValueError(
            f"No default model for scale={scale}. Pass --model explicitly, or use "
            f"a supported scale: {sorted(_DEFAULT_FOR_SCALE)}."
        )
    return MODELS[_DEFAULT_FOR_SCALE[scale]]
