"""Registry of pretrained Real-ESRGAN models and how to build/load each one.

Weights are downloaded lazily on first use (see ``upscaler.models.weights``) and
cached under ``upscaler/weights/``. They are NOT committed to the repo: they are
large and carry their own upstream license terms.

All weights here are from the official Real-ESRGAN releases (BSD-3-Clause):
https://github.com/xinntao/Real-ESRGAN
"""

from __future__ import annotations

from dataclasses import dataclass
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
        sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        notes="General-purpose 4x. Good default.",
    ),
    "realesrgan-x2plus": ModelSpec(
        name="realesrgan-x2plus",
        url=f"{_REL}/v0.2.1/RealESRGAN_x2plus.pth",
        filename="RealESRGAN_x2plus.pth",
        scale=2,
        num_block=23,
        sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        notes="General-purpose 2x.",
    ),
    "realesrgan-x4plus-anime": ModelSpec(
        name="realesrgan-x4plus-anime",
        url=f"{_REL}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        filename="RealESRGAN_x4plus_anime_6B.pth",
        scale=4,
        num_block=6,
        sha256="f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
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


# --- Deblur models (NAFNet) ------------------------------------------------
# Weights mirrored on Hugging Face (resolve/ gives a direct download); the
# upstream originals live on the official NAFNet Google Drive. NAFNet is MIT.


@dataclass(frozen=True)
class DeblurSpec:
    name: str
    url: str
    filename: str
    width: int
    middle_blk_num: int
    enc_blk_nums: tuple
    dec_blk_nums: tuple
    sha256: Optional[str] = None
    notes: str = ""


_HF = "https://huggingface.co/nyanko7/nafnet-models/resolve/main"

DEBLUR_MODELS: dict[str, DeblurSpec] = {
    "nafnet-gopro-width64": DeblurSpec(
        name="nafnet-gopro-width64",
        url=f"{_HF}/NAFNet-GoPro-width64.pth",
        filename="NAFNet-GoPro-width64.pth",
        width=64,
        middle_blk_num=1,
        enc_blk_nums=(1, 1, 1, 28),
        dec_blk_nums=(1, 1, 1, 1),
        sha256="329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f",
        notes="Motion deblur (GoPro), full quality. ~272MB.",
    ),
    "nafnet-gopro-width32": DeblurSpec(
        name="nafnet-gopro-width32",
        url=f"{_HF}/NAFNet-GoPro-width32.pth",
        filename="NAFNet-GoPro-width32.pth",
        width=32,
        middle_blk_num=1,
        enc_blk_nums=(1, 1, 1, 28),
        dec_blk_nums=(1, 1, 1, 1),
        sha256="19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c",
        notes="Motion deblur (GoPro), lighter/faster. ~69MB.",
    ),
    # Same NAFNet architecture, trained on SIDD for denoising rather than
    # deblurring — runs through the identical Deblurrer forward pass.
    "nafnet-sidd-width64": DeblurSpec(
        name="nafnet-sidd-width64",
        url=f"{_HF}/NAFNet-SIDD-width64.pth",
        filename="NAFNet-SIDD-width64.pth",
        width=64,
        middle_blk_num=12,
        enc_blk_nums=(2, 2, 4, 8),
        dec_blk_nums=(2, 2, 2, 2),
        sha256=None,  # pin before a release (see scripts/print_checksums.py)
        notes="Denoise (SIDD) — removes sensor noise / grain (not motion blur). ~272MB.",
    ),
}

DEFAULT_DEBLUR_MODEL = "nafnet-gopro-width64"


def resolve_deblur_model(model: Optional[str] = None) -> DeblurSpec:
    name = model or DEFAULT_DEBLUR_MODEL
    if name not in DEBLUR_MODELS:
        raise ValueError(
            f"Unknown deblur model {name!r}. Available: {', '.join(sorted(DEBLUR_MODELS))}"
        )
    return DEBLUR_MODELS[name]
