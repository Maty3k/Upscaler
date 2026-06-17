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
    # How to build the network from these weights:
    #   "rrdbnet"  – the vendored Real-ESRGAN generator (the default; covers
    #                every official + community ESRGAN checkpoint here).
    #   "spandrel" – load via spandrel.ModelLoader, which auto-detects the
    #                architecture (HAT / DRCT / SwinIR / FBCNN / …). The scale
    #                is then discovered from the loaded model, NOT from `scale`.
    loader: str = "rrdbnet"


_REL = "https://github.com/xinntao/Real-ESRGAN/releases/download"
# Community ESRGAN weights (old-arch `model.*` layout, auto-converted on load by
# engine._convert_esrgan_oldarch). Mirror: huggingface.co/uwg/upscaler.
_HF = "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN"

MODELS: dict[str, ModelSpec] = {
    "realesrgan-x4plus": ModelSpec(
        name="realesrgan-x4plus",
        url=f"{_REL}/v0.1.0/RealESRGAN_x4plus.pth",
        filename="RealESRGAN_x4plus.pth",
        scale=4,
        num_block=23,
        sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        notes="General-purpose. A good default.",
    ),
    "realesrgan-x2plus": ModelSpec(
        name="realesrgan-x2plus",
        url=f"{_REL}/v0.2.1/RealESRGAN_x2plus.pth",
        filename="RealESRGAN_x2plus.pth",
        scale=2,
        num_block=23,
        sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        notes="General-purpose, gentler on already-good photos.",
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
    # -- Community ESRGAN models (old-arch, converted transparently on load) --
    "4x-ultrasharp": ModelSpec(
        name="4x-ultrasharp",
        url=f"{_HF}/4x-UltraSharp.pth",
        filename="4x-UltraSharp.pth",
        scale=4,
        sha256="a5812231fc936b42af08a5edba784195495d303d5b3248c24489ef0c4021fe01",
        notes="Crisper, more detailed than the ×4 default — great on JPEG (non-commercial).",
    ),
    "4x-remacri": ModelSpec(
        name="4x-remacri",
        url=f"{_HF}/4x_foolhardy_Remacri.pth",
        filename="4x_foolhardy_Remacri.pth",
        scale=4,
        sha256="e1a73bd89c2da1ae494774746398689048b5a892bd9653e146713f9df8bca86a",
        notes="Natural skin & textures (less waxy) — good for portraits (non-commercial).",
    ),
    "4x-nmkd-siax": ModelSpec(
        name="4x-nmkd-siax",
        url=f"{_HF}/4x_NMKD-Siax_200k.pth",
        filename="4x_NMKD-Siax_200k.pth",
        scale=4,
        sha256="560424d9f68625713fc47e9e7289a98aabe1d744e1cd6a9ae5a35e9957fd127e",
        notes="Detailed restore for clean / lightly-compressed photos.",
    ),
    "4x-nmkd-superscale": ModelSpec(
        name="4x-nmkd-superscale",
        url=f"{_HF}/4x_NMKD-Superscale-SP_178000_G.pth",
        filename="4x_NMKD-Superscale-SP_178000_G.pth",
        scale=4,
        sha256="1d1b0078fe71446e0469d8d4df59e96baa80d83cda600d68237d655830821bcc",
        notes="Conservative, natural restore for everyday photos.",
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
        sha256="cd685efaae01f7c4e9951f2deab05780079c8eb1e49ed664b72f6db04dabb445",
        notes="Denoise (SIDD) — removes sensor noise / grain (not motion blur). ~443MB.",
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


# -- JPEG-artifact removal (optional, via the [face] extra) ------------------
# FBCNN de-blocks heavily-compressed JPEGs before upscaling. Loaded via spandrel
# (the arch ships in core spandrel), so no extra dependency beyond [face].

@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    url: str
    filename: str
    sha256: Optional[str] = None
    notes: str = ""


ARTIFACT_MODELS: dict[str, ArtifactSpec] = {
    "fbcnn-color": ArtifactSpec(
        name="fbcnn-color",
        url="https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth",
        filename="fbcnn_color.pth",
        sha256="8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8",
        notes="FBCNN — removes JPEG blocking/ringing artifacts (color). "
        "Run it before upscaling a heavily-compressed photo. Research use. ~288MB.",
    ),
}
DEFAULT_ARTIFACT_MODEL = "fbcnn-color"


def resolve_artifact_model(model: Optional[str] = None) -> ArtifactSpec:
    name = model or DEFAULT_ARTIFACT_MODEL
    if name not in ARTIFACT_MODELS:
        raise ValueError(
            f"Unknown artifact model {name!r}. Available: {', '.join(sorted(ARTIFACT_MODELS))}"
        )
    return ARTIFACT_MODELS[name]


# -- Face restoration (optional, via the [face] extra) -----------------------

@dataclass(frozen=True)
class FaceSpec:
    name: str
    url: str
    filename: str
    sha256: Optional[str] = None
    notes: str = ""
    # True if the model takes a per-call fidelity weight (CodeFormer's `w`):
    # higher = truer to the original face, lower = stronger (freer) restoration.
    fidelity: bool = False


FACE_MODELS: dict[str, FaceSpec] = {
    "gfpgan-v1.4": FaceSpec(
        name="gfpgan-v1.4",
        url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        filename="GFPGANv1.4.pth",
        sha256="e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad",
        notes="GFPGAN v1.4 — gentle, natural face restoration (Apache-2.0). ~349MB.",
    ),
    "codeformer": FaceSpec(
        name="codeformer",
        url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        filename="codeformer.pth",
        sha256="1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7",
        notes="CodeFormer — stronger restoration with an adjustable fidelity dial; "
        "best on badly degraded faces (S-Lab License 1.0, non-commercial). ~360MB.",
        fidelity=True,
    ),
}
DEFAULT_FACE_MODEL = "gfpgan-v1.4"

# OpenCV YuNet face detector (gives the 5 landmarks used to align each face).
FACE_DETECTOR = FaceSpec(
    name="yunet",
    url="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    filename="face_detection_yunet_2023mar.onnx",
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    notes="YuNet face detector (OpenCV Zoo).",
)


# -- Supply-chain guard ------------------------------------------------------

def iter_pinned_specs():
    """Yield every weight spec that MUST ship sha256-pinned over https.

    This is the single source of truth for the CI registry guard
    (``tests/test_registry_guard.py``). When a later batch adds a new model
    registry (e.g. FBCNN artifact removal, DDColor colorize, LaMa inpaint),
    append it here so the guard covers it automatically.

    Background-removal weights (``upscaler.background.BG_MODELS``) are
    deliberately excluded: they are served unpinned (sha256=None) today.
    """
    yield from MODELS.values()
    yield from DEBLUR_MODELS.values()
    yield from ARTIFACT_MODELS.values()
    yield from FACE_MODELS.values()
    yield FACE_DETECTOR
