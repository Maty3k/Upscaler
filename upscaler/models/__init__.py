"""Model architectures and the pretrained-weight registry."""

from upscaler.models.nafnet import NAFNet
from upscaler.models.registry import (
    DEBLUR_MODELS,
    DEFAULT_DEBLUR_MODEL,
    MODELS,
    DeblurSpec,
    ModelSpec,
    resolve_deblur_model,
    resolve_model,
)
from upscaler.models.rrdbnet import RRDBNet

__all__ = [
    "MODELS",
    "ModelSpec",
    "resolve_model",
    "RRDBNet",
    "NAFNet",
    "DEBLUR_MODELS",
    "DEFAULT_DEBLUR_MODEL",
    "DeblurSpec",
    "resolve_deblur_model",
]
