"""Model architectures and the pretrained-weight registry."""

from upscaler.models.registry import MODELS, ModelSpec, resolve_model
from upscaler.models.rrdbnet import RRDBNet

__all__ = ["MODELS", "ModelSpec", "resolve_model", "RRDBNet"]
