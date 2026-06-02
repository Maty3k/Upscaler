"""Local, open-source image upscaling + sharpening built on pretrained Real-ESRGAN weights."""

from upscaler.engine import Upscaler
from upscaler.pipeline import enhance

__version__ = "0.1.0"
__all__ = ["Upscaler", "enhance", "__version__"]
