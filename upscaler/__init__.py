"""Local, open-source image upscaling + sharpening built on pretrained Real-ESRGAN weights."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from upscaler.deblur import Deblurrer
from upscaler.engine import Upscaler
from upscaler.pipeline import enhance

# Single source of truth is pyproject.toml — read it from the installed
# distribution ("local-upscaler" on PyPI) so this can't drift on a release.
try:
    __version__ = _pkg_version("local-upscaler")
except PackageNotFoundError:  # running from a checkout that isn't pip-installed
    __version__ = "0.0.0+unknown"

__all__ = ["Upscaler", "Deblurrer", "enhance", "__version__"]
