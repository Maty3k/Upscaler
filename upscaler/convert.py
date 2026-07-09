"""Image format conversion (PNG / JPEG / WebP / BMP / TIFF / ...).

Pure Pillow, no models — fast and local. Handles the awkward bits: flattening
alpha onto a background for formats that can't store it (JPEG), quality for
lossy formats, and lossless WebP.
"""

from __future__ import annotations

import io
from typing import Optional

from PIL import Image

# HEIC/HEIF (iPhone photos) via the optional pillow-heif plugin.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _HEIF_OK = True
except Exception:  # pragma: no cover - plugin missing
    _HEIF_OK = False

# Display name -> (Pillow format, file extension, lossy?)
FORMATS: dict[str, tuple[str, str, bool]] = {
    "PNG": ("PNG", "png", False),
    "JPEG": ("JPEG", "jpg", True),
    "WebP": ("WEBP", "webp", True),
    "AVIF": ("AVIF", "avif", True),
    "JPEG 2000": ("JPEG2000", "jp2", False),
    "TIFF": ("TIFF", "tiff", False),
    "GIF": ("GIF", "gif", False),
    "BMP": ("BMP", "bmp", False),
    "ICO": ("ICO", "ico", False),
    "ICNS": ("ICNS", "icns", False),
    "TGA": ("TGA", "tga", False),
    "PCX": ("PCX", "pcx", False),
    "DIB": ("DIB", "dib", False),
    "SGI": ("SGI", "sgi", False),
    "PPM": ("PPM", "ppm", False),
}
if _HEIF_OK:
    FORMATS["HEIC"] = ("HEIF", "heic", True)

# AVIF needs a codec: built into Pillow 11.2+ wheels, else the pillow-heif
# plugin. Without one, offering it would end in an opaque KeyError at save time
# — drop it from the menu instead.
try:
    from PIL import features as _pil_features

    _AVIF_OK = bool(_pil_features.check("avif"))
except Exception:  # pragma: no cover - very old Pillow
    _AVIF_OK = False
if not _AVIF_OK:
    try:  # pragma: no cover - depends on installed pillow-heif build
        from pillow_heif import register_avif_opener

        register_avif_opener()
        _AVIF_OK = True
    except Exception:
        pass
if not _AVIF_OK:  # pragma: no cover
    FORMATS.pop("AVIF", None)

# Formats that cannot store an alpha channel — alpha is flattened onto a bg.
_NO_ALPHA = {"JPEG", "BMP", "PPM", "PCX"}


def extension_for(fmt: str) -> str:
    return FORMATS[fmt][1]


def convert(
    image: Image.Image,
    fmt: str,
    *,
    quality: int = 90,
    background: tuple[int, int, int] = (255, 255, 255),
    lossless: bool = False,
) -> bytes:
    """Convert ``image`` to ``fmt`` and return the encoded file as bytes.

    ``quality`` (1–100) applies to lossy formats (JPEG/WebP). ``background`` is
    used to flatten transparency when the target can't keep it. ``lossless``
    forces lossless WebP.
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unsupported format {fmt!r}. Choose from {', '.join(FORMATS)}.")
    pil_fmt, _, lossy = FORMATS[fmt]

    img = image
    if pil_fmt in _NO_ALPHA and img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, background)
        flat.paste(rgba, mask=rgba.split()[-1])
        img = flat
    elif pil_fmt in _NO_ALPHA:
        img = img.convert("RGB")
    elif img.mode == "P":
        img = img.convert("RGBA")

    save_kwargs: dict = {}
    if pil_fmt == "JPEG":
        save_kwargs.update(quality=int(quality), optimize=True)
    elif pil_fmt in ("AVIF", "HEIF"):
        save_kwargs.update(quality=int(quality))
    elif pil_fmt == "WEBP":
        if lossless:
            save_kwargs.update(lossless=True)
        else:
            save_kwargs.update(quality=int(quality))

    buf = io.BytesIO()
    img.save(buf, format=pil_fmt, **save_kwargs)
    return buf.getvalue()


def convert_file(
    src,
    dst,
    fmt: Optional[str] = None,
    *,
    quality: int = 90,
    lossless: bool = False,
) -> None:
    """Convert an image file on disk. ``fmt`` defaults to inferring from ``dst``."""
    if fmt is None:
        ext = str(dst).rsplit(".", 1)[-1].lower()
        match = next((k for k, v in FORMATS.items() if v[1] == ext), None)
        if match is None:
            raise ValueError(f"Can't infer format from {dst!r}; pass fmt=...")
        fmt = match
    data = convert(Image.open(src), fmt, quality=quality, lossless=lossless)
    with open(dst, "wb") as f:
        f.write(data)
