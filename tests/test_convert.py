"""File-converter tests — pure Pillow, no models or downloads."""

import io

import numpy as np
import pytest
from PIL import Image

from upscaler.convert import FORMATS, convert, convert_file, extension_for


def _img(mode="RGB", size=(24, 16)):
    arr = (np.random.rand(size[1], size[0], len(mode)) * 255).astype("uint8")
    return Image.fromarray(arr.squeeze(), mode=mode)


@pytest.mark.parametrize("fmt", list(FORMATS))
def test_convert_roundtrips_each_format(fmt):
    data = convert(_img(), fmt, quality=85)
    out = Image.open(io.BytesIO(data))
    assert out.format == FORMATS[fmt][0]
    # ICO/ICNS are icon containers that rescale to standard sizes; others preserve.
    if fmt not in ("ICO", "ICNS"):
        assert out.size == (24, 16)


def test_jpeg_flattens_alpha_without_error():
    rgba = _img(mode="RGBA")
    out = Image.open(io.BytesIO(convert(rgba, "JPEG")))
    assert out.mode == "RGB"  # alpha dropped onto background, no crash


def test_png_preserves_alpha():
    rgba = _img(mode="RGBA")
    out = Image.open(io.BytesIO(convert(rgba, "PNG")))
    assert out.mode in ("RGBA", "LA", "P")


def test_webp_lossless_vs_lossy_differ():
    img = _img()
    lossy = convert(img, "WebP", quality=10)
    lossless = convert(img, "WebP", lossless=True)
    assert lossy != lossless  # different encodings


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        convert(_img(), "XYZ")


def test_convert_file_infers_format_from_extension(tmp_path):
    src = tmp_path / "in.png"
    _img().save(src)
    dst = tmp_path / "out.webp"
    convert_file(src, dst)
    assert dst.exists()
    assert Image.open(dst).format == "WEBP"


def test_extension_lookup():
    assert extension_for("JPEG") == "jpg"
    assert extension_for("PNG") == "png"
