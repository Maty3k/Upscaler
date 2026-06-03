"""Image ⇄ PDF tests."""

import io

import numpy as np
import pytest
from PIL import Image

from upscaler.document import images_to_pdf, pdf_to_images


def _img(size=(40, 30), color=None):
    arr = np.full((size[1], size[0], 3), color or 128, dtype="uint8")
    return Image.fromarray(arr, "RGB")


def test_images_to_pdf_produces_pdf_bytes():
    data = images_to_pdf([_img()])
    assert data[:5] == b"%PDF-"


def test_multipage_pdf_page_count():
    data = images_to_pdf([_img(), _img(), _img()])
    pages = pdf_to_images(data, dpi=72)
    assert len(pages) == 3


def test_images_to_pdf_empty_raises():
    with pytest.raises(ValueError):
        images_to_pdf([])


def test_roundtrip_size_at_72dpi():
    """At 72 dpi a page renders back to roughly the source pixel size."""
    src = _img(size=(96, 48))
    pages = pdf_to_images(images_to_pdf([src]), dpi=72)
    assert len(pages) == 1
    w, h = pages[0].size
    assert abs(w - 96) <= 2 and abs(h - 48) <= 2


def test_images_to_pdf_flattens_rgba():
    rgba = Image.fromarray(np.zeros((10, 10, 4), "uint8"), "RGBA")
    assert images_to_pdf([rgba])[:5] == b"%PDF-"  # no crash on alpha
