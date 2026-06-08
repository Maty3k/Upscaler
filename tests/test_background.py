"""Tests for background removal + the denoise registry entry.

These avoid any model download/inference (network-free): they cover the
registry, error handling, and the checkerboard compositor.
"""

from __future__ import annotations

import pytest
from PIL import Image

from upscaler import background
from upscaler.models.registry import DEBLUR_MODELS


def test_bg_models_registry():
    assert "u2net" in background.BG_MODELS
    assert background.DEFAULT_BG_MODEL in background.BG_MODELS
    for spec in background.BG_MODELS.values():
        assert spec.url.endswith(".onnx")
        assert spec.filename.endswith(".onnx")


def test_remove_background_unknown_model():
    with pytest.raises(ValueError):
        background.remove_background(Image.new("RGB", (8, 8)), model="does-not-exist")


def test_on_checkerboard_size_and_mode():
    rgba = Image.new("RGBA", (20, 16), (255, 0, 0, 128))
    out = background.on_checkerboard(rgba)
    assert out.size == (20, 16)
    assert out.mode == "RGB"


def test_sidd_denoise_registered():
    spec = DEBLUR_MODELS.get("nafnet-sidd-width64")
    assert spec is not None
    # the SIDD architecture that matched the weights at strict load
    assert spec.width == 64
    assert spec.enc_blk_nums == (2, 2, 4, 8)
    assert spec.middle_blk_num == 12
    assert spec.dec_blk_nums == (2, 2, 2, 2)
