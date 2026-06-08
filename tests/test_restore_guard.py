"""Tests for the restoration garbage-guard (_structural_ok).

A real deblur/denoise preserves image structure; a failed one (e.g. GoPro
motion-deblur on a grainy photo) returns garbage. The guard skips the latter.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

import app


def _img(arr):
    return Image.fromarray(arr.astype("uint8"), "RGB")


def test_structural_ok_true_for_faithful_restore():
    rng = np.random.default_rng(0)
    base = rng.random((48, 48, 3)) * 255
    a = _img(base)
    b = _img(np.clip(base * 0.95 + 4, 0, 255))  # mild change — structure intact
    assert app._structural_ok(a, b) is True


def test_structural_ok_false_for_garbage():
    ramp = np.tile(np.linspace(0, 255, 48), (48, 1))[:, :, None].repeat(3, 2)
    a = _img(ramp)
    b = _img(np.random.default_rng(1).integers(0, 256, (48, 48, 3)))  # uncorrelated
    assert app._structural_ok(a, b) is False
