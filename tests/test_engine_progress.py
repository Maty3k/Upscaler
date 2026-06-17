"""Tests for the tile-count helper, CancelledError, and progress/cancel plumbing."""

import numpy as np
import pytest
import torch
from PIL import Image

from upscaler.engine import CancelledError, Upscaler, tile_count
from upscaler.models.rrdbnet import RRDBNet


def test_tile_count_helper():
    assert tile_count(100, 100, 512) == 1
    assert tile_count(1000, 1000, 512) == 4   # 2x2
    assert tile_count(1025, 100, 512) == 3     # 3x1
    assert tile_count(100, 100, 0) == 1        # tiling off
    for w, h, t in [(700, 300, 256), (513, 513, 512), (2048, 1024, 512)]:
        assert tile_count(w, h, t) == ((w + t - 1) // t) * ((h + t - 1) // t)


def test_cancelled_error_is_runtimeerror():
    assert issubclass(CancelledError, RuntimeError)


def _tiny_upscaler(tile):
    """Real Upscaler with __init__ bypassed and a tiny untrained net."""
    up = Upscaler.__new__(Upscaler)
    up.device = torch.device("cpu")
    up.use_fp16 = False
    up._scale = 4
    up._pad = None
    up.tile = tile
    up.tile_pad = 8
    torch.manual_seed(0)
    up.net = RRDBNet(scale=4, num_block=2).eval()
    return up


def test_progress_cb_called_per_tile_monotonic():
    up = _tiny_upscaler(tile=16)
    img = Image.fromarray((np.random.rand(40, 40, 3) * 255).astype("uint8"), "RGB")
    calls = []
    up.upscale(img, progress_cb=lambda d, t: calls.append((d, t)))
    n = tile_count(40, 40, 16)
    assert calls[-1] == (n, n)
    assert [d for d, _ in calls] == list(range(1, n + 1))  # monotonic 1..n


def test_should_cancel_stops_early():
    up = _tiny_upscaler(tile=16)
    img = Image.fromarray((np.random.rand(40, 40, 3) * 255).astype("uint8"), "RGB")
    with pytest.raises(CancelledError):
        up.upscale(img, should_cancel=lambda: True)


def test_progress_cb_called_once_when_tiling_off():
    up = _tiny_upscaler(tile=0)
    img = Image.fromarray((np.random.rand(24, 24, 3) * 255).astype("uint8"), "RGB")
    calls = []
    up.upscale(img, progress_cb=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 1)]
