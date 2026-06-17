"""Architecture/plumbing tests that run on CPU without downloading weights."""

import numpy as np
import pytest
import torch
from PIL import Image

from upscaler.engine import Upscaler, resolve_device
from upscaler.models.rrdbnet import RRDBNet, pixel_unshuffle
from upscaler.models.registry import resolve_model
from upscaler.sharpen import unsharp_mask


@pytest.mark.parametrize("scale,num_block", [(4, 23), (2, 23), (4, 6)])
def test_rrdbnet_output_shape(scale, num_block):
    net = RRDBNet(scale=scale, num_block=num_block).eval()
    x = torch.rand(1, 3, 32, 32)
    with torch.inference_mode():
        y = net(x)
    assert y.shape == (1, 3, 32 * scale, 32 * scale)


def test_pixel_unshuffle_roundtrip_shape():
    x = torch.rand(1, 3, 16, 16)
    assert pixel_unshuffle(x, 2).shape == (1, 12, 8, 8)


def test_resolve_model_defaults():
    assert resolve_model(scale=4).scale == 4
    assert resolve_model(scale=2).scale == 2
    assert resolve_model(model="realesrgan-x4plus-anime").num_block == 6
    with pytest.raises(ValueError):
        resolve_model(model="does-not-exist")


def test_resolve_device_returns_torch_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_tiled_matches_untiled_on_random_weights():
    """Tiling must reconstruct the same result as a single pass (no seams)."""
    up = Upscaler.__new__(Upscaler)  # bypass __init__/weight download
    up.spec = resolve_model(scale=4)
    up.device = torch.device("cpu")
    up.use_fp16 = False
    up._scale = up.spec.scale  # set by __init__ normally; bypassed here
    up._pad = None  # native RRDBNet padding path
    up.tile_pad = 8
    torch.manual_seed(0)
    up.net = RRDBNet(scale=4, num_block=2).eval()  # tiny net, real weights not needed

    x = torch.rand(1, 3, 24, 24)
    with torch.inference_mode():
        full = up.net(x)
        up.tile = 16
        tiled = up._run_tiled(x)
    assert torch.allclose(full, tiled, atol=1e-5)


def test_unsharp_mask_preserves_size():
    img = Image.fromarray((np.random.rand(20, 20, 3) * 255).astype("uint8"))
    out = unsharp_mask(img, strength=1.5)
    assert out.size == img.size and out.mode == "RGB"
