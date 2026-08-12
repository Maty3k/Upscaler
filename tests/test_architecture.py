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


def _tiny_upscaler(tile_pad=8):
    """Real Upscaler with __init__ bypassed and a tiny untrained net."""
    up = Upscaler.__new__(Upscaler)  # bypass __init__/weight download
    up.spec = resolve_model(scale=4)
    up.device = torch.device("cpu")
    up.use_fp16 = False
    up._scale = up.spec.scale  # set by __init__ normally; bypassed here
    up._pad = None  # native RRDBNet padding path
    up.tile_pad = tile_pad
    torch.manual_seed(0)
    up.net = RRDBNet(scale=4, num_block=2).eval()  # tiny net, real weights not needed
    return up


def _quantize(t):
    """The whole-image float->uint8 conversion `upscale` used to end with."""
    arr = t.clamp(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    return np.round(arr * 255.0).astype(np.uint8)


def test_tiled_matches_untiled_on_random_weights():
    """Tiling must reconstruct the same result as a single pass (no seams)."""
    up = _tiny_upscaler()

    x = torch.rand(1, 3, 24, 24)
    with torch.inference_mode():
        full = up.net(x)
        up.tile = 16
        tiled = up._run_tiled(x)
    # The tiler now returns quantized uint8. Context padding keeps the float
    # tiles within ~1e-5 of the single pass, which after 8-bit rounding is at
    # most one grey level at a rounding boundary — but not zero, so this stays
    # a seam test, not a bit-exactness test (that's the one below).
    full_q = _quantize(full)
    assert tiled.shape == full_q.shape
    assert np.abs(tiled.astype(int) - full_q.astype(int)).max() <= 1


def test_per_tile_quantization_is_bit_identical_to_stitch_then_quantize():
    """The uint8 accumulator must not change a single pixel.

    The reference is computed the way `_run_tiled` used to work: stitch the
    float tiles into a float32 buffer with the same grid, then quantize the
    whole image at once. Tiles cover disjoint output regions and quantization
    is elementwise, so the per-tile version must be *bit-identical* —
    np.array_equal, not allclose. (Tiled vs untiled is deliberately not
    compared here: context padding makes those legitimately differ.)
    """
    up = _tiny_upscaler()
    up.tile = 16
    s = up.scale
    x = torch.rand(1, 3, 24, 20)  # 2x2 grid with ragged edge tiles

    with torch.inference_mode():
        new = up._run_tiled(x)

        # Reference: the old accumulator, same geometry as the tiler.
        b, c, h, w = x.shape
        ref = torch.zeros((b, c, h * s, w * s), dtype=x.dtype)
        for ty in range((h + up.tile - 1) // up.tile):
            for tx in range((w + up.tile - 1) // up.tile):
                x0, y0 = tx * up.tile, ty * up.tile
                x1, y1 = min(x0 + up.tile, w), min(y0 + up.tile, h)
                px0, py0 = max(x0 - up.tile_pad, 0), max(y0 - up.tile_pad, 0)
                px1, py1 = min(x1 + up.tile_pad, w), min(y1 + up.tile_pad, h)
                tile_out = up._net(x[:, :, py0:py1, px0:px1])
                ox0, oy0 = (x0 - px0) * s, (y0 - py0) * s
                ox1, oy1 = ox0 + (x1 - x0) * s, oy0 + (y1 - y0) * s
                ref[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = (
                    tile_out[:, :, oy0:oy1, ox0:ox1]
                )

    assert new.dtype == np.uint8
    assert np.array_equal(new, _quantize(ref))


def test_unsharp_mask_preserves_size():
    img = Image.fromarray((np.random.rand(20, 20, 3) * 255).astype("uint8"))
    out = unsharp_mask(img, strength=1.5)
    assert out.size == img.size and out.mode == "RGB"
