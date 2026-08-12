"""ONNX export + runtime tests. Use torch only to build a reference; the ONNX
path itself runs on onnxruntime. No weight downloads."""

import numpy as np
import onnxruntime as ort
import pytest
import torch

from upscaler.models.nafnet import NAFNet
from upscaler.models.rrdbnet import RRDBNet
from upscaler.onnx_engine import OnnxUpscaler
from upscaler.onnx_export import _export


def _session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("scale", [4, 2])
def test_onnx_rrdbnet_matches_torch_at_new_size(tmp_path, scale):
    """Export at 32x32, run at a different size — proves dynamic axes + parity."""
    torch.manual_seed(0)
    net = RRDBNet(scale=scale, num_block=2).eval()
    path = _export(net, torch.rand(1, 3, 32, 32), tmp_path / "up.onnx")

    x = np.random.rand(1, 3, 24, 16).astype("float32")
    onnx_out = _session(path).run(None, {"input": x})[0]
    with torch.inference_mode():
        torch_out = net(torch.from_numpy(x)).numpy()

    assert onnx_out.shape == (1, 3, 24 * scale, 16 * scale)
    np.testing.assert_allclose(onnx_out, torch_out, atol=2e-3)


def test_onnx_nafnet_body_matches_torch_at_new_size(tmp_path):
    torch.manual_seed(0)
    net = NAFNet(width=16, middle_blk_num=1, enc_blk_nums=(1, 1), dec_blk_nums=(1, 1)).eval()

    class _Body(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.body(x)

    path = _export(_Body(net), torch.rand(1, 3, 32, 32), tmp_path / "db.onnx")
    x = np.random.rand(1, 3, 16, 24).astype("float32")  # divisible by padder (4)
    onnx_out = _session(path).run(None, {"input": x})[0]
    with torch.inference_mode():
        torch_out = net.body(torch.from_numpy(x)).numpy()

    assert onnx_out.shape == x.shape
    np.testing.assert_allclose(onnx_out, torch_out, atol=2e-3)


def _stub_upscaler():
    """OnnxUpscaler over a local nearest-4x stand-in session — no export."""
    class _StubSess:
        def run(self, _names, feed):
            x = next(iter(feed.values()))
            return [np.repeat(np.repeat(x, 4, axis=2), 4, axis=3)]  # local nearest 4x

    up = OnnxUpscaler.__new__(OnnxUpscaler)
    up.scale = 4
    up.tile_pad = 8
    up.sess = _StubSess()
    up._inp = "input"
    return up


def _quantize(chw):
    """The whole-image float->uint8 conversion `_to_image` applies."""
    return np.round(np.clip(chw, 0.0, 1.0)[0].transpose(1, 2, 0) * 255.0).astype(np.uint8)


def test_onnx_numpy_tiler_stitches_local_op():
    """The numpy tiler must reconstruct a purely-local 4x op exactly.

    The op is local, so tiling introduces no context effects and the tiler's
    uint8 output must equal the quantized single pass bit-for-bit.
    """
    up = _stub_upscaler()
    x = np.random.rand(1, 3, 24, 20).astype("float32")
    up.tile = 0
    full = up._run(x)
    up.tile = 16
    tiled = up._run_tiled(x)
    assert np.array_equal(_quantize(full), tiled)


def test_onnx_per_tile_quantization_is_bit_identical_to_stitch_then_quantize():
    """Mirror of the torch tiler's bit-exactness contract.

    Reference computed the old way: stitch float tiles into a float32 buffer
    with the same grid, then quantize whole. Disjoint tiles + elementwise
    quantization means the per-tile version must be bit-identical.
    """
    up = _stub_upscaler()
    up.tile = 16
    s = up.scale
    x = np.random.rand(1, 3, 24, 20).astype("float32")

    new = up._run_tiled(x)

    b, c, h, w = x.shape
    ref = np.zeros((b, c, h * s, w * s), dtype=np.float32)
    for ty in range((h + up.tile - 1) // up.tile):
        for tx in range((w + up.tile - 1) // up.tile):
            x0, y0 = tx * up.tile, ty * up.tile
            x1, y1 = min(x0 + up.tile, w), min(y0 + up.tile, h)
            px0, py0 = max(x0 - up.tile_pad, 0), max(y0 - up.tile_pad, 0)
            px1, py1 = min(x1 + up.tile_pad, w), min(y1 + up.tile_pad, h)
            tile_out = up._run(x[:, :, py0:py1, px0:px1])
            ox0, oy0 = (x0 - px0) * s, (y0 - py0) * s
            ox1, oy1 = ox0 + (x1 - x0) * s, oy0 + (y1 - y0) * s
            ref[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = tile_out[:, :, oy0:oy1, ox0:ox1]

    assert new.dtype == np.uint8
    assert np.array_equal(new, _quantize(ref))
