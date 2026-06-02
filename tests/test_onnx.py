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


def test_onnx_numpy_tiler_stitches_local_op():
    """The numpy tiler must reconstruct a purely-local 4x op exactly."""
    class _StubSess:
        def run(self, _names, feed):
            x = next(iter(feed.values()))
            return [np.repeat(np.repeat(x, 4, axis=2), 4, axis=3)]  # local nearest 4x

    up = OnnxUpscaler.__new__(OnnxUpscaler)
    up.scale = 4
    up.tile_pad = 8
    up.sess = _StubSess()
    up._inp = "input"

    x = np.random.rand(1, 3, 24, 20).astype("float32")
    up.tile = 0
    full = up._run(x)
    up.tile = 16
    tiled = up._run_tiled(x)
    np.testing.assert_allclose(full, tiled, atol=1e-6)
