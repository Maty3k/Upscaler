"""Reproduce-then-verify tests for the high-severity bug fixes:
odd-dimension ×2 tiling, network-error wrapping, ffmpeg-failure surfacing, and
the CLI directory-vs-single-file guard.
"""

import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image

from upscaler import cli, panel
from upscaler.engine import Upscaler
from upscaler.models import weights


def _img(w, h):
    return Image.fromarray((np.random.rand(h, w, 3) * 255).astype("uint8"), "RGB")


def test_upscale_scale2_odd_dimensions_no_crash():
    # ×2 models pixel-unshuffle by 2, so odd H/W used to trip the divisibility
    # assert in RRDBNet — both non-tiled and at an odd edge tile.
    u = Upscaler(model="realesrgan-x2plus", device="cpu", tile=0)
    assert u.upscale(_img(33, 21)).size == (66, 42)
    ut = Upscaler(model="realesrgan-x2plus", device="cpu", tile=16)
    assert ut.upscale(_img(51, 51)).size == (102, 102)


def test_onnx_upscale_scale2_odd_dimensions_no_crash():
    pytest.importorskip("onnxruntime")
    from upscaler.onnx_engine import OnnxUpscaler

    u = OnnxUpscaler(model="realesrgan-x2plus", device="cpu", tile=0)
    assert u.upscale(_img(33, 21)).size == (66, 42)


def test_download_network_error_becomes_runtimeerror(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("name or service not known")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="internet connection"):
        weights._download("https://example.invalid/x.pth", tmp_path / "x.pth")


def test_ffmpeg_run_raises_on_nonzero_exit():
    import sys

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        panel._ffmpeg_run([sys.executable, "-c", "import sys; sys.exit(1)"])


def _dir_with_two_images(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(2):
        Image.new("RGB", (8, 8), (i * 50, i * 50, i * 50)).save(d / f"{i}.png")
    return d


def test_cli_convert_dir_to_single_file_is_rejected(tmp_path, capsys):
    d = _dir_with_two_images(tmp_path)
    rc = cli.run_convert([str(d), "-o", str(tmp_path / "out.png")])
    assert rc == 2
    assert "must be a directory" in capsys.readouterr().err


def test_cli_upscale_dir_to_single_file_is_rejected(tmp_path, capsys):
    d = _dir_with_two_images(tmp_path)
    rc = cli.main([str(d), "-o", str(tmp_path / "out.png")])
    assert rc == 2
    assert "must be a directory" in capsys.readouterr().err
