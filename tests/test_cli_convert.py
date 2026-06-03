"""CLI `convert` subcommand tests — no models/downloads."""

import numpy as np
import pytest
from PIL import Image

from upscaler.cli import main


def _make(path, mode="RGB", size=(20, 16)):
    arr = (np.random.rand(size[1], size[0], len(mode)) * 255).astype("uint8")
    Image.fromarray(arr.squeeze(), mode=mode).save(path)


def test_convert_infers_format_from_output(tmp_path):
    src = tmp_path / "a.png"
    _make(src)
    dst = tmp_path / "a.webp"
    assert main(["convert", str(src), "-o", str(dst)]) == 0
    assert Image.open(dst).format == "WEBP"


def test_convert_explicit_format_default_name(tmp_path):
    src = tmp_path / "a.png"
    _make(src)
    assert main(["convert", str(src), "-f", "JPEG"]) == 0
    assert (tmp_path / "a.jpg").exists()


def test_convert_batch_directory(tmp_path):
    for n in ("a", "b", "c"):
        _make(tmp_path / f"{n}.png")
    out = tmp_path / "out"
    assert main(["convert", str(tmp_path), "-o", str(out), "-f", "WebP"]) == 0
    assert sorted(p.name for p in out.glob("*.webp")) == ["a.webp", "b.webp", "c.webp"]


def test_convert_missing_format_errors(tmp_path):
    src = tmp_path / "a.png"
    _make(src)
    # output is a directory (no extension) and no --format -> error
    assert main(["convert", str(src), "-o", str(tmp_path)]) == 2


def test_convert_missing_input_errors():
    assert main(["convert", "/no/such/file.png", "-f", "PNG"]) == 2


def test_bare_input_still_routes_to_upscaler(monkeypatch, tmp_path):
    """`upscaler <input>` (no subcommand) must NOT be treated as convert."""
    called = {}

    class _FakeUp:
        spec = type("S", (), {"name": "fake"})()
        scale = 2
        device = type("D", (), {"type": "cpu"})()

        def __init__(self, **k):
            called["built"] = True

        def upscale(self, img):
            return img

    monkeypatch.setattr("upscaler.cli.Upscaler", _FakeUp)
    src = tmp_path / "a.png"
    _make(src)
    rc = main([str(src), "-o", str(tmp_path / "o.png"), "--scale", "2"])
    assert rc == 0 and called.get("built")
