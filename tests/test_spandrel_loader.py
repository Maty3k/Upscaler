"""Tests for the arch-agnostic spandrel loader path in the engine.

All of these run fully offline: a fake ``spandrel`` module is injected via
``sys.modules`` and ``ensure_weights`` is monkeypatched, so no weights download
and the real spandrel/torch arch code is never touched.
"""

import sys
import types

import numpy as np
import pytest
from PIL import Image
from torch.nn import functional as F

from upscaler import engine
from upscaler.models.registry import ModelSpec, resolve_model


class _FakeDesc:
    """Stand-in for a spandrel ImageModelDescriptor: callable, with the handful
    of attributes load_spandrel reads."""

    def __init__(self, scale=2, supports_half=False, in_ch=3, out_ch=3, size_req=None):
        self.scale = scale
        self.supports_half = supports_half
        self.input_channels = in_ch
        self.output_channels = out_ch
        self.size_requirements = size_req

    def to(self, device):
        return self

    def eval(self):
        return self

    def half(self):
        return self

    def __call__(self, t):
        return F.interpolate(t, scale_factor=self.scale, mode="nearest")


def _install_fake_spandrel(monkeypatch, desc):
    spandrel = types.ModuleType("spandrel")

    class ModelLoader:
        def load_from_file(self, path):
            return desc

    spandrel.ModelLoader = ModelLoader
    sea = types.ModuleType("spandrel_extra_arches")
    sea.install = lambda: None
    monkeypatch.setitem(sys.modules, "spandrel", spandrel)
    monkeypatch.setitem(sys.modules, "spandrel_extra_arches", sea)


def _spandrel_upscaler(monkeypatch, desc, **kwargs):
    _install_fake_spandrel(monkeypatch, desc)
    monkeypatch.setattr(engine, "ensure_weights", lambda spec: "dummy.pth")
    spec = ModelSpec(
        name="fake-spandrel",
        url="https://example.test/fake.pth",
        scale=99,  # deliberately wrong: spandrel path must ignore spec.scale
        filename="fake.pth",
        sha256="0" * 64,
        loader="spandrel",
    )
    monkeypatch.setattr(engine, "resolve_model", lambda model=None, scale=None: spec)
    return engine.Upscaler(model="fake-spandrel", device="cpu", **kwargs)


def test_native_path_unchanged():
    # The opt-in field exists with the safe default, and every shipped model
    # stays on the native RRDBNet path.
    assert ModelSpec(name="x", url="u", scale=4, filename="f").loader == "rrdbnet"
    assert resolve_model("realesrgan-x4plus").loader == "rrdbnet"


def test_spandrel_branch_builds_and_uses_descriptor_scale(monkeypatch):
    up = _spandrel_upscaler(monkeypatch, _FakeDesc(scale=2), tile=0)
    assert up.scale == 2  # discovered from the model, not spec.scale (99)
    out = up.upscale(Image.fromarray(np.zeros((8, 8, 3), dtype="uint8"), "RGB"))
    assert out.size == (16, 16)


def test_spandrel_scale_drives_tiler(monkeypatch):
    # With tiling on, the tiler must also honour the descriptor scale.
    up = _spandrel_upscaler(monkeypatch, _FakeDesc(scale=2), tile=4)
    out = up.upscale(Image.fromarray(np.zeros((10, 10, 3), dtype="uint8"), "RGB"))
    assert out.size == (20, 20)


def test_spandrel_fp16_gated_by_supports_half(monkeypatch):
    # Even with fp16 requested and supports_half=True, CPU/MPS must stay fp32.
    up = _spandrel_upscaler(monkeypatch, _FakeDesc(supports_half=True), tile=0, fp16=True)
    assert up.use_fp16 is False


def test_spandrel_rejects_non_rgb(monkeypatch):
    with pytest.raises(RuntimeError, match="channel"):
        _spandrel_upscaler(monkeypatch, _FakeDesc(in_ch=1, out_ch=3), tile=0)
