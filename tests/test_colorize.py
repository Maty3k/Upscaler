"""Tests for DDColor colorization (the Colorize tab)."""

import numpy as np
import pytest
from PIL import Image

from upscaler.models.registry import (
    COLORIZE_MODELS,
    DEFAULT_COLORIZE_MODEL,
    resolve_colorize_model,
)


def test_colorize_models_pinned():
    assert COLORIZE_MODELS
    for s in COLORIZE_MODELS.values():
        assert s.sha256 and len(s.sha256) == 64
        assert s.url.startswith("https://") and s.filename
    assert resolve_colorize_model().name == DEFAULT_COLORIZE_MODEL


def test_colorize_unknown_model_raises():
    from upscaler.colorize import Colorizer

    with pytest.raises(ValueError, match="Unknown colorize model"):
        Colorizer(model="nope", device="cpu")  # checked before any download


def test_colorize_deps_message(monkeypatch):
    # When the spandrel extra is missing, load_spandrel raises the friendly
    # install message; the Colorizer must surface it unchanged.
    from upscaler import colorize as C

    monkeypatch.setattr(C, "ensure_weights", lambda spec: "dummy.pth")

    def _boom(*a, **k):
        raise RuntimeError('needs extra packages. Install them with: pip install -e ".[face]"')

    monkeypatch.setattr(C, "load_spandrel", _boom)
    c = C.Colorizer(model="ddcolor", device="cpu")
    with pytest.raises(RuntimeError, match=r"\[face\]"):
        c.colorize(Image.new("L", (8, 8)))


def test_colorize_grayscale_returns_rgb_same_size():
    pytest.importorskip("spandrel")
    pytest.importorskip("spandrel_extra_arches")
    from upscaler.colorize import Colorizer

    try:
        c = Colorizer(device="cpu")
        g = Image.fromarray(np.tile(np.linspace(0, 255, 64).astype("uint8"), (64, 1)), "L")
        out = c.colorize(g, strength=1.0)
    except (OSError, RuntimeError) as e:
        pytest.skip(f"DDColor weights unavailable offline: {e}")
    assert out.size == (64, 64) and out.mode == "RGB"


def test_colorize_strength_zero_is_grayscale():
    pytest.importorskip("spandrel")
    pytest.importorskip("spandrel_extra_arches")
    from upscaler.colorize import Colorizer

    try:
        c = Colorizer(device="cpu")
        g = Image.fromarray(np.tile(np.linspace(0, 255, 64).astype("uint8"), (64, 1)), "L")
        out = c.colorize(g, strength=0.0)
    except (OSError, RuntimeError) as e:
        pytest.skip(f"DDColor weights unavailable offline: {e}")
    a = np.asarray(out)
    # strength 0 -> source as grayscale RGB (all channels equal)
    assert np.allclose(a[..., 0], a[..., 1], atol=1)
    assert np.allclose(a[..., 1], a[..., 2], atol=1)
