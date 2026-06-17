"""Tests for LaMa object removal / inpainting (the Inpaint tab)."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from upscaler.models.registry import (
    DEFAULT_INPAINT_MODEL,
    INPAINT_MODELS,
    resolve_inpaint_model,
)


def test_inpaint_models_pinned():
    assert INPAINT_MODELS
    for s in INPAINT_MODELS.values():
        assert s.sha256 and len(s.sha256) == 64
        assert s.url.startswith("https://") and s.filename
    assert resolve_inpaint_model().name == DEFAULT_INPAINT_MODEL


def test_inpaint_unknown_model_raises():
    from upscaler.inpaint import Inpainter

    with pytest.raises(ValueError, match="Unknown inpaint model"):
        Inpainter(model="nope", device="cpu")  # checked before any download


def test_mask_from_editor_layer():
    pytest.importorskip("gradio")
    import app

    bg = Image.new("RGB", (20, 16), (10, 10, 10))
    layer = Image.new("RGBA", (20, 16), (0, 0, 0, 0))
    ImageDraw.Draw(layer).rectangle([5, 5, 8, 8], fill=(255, 255, 255, 255))

    out_bg, mask = app._mask_from_editor({"background": bg, "layers": [layer]})
    assert out_bg.size == (20, 16)
    assert mask.mode == "L" and mask.size == (20, 16)
    arr = np.asarray(mask)
    assert arr[6, 6] == 255  # painted
    assert arr[0, 0] == 0    # untouched


def test_inpaint_empty_inputs_raise():
    pytest.importorskip("gradio")
    import app

    with pytest.raises(app.gr.Error):  # no image at all
        app.inpaint_ui(None, "big-lama")

    bg = Image.new("RGB", (16, 16), (20, 20, 20))
    empty = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    with pytest.raises(app.gr.Error):  # nothing painted -> never loads the model
        app.inpaint_ui({"background": bg, "layers": [empty]}, "big-lama")


def test_inpaint_runs_and_preserves_size():
    pytest.importorskip("torch")
    from upscaler.inpaint import Inpainter

    try:
        ip = Inpainter(device="cpu")
        img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"), "RGB")
        mask = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(mask).rectangle([20, 20, 40, 40], fill=255)
        out = ip.inpaint(img, mask)
    except (OSError, RuntimeError) as e:
        pytest.skip(f"LaMa weights unavailable offline: {e}")
    assert out.size == (64, 64) and out.mode == "RGB"
