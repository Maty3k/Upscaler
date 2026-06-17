"""Tests for CodeFormer as a second, fidelity-adjustable face model."""

import types

import numpy as np
import pytest
from PIL import Image

from upscaler.models.registry import DEFAULT_FACE_MODEL, FACE_MODELS


def test_codeformer_registered_and_pinned():
    assert "codeformer" in FACE_MODELS
    s = FACE_MODELS["codeformer"]
    assert s.sha256 and len(s.sha256) == 64
    assert s.url.startswith("https://") and s.filename


def test_default_face_model_unchanged():
    # Adding CodeFormer must not silently change the default behaviour.
    assert DEFAULT_FACE_MODEL == "gfpgan-v1.4"


def test_fidelity_flag_only_on_codeformer():
    assert FACE_MODELS["codeformer"].fidelity is True
    assert FACE_MODELS["gfpgan-v1.4"].fidelity is False


@pytest.mark.parametrize("model", ["gfpgan-v1.4", "codeformer"])
def test_no_face_image_passes_through(model):
    pytest.importorskip("cv2")
    pytest.importorskip("spandrel")
    pytest.importorskip("spandrel_extra_arches")
    from upscaler.face import FaceRestorer

    try:
        fr = FaceRestorer(model=model, device="cpu")  # only the tiny YuNet detector
    except (OSError, RuntimeError) as e:
        pytest.skip(f"face detector unavailable offline: {e}")
    noise = Image.fromarray((np.random.rand(96, 96, 3) * 255).astype("uint8"), "RGB")
    out = fr.restore(noise, fidelity=0.3)
    # no faces -> identical image, and the heavy net is never loaded
    assert np.array_equal(np.asarray(out), np.asarray(noise))
    assert fr._net is None


def test_unknown_face_model_raises():
    pytest.importorskip("cv2")
    pytest.importorskip("spandrel")
    from upscaler.face import FaceRestorer

    with pytest.raises(ValueError, match="Unknown face model"):
        FaceRestorer(model="nope", device="cpu")


def test_enhance_threads_face_model_and_fidelity(monkeypatch):
    pytest.importorskip("gradio")
    import app

    seen = {}

    class _StubFR:
        def restore(self, img, strength, fidelity=0.5):
            seen["strength"] = strength
            seen["fidelity"] = fidelity
            return img

    class _StubUp:
        scale = 2
        device = types.SimpleNamespace(type="cpu")

        def upscale(self, img):
            return img

    def _fake_get_fr(model, device):
        seen["model"] = model
        return _StubFR()

    monkeypatch.setattr(app, "_get_face_restorer", _fake_get_fr)
    monkeypatch.setattr(app, "_get_upscaler", lambda *a, **k: _StubUp())
    monkeypatch.setattr(app.library, "save_image", lambda *a, **k: None)

    img = Image.fromarray(np.zeros((8, 8, 3), dtype="uint8"), "RGB")
    (_orig, _result), info = app.enhance(
        img, "realesrgan-x4plus", "cpu", False, "nafnet-gopro-width64",
        1.0, 0, 0, False, "", face=True, face_strength=0.8,
        face_model="codeformer", face_fidelity=0.3,
    )
    assert seen["model"] == "codeformer"
    assert seen["fidelity"] == 0.3
    assert "codeformer" in info
