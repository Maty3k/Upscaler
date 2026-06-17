"""Tests for FBCNN JPEG-artifact removal (the Clean-up checkbox)."""

import types

import numpy as np
import pytest
from PIL import Image

from upscaler.models.registry import (
    ARTIFACT_MODELS,
    DEFAULT_ARTIFACT_MODEL,
    resolve_artifact_model,
)


def test_fbcnn_spec_pinned():
    s = ARTIFACT_MODELS["fbcnn-color"]
    assert s.sha256 and len(s.sha256) == 64
    assert s.url.startswith("https://") and s.filename


def test_fbcnn_in_registry_and_default():
    assert ARTIFACT_MODELS
    assert resolve_artifact_model().name == DEFAULT_ARTIFACT_MODEL


def test_fbcnn_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown artifact model"):
        resolve_artifact_model("nope")


def test_artifact_remover_rejects_unknown_model():
    pytest.importorskip("torch")
    from upscaler.restore import ArtifactRemover

    with pytest.raises(ValueError, match="Unknown artifact model"):
        ArtifactRemover(model="nope", device="cpu")


def test_enhance_threads_fbcnn_flag(monkeypatch):
    pytest.importorskip("gradio")
    import app

    seen = {}

    class _StubFBCNN:
        def restore(self, img):
            seen["called"] = True
            return img

    class _StubUp:
        scale = 2
        device = types.SimpleNamespace(type="cpu")

        def upscale(self, img):
            return img

    monkeypatch.setattr(app, "_get_fbcnn", lambda device: _StubFBCNN())
    monkeypatch.setattr(app, "_get_upscaler", lambda *a, **k: _StubUp())
    monkeypatch.setattr(app.library, "save_image", lambda *a, **k: None)

    img = Image.fromarray(np.zeros((8, 8, 3), dtype="uint8"), "RGB")
    (_orig, _result), info = app.enhance(
        img, "realesrgan-x4plus", "cpu", False, "nafnet-gopro-width64",
        1.0, 0, 0, False, "", fbcnn=True,
    )
    assert seen.get("called")
    assert "FBCNN" in info
