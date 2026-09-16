"""App-level wiring for the file-size budget mode in the Convert tab."""

import os

import numpy as np
import pytest
from PIL import Image


def _photo(w=800, h=800):
    rng = np.random.default_rng(0)
    base = np.linspace(0, 255, w, dtype=np.float32)[None, :, None].repeat(h, 0).repeat(3, 2)
    return Image.fromarray(np.clip(base + rng.normal(0, 40, (h, w, 3)), 0, 255)
                           .astype(np.uint8), "RGB")


def test_app_convert_methods_include_the_budget_mode():
    pytest.importorskip("gradio")
    import app

    assert len(app._CONVERT_METHODS) == 4
    assert "Fit a file-size budget" in app._CONVERT_METHODS
    # the switcher returns one update per method, in order
    updates = app._switch_method(app._CONVERT_METHODS[1])
    assert len(updates) == len(app._CONVERT_METHODS)
    assert updates[1]["visible"] and not updates[0]["visible"]


def test_app_optimize_hits_the_budget_and_writes_a_file(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    import app
    from upscaler import library

    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "lib")
    pair, path, note = app.optimize_ui(_photo(), "80KB", "auto", 40, True, 0,
                                       progress=lambda *a, **k: None)
    assert os.path.getsize(path) <= 80 * 1024 and path.endswith(".webp")
    assert pair[0].size == (800, 800)          # before is the original
    assert "✅" in note and "Budget" in note


def test_app_optimize_reports_when_it_cannot_reach_the_target(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    import app
    from upscaler import library

    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "lib")
    _pair, _path, note = app.optimize_ui(_photo(), "2KB", "auto", 40, False, 0,
                                         progress=lambda *a, **k: None)
    assert "⚠" in note and "still over" in note


def test_app_optimize_errors():
    gr = pytest.importorskip("gradio")
    import app

    with pytest.raises(gr.Error):
        app.optimize_ui(None, "500KB", "auto", 40, True, 0, progress=lambda *a, **k: None)
    with pytest.raises(gr.Error):
        app.optimize_ui(_photo(200, 200), "nonsense", "auto", 40, True, 0,
                        progress=lambda *a, **k: None)
