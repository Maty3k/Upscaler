"""App-level wiring for the depth region in the Blur tab."""

import numpy as np
import pytest
from PIL import Image


def _ramp(w=200, h=100):
    return Image.fromarray(np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1)), "L")


def _photo(w=200, h=100):
    return Image.new("RGB", (w, h), (120, 120, 120))


def test_app_depth_controls_show_only_for_the_depth_region():
    pytest.importorskip("gradio")
    import app
    from upscaler import blur

    focus, dof = app._blur_depth_vis(blur.DEPTH)
    assert focus["visible"] and dof["visible"]
    focus, dof = app._blur_depth_vis("whole")
    assert not focus["visible"] and not dof["visible"]


def test_app_mask_params_carry_the_depth_map():
    pytest.importorskip("gradio")
    import app
    from upscaler import blur

    dmap = _ramp()
    m = app._mask_params(blur.DEPTH, 50, 50, 50, 50, 0, 0, 0, False, True, 25, None,
                         None, dmap, 68.2, 30)
    assert m.shape == blur.DEPTH and m.depth is dmap
    assert m.focus == 68.2 and m.dof == 30.0
    # the default keeps every other tab's call working unchanged
    plain = app._mask_params("whole", 50, 50, 50, 50, 0, 0, 0, False, True, 25, None, None)
    assert plain.depth is None


def test_app_estimate_depth_only_runs_for_the_depth_region(monkeypatch):
    pytest.importorskip("gradio")
    import app
    from upscaler import blur, depth

    monkeypatch.setattr(depth, "estimate", lambda img, **kw: _ramp(*img.size))
    state, view, note = app.estimate_depth_ui(_photo(), "whole")
    assert state is None and not view["visible"]
    state, view, note = app.estimate_depth_ui(None, blur.DEPTH)
    assert state is None
    state, view, note = app.estimate_depth_ui(_photo(), blur.DEPTH)
    assert state.size == (200, 100) and view["visible"]
    assert "Click the picture" in note["value"]


def test_app_reports_a_missing_runtime_instead_of_crashing(monkeypatch):
    pytest.importorskip("gradio")
    import app
    from upscaler import blur, depth

    def boom(img, **kw):
        raise RuntimeError(depth.ONNX_HINT)

    monkeypatch.setattr(depth, "estimate", boom)
    state, view, note = app.estimate_depth_ui(_photo(), blur.DEPTH)
    assert state is None and not view["visible"] and "onnxruntime" in note["value"]


def test_app_click_to_focus_reads_the_depth_there():
    gr = pytest.importorskip("gradio")
    import app

    dmap = _ramp()

    class Evt:
        pass

    evt = Evt()
    evt.index = (dmap.width - 1, 50)                 # the near edge
    assert app.depth_focus_from_click(dmap, evt) == pytest.approx(100, abs=3)
    evt.index = (0, 50)                               # the far edge
    assert app.depth_focus_from_click(dmap, evt) == pytest.approx(0, abs=3)
    # nothing to read yet: leave the slider alone rather than guessing
    assert isinstance(app.depth_focus_from_click(None, evt), dict)


def test_app_blur_apply_refuses_depth_without_a_map():
    gr = pytest.importorskip("gradio")
    import app
    from upscaler import blur

    base = dict(kind="gaussian", strength=30, angle=0, cx=50, cy=50, highlights=0,
                threshold=25, shape=blur.DEPTH, x=50, y=50, w=50, h=50, mangle=0,
                roundness=0, feather=10, outside=False, progressive=True, face_pad=25,
                faces=None, editor=None, dmap=None, focus=70, dof=25)
    with pytest.raises(gr.Error, match="depth map"):
        app.blur_apply_ui(_photo(), *base.values(), progress=lambda *a, **k: None)
