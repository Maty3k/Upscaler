"""App-level wiring for the Blur tab: control → params mapping, the
visibility helpers, and the preview / apply handlers (no server)."""

import pytest
from PIL import Image, ImageDraw


def _vals(**over):
    base = dict(kind="gaussian", strength=30, angle=0, cx=50, cy=50, highlights=0, threshold=25,
                shape="whole", x=50, y=50, w=50, h=50, mangle=0, roundness=0, feather=10,
                outside=False, progressive=True, face_pad=25, faces=None, editor=None)
    base.update(over)
    return list(base.values())


def test_app_blur_params_mapping():
    pytest.importorskip("gradio")
    import app

    bp, mp = app._blur_split(_vals(kind="motion", strength=55, angle=30, shape="band", h=20,
                                   mangle=15, outside=True, feather=12.5))
    assert (bp.kind, bp.strength, bp.angle) == ("motion", 55.0, 30.0)
    assert (mp.shape, mp.h, mp.angle, mp.outside, mp.feather, mp.progressive) == (
        "band", 20.0, 15.0, True, 12.5, True)
    assert mp.painted is None


def test_app_blur_visibility_helpers():
    pytest.importorskip("gradio")
    import app

    kind = app._blur_kind_vis("spin")
    assert len(kind) == 6 and kind[1]["visible"] and kind[2]["visible"] and not kind[0]["visible"]
    assert "Rotation" in kind[5]
    shape = app._blur_shape_vis("band")
    assert len(shape) == 11
    assert shape[7]["value"] is True            # band defaults to "blur outside"
    assert shape[4]["visible"] and not shape[5]["visible"] and not shape[10]["visible"]
    whole = app._blur_shape_vis("whole")
    assert not any(u["visible"] for u in whole[:7]) and not whole[10]["visible"]
    painted = app._blur_shape_vis("painted")
    assert painted[10]["visible"] and painted[7]["value"] is False
    faces = app._blur_shape_vis("faces")
    assert faces[9]["visible"] and not faces[10]["visible"]   # padding shown, brush hidden


def test_app_blur_preview_and_apply(tmp_path, monkeypatch):
    gr = pytest.importorskip("gradio")
    import app
    from upscaler import library

    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "lib")
    img = Image.new("RGB", (1200, 600), (90, 90, 90))
    ImageDraw.Draw(img).rectangle([500, 200, 700, 400], fill=(255, 255, 255))
    assert app.blur_preview_ui(None, *_vals()) is None
    before, after = app.blur_preview_ui(img, *_vals(strength=50))
    assert before.size == (900, 450) == after.size          # preview is downscaled
    pair, path, info = app.blur_apply_ui(img, *_vals(kind="pixelate", shape="ellipse", strength=60),
                                         progress=lambda *a, **k: None)
    assert pair[0].size == (1200, 600) == pair[1].size and path.endswith(".png")
    assert "pixelate" in info and "inside the ellipse" in info
    with pytest.raises(gr.Error):
        app.blur_apply_ui(img, *_vals(shape="painted"), progress=lambda *a, **k: None)
    with pytest.raises(gr.Error):   # region = faces, but none were detected
        app.blur_apply_ui(img, *_vals(shape="faces"), progress=lambda *a, **k: None)
    with pytest.raises(gr.Error):
        app.blur_apply_ui(None, *_vals(), progress=lambda *a, **k: None)


def test_app_blur_painted_mask_from_editor(tmp_path):
    pytest.importorskip("gradio")
    import app

    bg = Image.new("RGB", (200, 100), (50, 50, 50))
    layer = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    ImageDraw.Draw(layer).rectangle([0, 0, 99, 99], fill=(255, 255, 255, 255))
    editor = {"background": bg, "layers": [layer], "composite": bg}
    _bp, mp = app._blur_split(_vals(shape="painted", editor=editor, feather=0))
    assert mp.painted is not None and mp.painted.getpixel((10, 10)) == 255
    assert mp.painted.getpixel((150, 50)) == 0
