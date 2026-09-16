"""App-level wiring for the Crop & Frame tab."""

import pytest
from PIL import Image, ImageDraw


def _vals(**over):
    import app
    from upscaler import frame

    base = {name: getattr(frame.FrameParams(), name) for name in app._FRAME_FIELDS}
    base.update({k: v for k, v in over.items() if k in base})
    return list(base.values())


def _photo(w=800, h=600):
    img = Image.new("RGB", (w, h), (120, 120, 120))
    ImageDraw.Draw(img).rectangle([0, 0, w // 2, h // 2], fill=(220, 30, 30))
    return img


def test_app_frame_fields_match_the_dataclass():
    pytest.importorskip("gradio")
    import app
    from upscaler import frame

    names = {f.name for f in frame.FrameParams.__dataclass_fields__.values()}
    assert set(app._FRAME_FIELDS) == names
    assert len(app._FRAME_FIELDS) == len(names)
    assert app._FRAME_TEXT | app._FRAME_BOOL <= names


def test_app_frame_params_typing():
    pytest.importorskip("gradio")
    import app

    p = app._frame_params(*_vals(rotate=90.0, flip_h=1, border=8.0, aspect="Square · 1:1",
                                 zoom=2.0, out_size=None))
    assert p.rotate == 90 and isinstance(p.rotate, int)
    assert p.flip_h is True and p.border == 8.0 and p.out_size == ""
    assert p.aspect == "Square · 1:1" and p.zoom == 2.0


def test_app_frame_visibility():
    pytest.importorskip("gradio")
    import app
    from upscaler import frame

    custom, mode, color, blur = app._frame_vis(frame.CUSTOM_ASPECT, 0, "solid")
    assert custom["visible"] and mode["visible"] and not color["visible"]
    custom, mode, color, blur = app._frame_vis("Original", 10, "solid")
    assert not custom["visible"] and not mode["visible"] and color["visible"]
    _c, _m, color, blur = app._frame_vis("Square · 1:1", 10, "blurred photo")
    assert blur["visible"] and not color["visible"]


def test_app_frame_preset_fans_out():
    pytest.importorskip("gradio")
    import app
    from upscaler import frame

    values = app.frame_preset_ui("Polaroid")
    assert len(values) == len(app._FRAME_FIELDS)
    assert app._frame_params(*values) == frame.PRESETS["Polaroid"]
    assert app._frame_params(*app.frame_preset_ui("None")).is_identity()


def test_app_frame_preview_reports_the_output_size():
    pytest.importorskip("gradio")
    import app

    assert app.frame_preview_ui(None, *_vals())[0] is None
    out, note = app.frame_preview_ui(_photo(), *_vals(aspect="Square · 1:1", border=10))
    assert out is not None
    assert "800×600" in note["value"] and "crop to" in note["value"]
    # a bad custom ratio is reported, not raised
    bad, note2 = app.frame_preview_ui(_photo(), *_vals(aspect="Custom…", custom_aspect="xx"))
    assert bad is None and "⚠" in note2["value"]


def test_app_frame_apply(tmp_path, monkeypatch):
    gr = pytest.importorskip("gradio")
    import app
    from upscaler import library

    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "lib")
    out, path, info = app.frame_apply_ui(_photo(), *_vals(aspect="Square · 1:1", border=6),
                                         progress=lambda *a, **k: None)
    assert out.size[0] == out.size[1] and path.endswith(".png")
    assert "crop to" in info and "800×600 →" in info
    with pytest.raises(gr.Error):
        app.frame_apply_ui(None, *_vals(), progress=lambda *a, **k: None)
    with pytest.raises(gr.Error):
        app.frame_apply_ui(_photo(), *_vals(aspect="Custom…", custom_aspect="xx"),
                           progress=lambda *a, **k: None)
