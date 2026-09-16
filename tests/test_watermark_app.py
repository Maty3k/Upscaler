"""App-level wiring for the Watermark tab."""

import pytest
from PIL import Image, ImageDraw


def _vals(**over):
    import app
    from upscaler import watermark as wm

    base = {name: getattr(wm.WatermarkParams(), name) for name in app._WM_FIELDS}
    base.update({k: v for k, v in over.items() if k in base})
    return list(base.values())


def _photo(w=600, h=400):
    return Image.new("RGB", (w, h), (120, 120, 120))


def _logo():
    img = Image.new("RGBA", (120, 60), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([0, 0, 119, 59], fill=(255, 0, 0, 255))
    return img


def test_app_wm_fields_match_the_dataclass():
    pytest.importorskip("gradio")
    import app
    from upscaler import watermark as wm

    names = {f.name for f in wm.WatermarkParams.__dataclass_fields__.values()}
    assert set(app._WM_FIELDS) == names
    assert len(app._WM_FIELDS) == len(names)
    assert app._WM_TEXT <= names


def test_app_wm_params_typing():
    pytest.importorskip("gradio")
    import app

    p = app._wm_params(*_vals(kind="logo", size=6.0, opacity=55.0, text=None))
    assert p.kind == "logo" and p.size == 6.0 and p.opacity == 55.0
    assert p.text == "" and isinstance(p.margin, float)


def test_app_wm_visibility():
    pytest.importorskip("gradio")
    import app
    from upscaler import watermark as wm

    v = app._wm_vis("text", "bottom right")
    assert len(v) == 12
    assert v[0]["visible"] and not v[7]["visible"]       # text shown, logo hidden
    assert v[9]["visible"] and not v[10]["visible"]      # margin shown, tile gap hidden
    v = app._wm_vis("logo", wm.TILED)
    assert not v[0]["visible"] and v[7]["visible"]       # logo shown, text hidden
    assert not v[9]["visible"] and v[10]["visible"] and v[11]["visible"]


def test_app_wm_preset_fans_out():
    pytest.importorskip("gradio")
    import app
    from upscaler import watermark as wm

    values = app.watermark_preset_ui("Proof (tiled)")
    assert len(values) == len(app._WM_FIELDS)
    assert app._wm_params(*values) == wm.PRESETS["Proof (tiled)"]
    assert app._wm_params(*app.watermark_preset_ui("None")).is_identity()


def test_app_wm_preview_and_missing_logo_note():
    pytest.importorskip("gradio")
    import app

    assert app.watermark_preview_ui(None, None, *_vals())[0] is None
    out, note = app.watermark_preview_ui(_photo(), None, *_vals(text="© Hi"))
    assert out is not None and '"© Hi"' in note["value"]
    # a logo mark with no logo yet warns instead of raising
    out2, note2 = app.watermark_preview_ui(_photo(), None, *_vals(kind="logo"))
    assert out2 is not None and "⚠" in note2["value"]
    out3, note3 = app.watermark_preview_ui(_photo(), _logo(), *_vals(kind="logo"))
    assert out3 is not None and "logo at" in note3["value"]


def test_app_wm_apply(tmp_path, monkeypatch):
    gr = pytest.importorskip("gradio")
    import app
    from upscaler import library

    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "lib")
    out, path, info = app.watermark_apply_ui(_photo(), None, *_vals(text="© Me"),
                                             progress=lambda *a, **k: None)
    assert out.size == (600, 400) and path.endswith(".png") and '"© Me"' in info
    with pytest.raises(gr.Error):
        app.watermark_apply_ui(None, None, *_vals(), progress=lambda *a, **k: None)
    with pytest.raises(gr.Error):
        app.watermark_apply_ui(_photo(), None, *_vals(kind="logo"),
                               progress=lambda *a, **k: None)
