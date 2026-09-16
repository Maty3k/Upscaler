"""App-level wiring for the Sharpen tab: the control list must stay in step
with SharpenParams, and the 1:1 preview must report what it is showing."""

import pytest
from PIL import Image, ImageDraw, ImageFilter


def _vals(**over):
    """Control values in _SHARPEN_FIELDS order, then the region ones, then the
    two preview-centre sliders."""
    import app
    from upscaler import sharpen

    base = {name: getattr(sharpen.SharpenParams(), name) for name in app._SHARPEN_FIELDS}
    base.update({k: v for k, v in over.items() if k in base})
    region = [over.get("shape", "whole"), 50, 50, 50, 50, 0, 0, 15,
              over.get("outside", False), 25, over.get("faces"), over.get("editor")]
    return list(base.values()) + region + [over.get("cx", 50), over.get("cy", 50)]


def _photo(w=2000, h=1400):
    img = Image.new("RGB", (w, h), (80, 80, 80))
    ImageDraw.Draw(img).rectangle([w // 2, 0, w, h], fill=(200, 200, 200))
    return img.filter(ImageFilter.GaussianBlur(2))


def test_app_sharpen_fields_match_the_dataclass():
    pytest.importorskip("gradio")
    import app
    from upscaler import sharpen

    names = {f.name for f in sharpen.SharpenParams.__dataclass_fields__.values()}
    assert set(app._SHARPEN_FIELDS) == names
    assert len(app._SHARPEN_FIELDS) == len(names)


def test_app_sharpen_split_maps_params_mask_and_centre():
    pytest.importorskip("gradio")
    import app

    p, m, center = app._sharpen_split(_vals(kind="smart", amount=150, radius=2.0,
                                            luminance_only=False, shape="band",
                                            outside=True, cx=25, cy=75))
    assert (p.kind, p.amount, p.radius, p.luminance_only) == ("smart", 150.0, 2.0, False)
    assert (m.shape, m.outside, m.progressive) == ("band", True, False)
    assert center == (0.25, 0.75)


def test_app_sharpen_kind_visibility():
    pytest.importorskip("gradio")
    import app

    edge, balance = app._sharpen_kind_vis("smart")
    assert edge["visible"] and not balance["visible"]
    edge, balance = app._sharpen_kind_vis("texture")
    assert balance["visible"] and not edge["visible"]
    edge, balance = app._sharpen_kind_vis("unsharp")
    assert not edge["visible"] and not balance["visible"]


def test_app_sharpen_preset_fans_out():
    pytest.importorskip("gradio")
    import app
    from upscaler import sharpen

    values = app.sharpen_preset_ui("Portrait (skin-safe)")
    assert len(values) == len(app._SHARPEN_FIELDS)
    p, _m, _c = app._sharpen_split(list(values) + _vals()[len(app._SHARPEN_FIELDS):])
    assert p == sharpen.PRESETS["Portrait (skin-safe)"]
    cleared, _m, _c = app._sharpen_split(list(app.sharpen_preset_ui("None")) +
                                        _vals()[len(app._SHARPEN_FIELDS):])
    assert cleared.is_identity()


def test_app_sharpen_preview_is_a_crop_and_says_so():
    pytest.importorskip("gradio")
    import app

    assert app.sharpen_preview_ui(None, *_vals())[0] is None
    pair, note = app.sharpen_preview_ui(_photo(), *_vals(amount=150))
    before, after = pair
    assert before.size == after.size == (app.sharpen_tools.PREVIEW_EDGE,) * 2
    assert "1:1 crop" in note["value"] and "2000×1400" in note["value"]
    # a photo that already fits is shown whole
    _pair, note2 = app.sharpen_preview_ui(Image.new("RGB", (400, 300)), *_vals())
    assert "whole photo" in note2["value"]


def test_app_sharpen_apply(tmp_path, monkeypatch):
    gr = pytest.importorskip("gradio")
    import app
    from upscaler import library

    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "lib")
    img = _photo(600, 400)
    pair, path, info = app.sharpen_apply_ui(img, *_vals(amount=120),
                                            progress=lambda *a, **k: None)
    assert pair[0].size == (600, 400) == pair[1].size and path.endswith(".png")
    assert "unsharp" in info and "amount 120" in info
    with pytest.raises(gr.Error):
        app.sharpen_apply_ui(None, *_vals(), progress=lambda *a, **k: None)
    with pytest.raises(gr.Error):
        app.sharpen_apply_ui(img, *_vals(shape="painted"), progress=lambda *a, **k: None)
    with pytest.raises(gr.Error):
        app.sharpen_apply_ui(img, *_vals(shape="faces"), progress=lambda *a, **k: None)
