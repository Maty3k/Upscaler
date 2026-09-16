"""App-level wiring for the Color & Light tab: the control list must stay in
step with AdjustParams, and preset / auto / apply must round-trip through it."""

import pytest
from PIL import Image, ImageDraw


def _vals(**over):
    """The tab's control values in _ADJUST_FIELDS order, then the region ones."""
    import app

    base = dict(zip(app._ADJUST_FIELDS,
                    (0, 0, 0, 0, 0, 100, 1.0, 0, 0, 0, 0, 0, 0, False, 30, 59, 11, "#d8b070", 0)))
    base.update({k: v for k, v in over.items() if k in base})
    region = [over.get("shape", "whole"), 50, 50, 50, 50, 0, 0, 15,
              over.get("outside", False), 25, over.get("faces"), over.get("editor")]
    return list(base.values()) + region


def test_app_adjust_fields_match_the_dataclass():
    pytest.importorskip("gradio")
    import app
    from upscaler import adjust

    names = {f.name for f in adjust.AdjustParams.__dataclass_fields__.values()}
    assert set(app._ADJUST_FIELDS) == names
    assert len(app._ADJUST_FIELDS) == len(names)      # no duplicates, no drift


def test_app_adjust_params_and_mask_mapping():
    pytest.importorskip("gradio")
    import app

    p, m = app._adjust_split(_vals(exposure=25, mono=True, tone_color="#112233",
                                   shape="band", outside=True))
    assert (p.exposure, p.mono, p.tone_color) == (25.0, True, "#112233")
    assert p.white_point == 100.0 and p.gamma == 1.0
    assert (m.shape, m.outside, m.progressive) == ("band", True, False)


def test_app_preset_and_auto_fan_out_to_every_control(tmp_path):
    pytest.importorskip("gradio")
    import app
    from upscaler import adjust

    values = app.adjust_preset_ui("Punchy")
    assert len(values) == len(app._ADJUST_FIELDS)
    p, _m = app._adjust_split(list(values) + _vals()[len(app._ADJUST_FIELDS):])
    assert p == adjust.PRESETS["Punchy"]
    assert app._adjust_split(list(app.adjust_preset_ui("None")) +
                             _vals()[len(app._ADJUST_FIELDS):])[0].is_identity()

    img = Image.new("RGB", (80, 60), (40, 50, 70))
    ImageDraw.Draw(img).rectangle([0, 0, 40, 30], fill=(190, 180, 170))
    auto = app.adjust_auto_ui(img, *_vals(clarity=30))
    p2, _m = app._adjust_split(list(auto) + _vals()[len(app._ADJUST_FIELDS):])
    assert p2.clarity == 30 and not p2.is_identity()      # auto keeps what it doesn't own


def test_app_adjust_preview_and_apply(tmp_path, monkeypatch):
    gr = pytest.importorskip("gradio")
    import app
    from upscaler import library

    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "lib")
    img = Image.new("RGB", (1400, 700), (70, 90, 120))
    assert app.adjust_preview_ui(None, *_vals()) is None
    before, after = app.adjust_preview_ui(img, *_vals(exposure=40))
    assert before.size == (900, 450) == after.size
    assert after.getpixel((10, 10))[0] > before.getpixel((10, 10))[0]
    pair, path, info = app.adjust_apply_ui(img, *_vals(mono=True, contrast=20),
                                           progress=lambda *a, **k: None)
    assert pair[0].size == (1400, 700) == pair[1].size and path.endswith(".png")
    assert "black & white" in info and "contrast +20" in info
    with pytest.raises(gr.Error):
        app.adjust_apply_ui(None, *_vals(), progress=lambda *a, **k: None)
    with pytest.raises(gr.Error):
        app.adjust_apply_ui(img, *_vals(shape="painted"), progress=lambda *a, **k: None)
    with pytest.raises(gr.Error):   # region = faces, but none were detected
        app.adjust_apply_ui(img, *_vals(shape="faces"), progress=lambda *a, **k: None)
    with pytest.raises(gr.Error):
        app.adjust_auto_ui(None, *_vals())


def test_app_region_vis_shared_by_both_tabs():
    pytest.importorskip("gradio")
    import app

    ten = app._region_vis("band")
    assert len(ten) == 10 and ten[7]["value"] is True         # band → "outside" on
    eleven = app._blur_shape_vis("band")
    assert len(eleven) == 11 and eleven[8]["visible"] is True  # blur's extra "graded" toggle
    assert eleven[:8] == ten[:8] and eleven[9:] == ten[8:]
    assert app._region_vis("faces")[8]["visible"] is True      # face padding
