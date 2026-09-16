"""App-level wiring for the Effects tab: the control list must stay in step
with EffectParams, and looks / apply must round-trip through it."""

import pytest
from PIL import Image, ImageDraw


def _vals(**over):
    """The tab's control values in _EFFECT_FIELDS order, then the region ones."""
    import app
    from upscaler import effects

    base = {name: getattr(effects.EffectParams(), name) for name in app._EFFECT_FIELDS}
    base.update({k: v for k, v in over.items() if k in base})
    region = [over.get("shape", "whole"), 50, 50, 50, 50, 0, 0, 15,
              over.get("outside", False), 25, over.get("faces"), over.get("editor")]
    return list(base.values()) + region


def test_app_effect_fields_match_the_dataclass():
    pytest.importorskip("gradio")
    import app
    from upscaler import effects

    names = {f.name for f in effects.EffectParams.__dataclass_fields__.values()}
    assert set(app._EFFECT_FIELDS) == names
    assert len(app._EFFECT_FIELDS) == len(names)      # no duplicates, no drift
    assert app._EFFECT_COLORS | app._EFFECT_INTS <= names


def test_app_effect_params_typing_and_mask():
    pytest.importorskip("gradio")
    import app

    p, m = app._effect_split(_vals(grain=20, posterize=6.0, halation_color="#112233",
                                   glitch_seed=3.0, shape="band", outside=True))
    assert p.grain == 20.0 and isinstance(p.grain, float)
    assert p.posterize == 6 and isinstance(p.posterize, int)     # sliders hand back floats
    assert p.glitch_seed == 3 and isinstance(p.glitch_seed, int)
    assert p.halation_color == "#112233"
    assert (m.shape, m.outside, m.progressive) == ("band", True, False)


def test_app_look_fans_out_to_every_control():
    pytest.importorskip("gradio")
    import app
    from upscaler import effects

    values = app.effects_preset_ui("Lomo")
    assert len(values) == len(app._EFFECT_FIELDS)
    p, _m = app._effect_split(list(values) + _vals()[len(app._EFFECT_FIELDS):])
    assert p == effects.PRESETS["Lomo"]
    cleared = app._effect_split(list(app.effects_preset_ui("None")) +
                               _vals()[len(app._EFFECT_FIELDS):])[0]
    assert cleared.is_identity()


def test_app_effects_preview_and_apply(tmp_path, monkeypatch):
    gr = pytest.importorskip("gradio")
    import app
    from upscaler import library

    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "lib")
    img = Image.new("RGB", (1400, 700), (120, 120, 120))
    ImageDraw.Draw(img).ellipse([600, 250, 800, 450], fill=(250, 250, 250))
    assert app.effects_preview_ui(None, *_vals()) is None
    before, after = app.effects_preview_ui(img, *_vals(vignette=70))
    assert before.size == (900, 450) == after.size
    assert after.getpixel((5, 5))[0] < before.getpixel((5, 5))[0]     # corners darkened
    pair, path, info = app.effects_apply_ui(img, *_vals(grain=25, halftone=60),
                                            progress=lambda *a, **k: None)
    assert pair[0].size == (1400, 700) == pair[1].size and path.endswith(".png")
    assert "grain 25" in info and "halftone 60" in info
    with pytest.raises(gr.Error):
        app.effects_apply_ui(None, *_vals(), progress=lambda *a, **k: None)
    with pytest.raises(gr.Error):
        app.effects_apply_ui(img, *_vals(shape="painted"), progress=lambda *a, **k: None)
    with pytest.raises(gr.Error):
        app.effects_apply_ui(img, *_vals(shape="faces"), progress=lambda *a, **k: None)
