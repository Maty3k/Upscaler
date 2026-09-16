"""App-level wiring for the Screenshot tab."""

from __future__ import annotations

import pytest
from PIL import Image


def _vals(**over):
    import app
    from upscaler import screenshot as ss

    base = {name: getattr(ss.ShotParams(), name) for name in app._SHOT_FIELDS}
    base.update({k: v for k, v in over.items() if k in base})
    return list(base.values())


def _shot(w=300, h=200):
    return Image.new("RGB", (w, h), (0, 200, 0))


def test_app_sh_fields_match_the_dataclass():
    pytest.importorskip("gradio")
    import app
    from upscaler import screenshot as ss

    names = {f.name for f in ss.ShotParams.__dataclass_fields__.values()}
    # Every field has a control, every control maps to a real field, no repeats.
    assert set(app._SHOT_FIELDS) == names
    assert len(app._SHOT_FIELDS) == len(set(app._SHOT_FIELDS))
    assert app._SHOT_TEXT <= names


def test_app_sh_params_typing():
    pytest.importorskip("gradio")
    import app

    p = app._shot_params(*_vals(padding=14.0, tilt_y=20.0, title=None, chrome="browser"))
    assert p.padding == 14.0 and p.tilt_y == 20.0 and p.chrome == "browser"
    assert p.title == "" and isinstance(p.shadow, float)


def test_app_sh_visibility_follows_the_look():
    pytest.importorskip("gradio")
    import app
    from upscaler import screenshot as ss

    colour, colour2, angle, title, custom = app._shot_vis(
        ss.GRADIENT, ss.NO_CHROME, ss.AUTO_ASPECT)
    assert colour["visible"] and colour2["visible"] and angle["visible"]
    assert not title["visible"] and not custom["visible"]

    colour, colour2, angle, _, _ = app._shot_vis(ss.SOLID, ss.NO_CHROME, ss.AUTO_ASPECT)
    assert colour["visible"] and not colour2["visible"] and not angle["visible"]

    colour, *_ = app._shot_vis(ss.BLURRED, ss.NO_CHROME, ss.AUTO_ASPECT)
    assert not colour["visible"]                     # nothing to pick a colour for

    *_, title, custom = app._shot_vis(ss.GRADIENT, "browser-dark", ss.CUSTOM_ASPECT)
    assert title["visible"] and custom["visible"]
    *_, title, _ = app._shot_vis(ss.GRADIENT, "window", ss.AUTO_ASPECT)
    assert not title["visible"]                      # a plain window has no address bar


def test_app_sh_preset_fans_out():
    pytest.importorskip("gradio")
    import app
    from upscaler import screenshot as ss

    values = app.screenshot_preset_ui("Browser")
    assert len(values) == len(app._SHOT_FIELDS)
    assert app._shot_params(*values) == ss.PRESETS["Browser"]
    assert app._shot_params(*app.screenshot_preset_ui("None")) == ss.ShotParams()


def test_app_sh_preview_reports_the_full_size():
    pytest.importorskip("gradio")
    import app
    from upscaler import screenshot as ss

    assert app.screenshot_preview_ui(None, *_vals())[0] is None
    out, note = app.screenshot_preview_ui(_shot(), *_vals(padding=10.0))
    w, h = ss.result_size((300, 200), app._shot_params(*_vals(padding=10.0)))
    assert out is not None and f"{w}×{h}px" in note["value"]


def test_app_sh_preview_warns_instead_of_raising_on_a_bad_ratio():
    pytest.importorskip("gradio")
    import app
    from upscaler import screenshot as ss

    out, note = app.screenshot_preview_ui(
        _shot(), *_vals(aspect=ss.CUSTOM_ASPECT, custom_aspect="sideways"))
    assert out is None and "⚠" in note["value"]


def test_app_sh_apply_writes_a_png_and_needs_an_image(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    import app

    monkeypatch.setattr(app, "_ensure_export_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app.library, "save_path", lambda *a, **k: None)
    out, path, info = app.screenshot_apply_ui(_shot(), *_vals(padding=10.0))
    assert out.width > 300 and path.endswith(".png") and "✅" in info
    with Image.open(path) as im:
        assert im.size == out.size
    with pytest.raises(app.gr.Error):
        app.screenshot_apply_ui(None, *_vals())


def test_the_tab_is_wired_to_its_own_controls():
    """Every tab is built inside one long function, so a repeated variable
    prefix silently rebinds one tab's controls to another's — which is exactly
    what `sh_` did here, handing this tab's preset dropdown to Sharpen. Check
    the built graph rather than the source.
    """
    pytest.importorskip("gradio")
    import app

    wanted = {"screenshot_preset_ui", "screenshot_preview_ui", "screenshot_apply_ui"}
    deps = [f for f in app.build_demo().fns.values()
            if getattr(f.fn, "__name__", "") in wanted]
    assert {f.fn.__name__ for f in deps} == wanted

    preset = next(f for f in deps if f.fn.__name__ == "screenshot_preset_ui")
    assert [c.label for c in preset.inputs] == ["Look"]
    assert len(preset.outputs) == len(app._SHOT_FIELDS)

    for f in deps:
        if f.fn.__name__ == "screenshot_preset_ui":
            continue
        assert f.inputs[0].label.startswith("Screenshot")     # this tab's own image box
        assert len(f.inputs) == len(app._SHOT_FIELDS) + 1
