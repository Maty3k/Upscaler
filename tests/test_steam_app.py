"""App-level wiring for the Steam Showcase tab: the control → params mapping
and the on-upload handler, without launching a server."""

import pytest


def test_app_steam_params_maps_controls_in_preview_input_order():
    pytest.importorskip("gradio")
    import app
    from upscaler import steam

    hidpi = list(steam.SCALES)[1]
    p = app._steam_params("manual", 1.5, -20, 35, "#123456", 300, hidpi)
    assert (p.fit, p.zoom, p.off_x, p.off_y) == ("manual", 1.5, -20.0, 35.0)
    assert (p.bg_color, p.tile_h, p.scale) == ("#123456", 300, 2)
    assert app._steam_params("cover", 1, 0, 0, "#000", 122, "unknown label").scale == 1


def test_app_steam_on_media_switches_format_and_animation_controls(tmp_path):
    pytest.importorskip("gradio")
    from PIL import Image
    import app

    still = tmp_path / "s.png"
    Image.new("RGB", (40, 40)).save(still)
    end, group, fmt, note = app.steam_on_media(str(still))
    assert group["visible"] is False and fmt["value"] == app._STEAM_FMT_STILL
    assert "Still image" in note

    end, group, fmt, note = app.steam_on_media(None)
    assert "Upload" in note and group["visible"] is False


def test_app_steam_format_choices_cover_all_three():
    pytest.importorskip("gradio")
    import app

    assert len({app._STEAM_FMT_STILL, app._STEAM_FMT_ANIM, app._STEAM_FMT_GIF}) == 3
    assert "GIF" in app._STEAM_FMT_GIF


def test_app_steam_has_cancel_flag():
    pytest.importorskip("gradio")
    import threading
    import app

    assert isinstance(app._STEAM_CANCEL, threading.Event)


def test_app_steam_export_requires_media():
    gr = pytest.importorskip("gradio")
    import app

    with pytest.raises(gr.Error):
        app.steam_export_ui(None, "cover", 1, 0, 0, "#000", 122, "x", app._STEAM_FMT_STILL,
                            "24", 0, 0, "normal", 5, True, "", progress=lambda *a, **k: None)
