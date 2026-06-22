"""App-level tests for the Lian Li flat-component-list invariant (Batch 5).

A miscount in _TEXT_FIELDS / N_CLOCK / the fan-out silently misaligns every
overlay value, so these guard the wiring directly."""

import pytest


def test_app_text_slot_has_motion_fields():
    pytest.importorskip("gradio")
    import app
    assert app._TEXT_FIELDS == 14  # added motion, speed, cps
    assert app.N_CLOCK == 1


def test_app_panel_params_includes_clock_and_motion():
    pytest.importorskip("gradio")
    import app
    from upscaler import panel

    base = ["Landscape · 1920×480", "cover", 1.0, 0, 0, "solid", "#000", "#333", 90]
    font = panel.DEFAULT_FONT
    text_on = [True, "HELLO", font, 120, "#fff", "center", 0, 0, 0, "#000", 0,
               "typewriter", 200, 5]
    text_off = [False, "", font, 180, "#fff", "center", 0, 0, 0, "#000", 0,
                "none", 120, 10]
    sticker = [False, None, 40, 0, 0, 0, 1.0]
    clock_on = [True, "%H:%M", font, 100, "#fff", "center", 0, 30, 0, "#000", 0]
    ov = text_on + text_off + text_off + sticker + sticker + clock_on
    assert len(ov) == app.N_OVERLAY_VALS

    p = app._panel_params(*(base + ov))
    kinds = {o["type"] for o in p.overlays}
    assert "clock" in kinds and "text" in kinds
    txt = next(o for o in p.overlays if o["type"] == "text")
    assert txt["motion"] == "typewriter" and txt["cps"] == 5.0
    clk = next(o for o in p.overlays if o["type"] == "clock")
    assert clk["content"] == "%H:%M"


def test_app_fan_layout_matches_component_order():
    pytest.importorskip("gradio")
    import app
    from upscaler.panel import PanelParams

    fan = app._fan_layout_to_components(PanelParams(overlays=[
        dict(type="text", content="A", motion="fade"),
        dict(type="clock", content="%S"),
    ]))
    # 9 base controls + the flat overlay-input list
    assert len(fan) == 9 + app.N_OVERLAY_VALS


def test_app_layout_roundtrip_through_panel_params():
    pytest.importorskip("gradio")
    import app
    from upscaler import panel_presets
    from upscaler.panel import PanelParams

    p = PanelParams(zoom=1.7, overlays=[
        dict(type="text", content="HELLO", motion="scroll-left", speed=150, cps=10),
        dict(type="clock", content="%H:%M:%S"),
    ])
    # fan to components, drop them back through _panel_params, expect the same
    fan = app._fan_layout_to_components(panel_presets.from_dict(panel_presets.to_dict(p)))
    p2 = app._panel_params(*fan)
    assert p2.zoom == 1.7
    assert {o["type"] for o in p2.overlays} == {"text", "clock"}
