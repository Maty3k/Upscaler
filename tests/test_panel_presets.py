"""Tests for save/load shareable panel layouts (upscaler/panel_presets.py)."""

import json

import pytest
from PIL import Image

from upscaler import panel, panel_presets as PP
from upscaler.panel import PanelParams


def test_roundtrip_params_with_text_and_clock():
    p = PanelParams(
        orientation="Portrait · 480×1920", fit="contain", zoom=1.5,
        off_x=10, off_y=-5, bg_type="gradient", bg_color="#112233",
        bg_color2="#445566", bg_angle=45,
        overlays=[
            dict(type="text", content="HI", motion="scroll-left", speed=200, cps=8),
            dict(type="clock", content="%H:%M"),
        ],
    )
    p2 = PP.from_dict(json.loads(json.dumps(PP.to_dict(p))))
    assert p2.orientation == p.orientation and p2.zoom == 1.5 and p2.bg_angle == 45
    assert p2.overlays[0]["motion"] == "scroll-left"
    assert p2.overlays[1]["content"] == "%H:%M"


def test_sticker_image_base64_roundtrip():
    sticker = Image.new("RGBA", (24, 16), (200, 50, 10, 255))
    d = PP.to_dict(PanelParams(overlays=[dict(type="sticker", image=sticker, scale=40)]))
    # no live PIL in the JSON; the image is base64-embedded instead
    assert "image" not in d["overlays"][0] and "image_b64" in d["overlays"][0]
    json.dumps(d)  # must be JSON-serializable
    p2 = PP.from_dict(json.loads(json.dumps(d)))
    img = p2.overlays[0]["image"]
    assert img.size == (24, 16) and img.getpixel((0, 0))[:3] == (200, 50, 10)


def test_save_and_load_layout_file(tmp_path, monkeypatch):
    monkeypatch.setattr(PP, "PRESETS_DIR", tmp_path)
    p = PanelParams(zoom=2.0, overlays=[dict(type="text", content="X")])
    path = PP.save_layout(p, "my layout!")  # name sanitized
    assert path.parent == tmp_path and path.suffix == ".json"
    assert PP.load_layout(path).zoom == 2.0


def test_from_dict_tolerates_missing_and_unknown_keys():
    p = PP.from_dict({"weird": 1, "overlays": [{"type": "text"}]})
    assert isinstance(p, PanelParams) and p.overlays[0]["type"] == "text"
    assert PP.from_dict({}).overlays == []  # empty/garbage → defaults, no raise


def test_compose_frame_renders_loaded_layout():
    sticker = Image.new("RGBA", (200, 200), (0, 255, 0, 255))
    p = PanelParams(bg_color="#000000",
                    overlays=[dict(type="sticker", image=sticker, scale=80, x=0, y=0)])
    p2 = PP.from_dict(json.loads(json.dumps(PP.to_dict(p))))  # "shared to another machine"
    out = panel.compose_frame(Image.new("RGB", (10, 10), (0, 0, 0)), p2)
    assert out.getpixel((960, 240))[1] > 200  # green sticker survived the round-trip
