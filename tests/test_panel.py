"""Tests for the Lian Li panel builder's PIL compositor.

These cover the pure-PIL paths (no ffmpeg): exact canvas dimensions for every
fit mode and orientation, gradient background, text overlay, and media
classification. Animated export is exercised separately (needs ffmpeg).
"""

from __future__ import annotations

from PIL import Image

from upscaler import panel


def _src(w=1280, h=720, color=(20, 40, 200)):
    return Image.new("RGB", (w, h), color)


def test_canvas_sizes():
    assert panel.canvas_size("Landscape · 1920×480") == (1920, 480)
    assert panel.canvas_size("Portrait · 480×1920") == (480, 1920)


def test_compose_exact_dimensions_all_fits():
    src = _src()
    for orient, size in panel.ORIENTATIONS.items():
        for fit in panel.FITS:
            p = panel.PanelParams(orientation=orient, fit=fit)
            out = panel.compose_frame(src, p)
            assert out.size == size, (orient, fit, out.size)
            assert out.mode == "RGB"


def test_cover_fills_no_background_shows():
    # A cover fit of a 16:9 source must completely fill 1920×480 — the centre
    # column should be the source colour, never the (red) background.
    src = _src(color=(0, 0, 255))
    p = panel.PanelParams(fit="cover", bg_type="solid", bg_color="#ff0000")
    out = panel.compose_frame(src, p)
    assert out.getpixel((960, 240))[2] > 200  # blue present
    assert out.getpixel((960, 240))[0] < 50   # not red bg


def test_contain_letterboxes_with_background():
    # Contain leaves gaps that must show the background colour. A 16:9 source in
    # a 4:1 frame is height-limited, so the gaps fall on the LEFT/RIGHT edges.
    src = _src(color=(0, 0, 255))
    p = panel.PanelParams(fit="contain", bg_type="solid", bg_color="#ff0000")
    out = panel.compose_frame(src, p)
    assert out.getpixel((2, 240))[0] > 200       # side gap → red background
    assert out.getpixel((960, 240))[2] > 200     # centre → blue source


def test_gradient_background_differs_across_canvas():
    p = panel.PanelParams(fit="contain", bg_type="gradient",
                          bg_color="#000000", bg_color2="#ffffff", bg_angle=0)
    out = panel.compose_frame(_src(80, 80), p)
    left = out.getpixel((1, 240))
    right = out.getpixel((1918, 240))
    assert sum(right) > sum(left)  # brightens left→right at angle 0


def test_text_overlay_renders():
    p = panel.PanelParams(fit="cover", text="HI", text_size=200, text_color="#ffffff")
    blank = Image.new("RGB", (10, 10), (0, 0, 0))
    out = panel.compose_frame(blank, p)
    # some near-white text pixels should exist near the centre
    found = any(
        min(out.getpixel((x, 240))) > 200
        for x in range(880, 1040, 4)
    )
    assert found


def test_media_kind_and_hex():
    assert panel.media_kind(None) is None
    assert panel.media_kind("a.png") == "image"
    assert panel.media_kind("a.mp4") == "animated"
    assert panel._hex("#ff0000") == (255, 0, 0)
    assert panel._hex("#abc") == (170, 187, 204)
