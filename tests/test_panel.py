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


def _make_comp(dirpath, n):
    dirpath.mkdir()
    for i in range(1, n + 1):
        Image.new("RGB", (8, 8), (i, i, i)).save(dirpath / f"c_{i:05d}.png")


def test_apply_loop_frame_counts(tmp_path):
    import os

    # normal: untouched pattern
    d0 = tmp_path / "c0"
    _make_comp(d0, 10)
    assert panel._apply_loop(str(d0), 10, "normal", 10).endswith("c_%05d.png")

    # boomerang: forward + back = 2n - 2 frames
    d1 = tmp_path / "c1"
    _make_comp(d1, 10)
    patt = panel._apply_loop(str(d1), 10, "boomerang", 10)
    seq = os.path.dirname(patt)
    assert len([f for f in os.listdir(seq) if f.startswith("s_")]) == 18

    # crossfade: n - k frames (k = min(fps//2, n//3))
    d2 = tmp_path / "c2"
    _make_comp(d2, 12)
    patt = panel._apply_loop(str(d2), 12, "crossfade", 10)
    seq = os.path.dirname(patt)
    assert len([f for f in os.listdir(seq) if f.startswith("s_")]) == 12 - 4


def test_media_kind_and_hex():
    assert panel.media_kind(None) is None
    assert panel.media_kind("a.png") == "image"
    assert panel.media_kind("a.mp4") == "animated"
    assert panel._hex("#ff0000") == (255, 0, 0)
    assert panel._hex("#abc") == (170, 187, 204)
