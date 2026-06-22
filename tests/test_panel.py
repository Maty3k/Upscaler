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
    ov = [dict(type="text", content="HI", font=panel.DEFAULT_FONT, size=200,
               color="#ffffff", align="center", x=0, y=0, rotation=0,
               stroke="#000000", stroke_w=0)]
    p = panel.PanelParams(fit="cover", overlays=ov)
    blank = Image.new("RGB", (10, 10), (0, 0, 0))
    out = panel.compose_frame(blank, p)
    # some near-white text pixels should exist near the centre
    found = any(
        min(out.getpixel((x, 240))) > 200
        for x in range(820, 1100, 4)
    )
    assert found


def test_sticker_overlay_renders():
    sticker = Image.new("RGBA", (120, 120), (255, 0, 0, 255))
    ov = [dict(type="sticker", image=sticker, scale=50, x=0, y=0, rotation=0, opacity=1.0)]
    p = panel.PanelParams(fit="cover", bg_type="solid", bg_color="#0000ff", overlays=ov)
    out = panel.compose_frame(Image.new("RGB", (10, 10), (0, 0, 255)), p)
    assert out.getpixel((960, 240))[0] > 200  # red sticker at centre


def test_fonts_discovered():
    assert len(panel.FONT_NAMES) >= 1
    assert panel.DEFAULT_FONT in panel.FONTS


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


# ── Batch 5: clock + animated text ────────────────────────────────────────────
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402


def _full_canvas(overlays, **frame_kw):
    p = panel.PanelParams(fit="cover", bg_type="solid", bg_color="#000000",
                          overlays=overlays)
    blank = Image.new("RGB", (10, 10), (0, 0, 0))
    return panel.compose_frame(blank, p, **frame_kw)


def _clock(**kw):
    base = dict(type="clock", content="%H:%M:%S", font=panel.DEFAULT_FONT,
                size=200, color="#ffffff", align="center", x=0, y=0,
                rotation=0, stroke="#000000", stroke_w=0)
    base.update(kw)
    return base


def _white_pixels(img):
    return int((np.asarray(img.convert("L")) > 200).sum())


def test_clock_overlay_renders_template():
    out = _full_canvas([_clock()], now=datetime(2020, 1, 1, 13, 37, 42))
    assert _white_pixels(out) > 0  # digits drawn


def test_clock_advances_with_elapsed():
    fixed = datetime(2020, 1, 1, 0, 0, 0)
    a = _full_canvas([_clock(content="%M")], now=fixed, elapsed=0)
    b = _full_canvas([_clock(content="%M")], now=fixed, elapsed=60)
    assert not np.array_equal(np.asarray(a), np.asarray(b))  # minute changed


def test_clock_bad_template_does_not_crash():
    out = _full_canvas([_clock(content="%Q nonsense")], now=datetime(2020, 1, 1))
    assert out.size == (1920, 480)  # rendered the literal, no exception


def test_compose_frame_backcompat_signature():
    p = panel.PanelParams(fit="cover")
    src = Image.new("RGB", (40, 40), (10, 20, 30))
    assert panel.compose_frame(src, p).size == (1920, 480)
    assert panel.compose_frame(src, p, fast=True).size == (1920, 480)


def _text(**kw):
    base = dict(type="text", content="SCROLLING", font=panel.DEFAULT_FONT,
                size=120, color="#ffffff", align="center", x=0, y=0,
                rotation=0, stroke="#000000", stroke_w=0)
    base.update(kw)
    return base


def _white_centroid_x(img):
    a = (np.asarray(img.convert("L")) > 200)
    xs = np.where(a.any(axis=0))[0]
    return float(xs.mean()) if xs.size else None


def test_text_static_when_motion_none():
    ov = [_text()]
    a = _full_canvas(ov, frame_index=0, total_frames=60, fps=30)
    b = _full_canvas(ov, frame_index=30, total_frames=60, fps=30)
    assert np.array_equal(np.asarray(a), np.asarray(b))  # unchanged across frames


def test_text_scroll_offsets_by_frame():
    ov = [_text(motion="scroll-left", speed=200)]
    a = _full_canvas(ov, frame_index=0, total_frames=60, fps=30)
    b = _full_canvas(ov, frame_index=30, total_frames=60, fps=30)
    cx_a, cx_b = _white_centroid_x(a), _white_centroid_x(b)
    assert cx_a is not None and cx_b is not None
    assert cx_b < cx_a  # scrolled left


def test_typewriter_reveals_chars():
    ov = lambda fi: _full_canvas(  # noqa: E731
        [_text(content="ABCDE", motion="typewriter", cps=5)],
        frame_index=fi, total_frames=60, fps=30,
    )
    early = _white_pixels(ov(0))
    mid = _white_pixels(ov(30))   # 1.0s * 5cps = 5 chars
    assert early < mid


def test_fade_alpha_cycles():
    ov = [_text(content="FADE", motion="fade")]
    first = _white_pixels(_full_canvas(ov, frame_index=0, total_frames=60, fps=30))
    middle = _white_pixels(_full_canvas(ov, frame_index=30, total_frames=60, fps=30))
    last = _white_pixels(_full_canvas(ov, frame_index=60, total_frames=60, fps=30))
    assert middle > first          # brightest in the middle
    assert abs(last - first) <= 2  # returns to the start for a seamless loop


def test_scroll_seamless_wrap():
    ov = [_text(motion="scroll-left", speed=200)]
    a = _full_canvas(ov, frame_index=0, total_frames=60, fps=30)
    b = _full_canvas(ov, frame_index=60, total_frames=60, fps=30)  # one past last
    diff = np.abs(np.asarray(a).astype(int) - np.asarray(b).astype(int)).mean()
    assert diff < 1.0  # offset wraps to the start → near-identical
