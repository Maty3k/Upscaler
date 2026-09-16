"""The screenshot beautifier: the pass, the window chrome, the 3D lean, the
canvas, and the CLI.

The shot in these tests is a flat green rectangle, so any pixel that isn't
green came from the background, the shadow or the chrome — which makes "did it
land whole", "did the corner get rounded" and "which way does it lean" all
countable rather than eyeballed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from upscaler import screenshot as ss
from upscaler.cli import main

SHOT = (0, 200, 0)


def _shot(w=200, h=120):
    return Image.new("RGB", (w, h), SHOT)


def _arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.int16)


def _shot_pixels(img) -> int:
    a = _arr(img)
    return int(((a[..., 0] < 40) & (a[..., 1] > 160) & (a[..., 2] < 40)).sum())


def _plain(**over):
    """Neutral params: nothing but the background, so one effect at a time."""
    base = dict(background=ss.SOLID, color="#ffffff", padding=10.0,
                corner_radius=0.0, shadow=0.0)
    base.update(over)
    return ss.ShotParams(**base)


# ── the pass ──────────────────────────────────────────────────────────────────

def test_the_shot_lands_whole_with_room_around_it():
    src = _shot()
    out = ss.apply(src, _plain())
    assert out.width > src.width and out.height > src.height
    assert out.getpixel((out.width // 2, out.height // 2)) == SHOT
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert _shot_pixels(out) == src.width * src.height     # nothing cropped or scaled


def test_padding_is_a_share_of_the_short_side():
    """One setting has to suit a phone capture and a 5K grab, so the padding
    scales with the picture rather than being a pixel count."""
    small = ss.apply(_shot(200, 120), _plain(padding=10))
    big = ss.apply(_shot(400, 240), _plain(padding=10))
    assert (small.width - 200) * 2 == big.width - 400
    assert ss.apply(_shot(), _plain(padding=0)).size == (200, 120)


def test_rounded_corners_eat_into_the_shot_and_not_the_canvas():
    square = ss.apply(_shot(), _plain(corner_radius=0))
    rounded = ss.apply(_shot(), _plain(corner_radius=14))
    assert rounded.size == square.size                     # the canvas is unchanged
    assert _shot_pixels(rounded) < _shot_pixels(square)
    assert rounded.getpixel((rounded.width // 2, rounded.height // 2)) == SHOT


def test_an_rgba_shot_keeps_the_holes_it_came_with():
    src = _shot().convert("RGBA")
    hole = Image.new("L", src.size, 255)
    hole.paste(0, (0, 0, 40, 40))
    src.putalpha(hole)
    out = ss.apply(src, _plain(padding=10, color="#ff0000"))
    assert out.getpixel((12, 12)) == (255, 0, 0)           # the background shows through
    assert out.getpixel((out.width // 2, out.height // 2)) == SHOT


# ── the window frame ──────────────────────────────────────────────────────────

def test_chrome_puts_a_title_bar_with_three_dots_above_the_shot():
    out = ss.apply(_shot(400, 300), _plain(chrome="window", padding=0))
    assert out.height > 300 and out.width > 400            # a bar, and a hairline edge
    bar = _arr(out)[: out.height - 300]
    red = (bar[..., 0] > 200) & (bar[..., 1] < 160) & (bar[..., 2] < 160)
    green = (bar[..., 1] > 150) & (bar[..., 0] < 120)
    assert red.any() and green.any()                       # the close and zoom lights
    assert _shot_pixels(out) == 400 * 300                  # the shot itself is untouched


def test_the_dark_chrome_is_actually_darker():
    light = ss.apply(_shot(400, 300), _plain(chrome="window", padding=0))
    dark = ss.apply(_shot(400, 300), _plain(chrome="window-dark", padding=0))
    assert light.size == dark.size
    assert _arr(dark)[:20].mean() < _arr(light)[:20].mean()


def test_the_address_bar_shows_only_in_a_browser_frame():
    blank = ss.apply(_shot(400, 300), _plain(chrome="browser", padding=0))
    typed = ss.apply(_shot(400, 300), _plain(chrome="browser", padding=0,
                                             title="example.com"))
    assert blank.size == typed.size
    assert not np.array_equal(_arr(blank), _arr(typed))    # the text is drawn
    # a plain window has no address bar, so the title has nowhere to go
    win = ss.apply(_shot(400, 300), _plain(chrome="window", padding=0))
    win_titled = ss.apply(_shot(400, 300), _plain(chrome="window", padding=0,
                                                  title="example.com"))
    assert np.array_equal(_arr(win), _arr(win_titled))


def test_no_chrome_leaves_the_shot_the_size_it_was():
    out = ss.apply(_shot(), _plain(chrome=ss.NO_CHROME, padding=0))
    assert out.size == (200, 120)


def test_every_option_has_a_label():
    assert set(ss.BACKGROUND_LABELS) == set(ss.BACKGROUNDS)
    assert set(ss.CHROME_LABELS) == set(ss.CHROMES)


# ── in space ──────────────────────────────────────────────────────────────────

def _column_heights(out):
    a = np.asarray(out.getchannel("A"))
    return int((a[:, 2] > 128).sum()), int((a[:, -3] > 128).sum())


def test_tilt_leans_the_right_edge_away_and_negative_leans_the_left():
    p = _plain(background=ss.TRANSPARENT, padding=0, tilt_y=40)
    left, right = _column_heights(ss.apply(_shot(400, 300), p))
    assert right < left
    left, right = _column_heights(ss.apply(_shot(400, 300), replace(p, tilt_y=-40)))
    assert left < right
    # the box is the same either way: a lean is a trapezoid inside it
    assert ss.apply(_shot(400, 300), p).size == (400, 300)


def test_pitch_leans_the_top_edge_away():
    p = _plain(background=ss.TRANSPARENT, padding=0, tilt_x=40)
    a = np.asarray(ss.apply(_shot(400, 300), p).getchannel("A"))
    assert (a[2] > 128).sum() < (a[-3] > 128).sum()


def test_a_lean_of_nothing_changes_nothing():
    base = ss.apply(_shot(), _plain(padding=0))
    assert np.array_equal(_arr(base), _arr(ss.apply(_shot(), _plain(padding=0,
                                                                   tilt_x=0, tilt_y=0))))


def test_spin_grows_the_canvas_to_hold_the_corners():
    flat = ss.apply(_shot(400, 300), _plain(padding=0))
    spun = ss.apply(_shot(400, 300), _plain(padding=0, spin=20))
    assert spun.width > flat.width and spun.height > flat.height


# ── shadow and edge ───────────────────────────────────────────────────────────

def _above_below(out, pad, h):
    a = _arr(out).mean(axis=(1, 2))
    return a[pad - 8:pad - 3].mean(), a[pad + h + 3:pad + h + 8].mean()


def test_the_shadow_falls_below_the_shot():
    pad = 12                                     # 10% of the 120px short side
    lit = ss.apply(_shot(), _plain(shadow=0))
    above, below = _above_below(lit, pad, 120)
    assert above == pytest.approx(below, abs=0.5)          # no shadow: even all round
    shaded = ss.apply(_shot(), _plain(shadow=90))
    above, below = _above_below(shaded, pad, 120)
    assert below < above - 5


def test_a_softer_shadow_spreads_further():
    reach = []
    for softness in (0, 100):
        out = _arr(ss.apply(_shot(), _plain(shadow=90, padding=20,
                                            shadow_softness=softness)))
        row = out.mean(axis=(1, 2))
        reach.append(float((row[:20] < 250).sum()))        # how far it creeps upward
    assert reach[0] < reach[1]


def test_the_edge_highlight_lights_the_rim_of_a_dark_shot():
    dark = Image.new("RGB", (200, 120), (10, 10, 12))
    p = _plain(background=ss.SOLID, color="#0b0b0e", shadow=0, corner_radius=0)
    plain = _arr(ss.apply(dark, p))
    lit = _arr(ss.apply(dark, replace(p, rim=100)))
    assert plain.max() < 60                                # nothing separates them
    assert lit.max() > 200                                 # now there's a lit edge


# ── the background ────────────────────────────────────────────────────────────

def test_solid_is_one_colour_and_a_gradient_is_not():
    solid = ss.apply(_shot(), _plain(background=ss.SOLID, color="#123456"))
    assert solid.getpixel((0, 0)) == (18, 52, 86) == solid.getpixel(
        (solid.width - 1, solid.height - 1))
    ramp = ss.apply(_shot(), _plain(background=ss.GRADIENT, color="#000000",
                                    color2="#ffffff", angle=0))
    assert ramp.getpixel((0, 0))[0] < ramp.getpixel((ramp.width - 1, 0))[0]


def test_mesh_is_its_own_look_and_is_repeatable():
    kw = dict(color="#ff0000", color2="#0000ff", padding=25)
    mesh = ss.apply(_shot(), _plain(background=ss.MESH, **kw))
    ramp = ss.apply(_shot(), _plain(background=ss.GRADIENT, **kw))
    assert not np.array_equal(_arr(mesh), _arr(ramp))
    # the blobs are placed, not drawn at random: the same input must repeat
    assert np.array_equal(_arr(mesh), _arr(ss.apply(_shot(),
                                                    _plain(background=ss.MESH, **kw))))


def test_a_blurred_backdrop_is_made_from_the_shot_itself():
    src = Image.new("RGB", (200, 120), (200, 30, 30))
    out = ss.apply(src, _plain(background=ss.BLURRED, padding=20))
    corner = out.getpixel((2, 2))
    assert corner[0] > corner[1] and corner[0] > corner[2]     # still red
    assert corner[0] < 200                                     # but dimmed


def test_transparent_keeps_the_alpha_and_the_shadow():
    out = ss.apply(_shot(), _plain(background=ss.TRANSPARENT, padding=20, shadow=90))
    assert out.mode == "RGBA"
    a = np.asarray(out.getchannel("A"))
    assert a[0, 0] == 0                                    # the corner is see-through
    assert a[a.shape[0] // 2, a.shape[1] // 2] == 255      # the shot is not
    assert a.max() > 0 and 0 < a[len(a) - 12].max() < 255  # the shadow, half-there


# ── the canvas ────────────────────────────────────────────────────────────────

def test_a_fixed_shape_grows_the_background_and_never_crops_the_shot():
    out = ss.apply(_shot(400, 200), _plain(aspect="Square · 1:1", padding=5))
    assert out.width == out.height
    assert _shot_pixels(out) == 400 * 200


def test_auto_follows_the_shot():
    out = ss.apply(_shot(400, 200), _plain(aspect=ss.AUTO_ASPECT, padding=0))
    assert out.size == (400, 200)


def test_an_exact_size_is_honoured():
    assert ss.apply(_shot(), _plain(out_size="640x360")).size == (640, 360)


def test_a_custom_ratio_is_read_and_a_broken_one_is_refused():
    out = ss.apply(_shot(400, 200), _plain(aspect=ss.CUSTOM_ASPECT,
                                           custom_aspect="1:2", padding=0))
    assert out.height == out.width * 2
    with pytest.raises(ValueError, match="custom ratio"):
        ss.apply(_shot(), _plain(aspect=ss.CUSTOM_ASPECT, custom_aspect="sideways"))


# ── the rest of the surface ───────────────────────────────────────────────────

@pytest.mark.parametrize("name", list(ss.PRESETS))
def test_every_preset_runs_and_result_size_agrees_with_it(name):
    p = ss.preset(name)
    out = ss.apply(_shot(400, 300), p)
    assert out.width > 0 and out.height > 0
    assert ss.result_size((400, 300), p) == out.size


def test_presets_are_copies_and_none_resets():
    assert ss.preset("Indigo mesh") is not ss.PRESETS["Indigo mesh"]
    assert ss.preset(ss.PRESET_NONE) == ss.ShotParams()
    assert ss.preset("Nonsense") == ss.ShotParams()
    assert ss.PRESET_NAMES[0] == ss.PRESET_NONE


def test_preview_matches_the_full_size_shape():
    src = _shot(2000, 1200)
    p = ss.preset("Tilted 3D")
    small = ss.preview(src, p, max_edge=400)
    full = ss.apply(src, p)
    # max_edge caps the *shot* it works from, not the finished canvas — padding
    # and the spin add to it afterwards. What matters is that it stays cheap…
    assert small.width < full.width / 4
    # …and that the shape is the one the export will have.
    assert small.width / small.height == pytest.approx(full.width / full.height, rel=0.02)


def test_describe_reads_as_a_sentence():
    text = ss.describe(ss.preset("Browser"))
    assert text.startswith("Screenshot") and "browser frame" in text
    assert "gradient" in text and "padding" in text
    assert "tilt" in ss.describe(ss.preset("Tilted 3D"))
    assert "transparency" in ss.describe(ss.preset("No background"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_lists_the_presets(capsys):
    assert main(["screenshot", "--list-presets"]) == 0
    out = capsys.readouterr().out
    assert "Indigo mesh" in out and "README hero" in out and "padding" in out


def test_cli_runs_a_preset_and_names_the_output(tmp_path):
    src = tmp_path / "grab.png"
    _shot(400, 300).save(src)
    assert main(["screenshot", str(src), "--preset", "Indigo mesh"]) == 0
    out = tmp_path / "grab_shot.png"
    assert out.exists()
    with Image.open(out) as im:
        assert im.width > 400 and im.height > 300


def test_cli_flags_land_on_the_params(tmp_path, capsys):
    src = tmp_path / "grab.png"
    _shot(400, 300).save(src)
    dst = tmp_path / "card.png"
    assert main(["screenshot", str(src), "-o", str(dst), "--aspect", "16:9",
                 "--size", "1600x900", "--background", "mesh", "--tilt", "20",
                 "--chrome", "window-dark"]) == 0
    with Image.open(dst) as im:
        assert im.size == (1600, 900)
    err = capsys.readouterr().err
    assert "app window (dark)" in err and "tilt 20" in err


def test_cli_a_title_alone_means_you_want_a_browser(tmp_path, capsys):
    src = tmp_path / "grab.png"
    _shot(400, 300).save(src)
    assert main(["screenshot", str(src), "--title", "example.com"]) == 0
    assert "browser frame" in capsys.readouterr().err


def test_cli_folder_and_errors(tmp_path, capsys):
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    for name in ("a", "b"):
        _shot().save(src_dir / f"{name}.png")
    out = tmp_path / "out"
    assert main(["screenshot", str(src_dir), "-o", str(out), "--preset", "Sunset"]) == 0
    assert sorted(p.name for p in out.glob("*")) == ["a_shot.png", "b_shot.png"]

    assert main(["screenshot", str(tmp_path / "nope.png")]) == 2
    assert "not found" in capsys.readouterr().err
    assert main(["screenshot", str(src_dir / "a.png"), "--aspect", "sideways"]) == 2
    assert "can't read the ratio" in capsys.readouterr().err
