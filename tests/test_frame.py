"""Crop, straighten and frame: the geometry pass and the `upscaler crop` CLI."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from upscaler import frame
from upscaler.cli import main


def _photo(w=800, h=600):
    """A picture with a distinct colour in each corner, so any rotation,
    mirror or crop is identifiable from the pixels alone."""
    img = Image.new("RGB", (w, h), (120, 120, 120))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w // 2, h // 2], fill=(220, 30, 30))          # TL red
    d.rectangle([w // 2, 0, w, h // 2], fill=(30, 220, 30))          # TR green
    d.rectangle([0, h // 2, w // 2, h], fill=(30, 30, 220))          # BL blue
    d.rectangle([w // 2, h // 2, w, h], fill=(220, 220, 30))         # BR yellow
    return img


def _arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _corner(img, which):
    a = _arr(img)
    return {"tl": a[2, 2], "tr": a[2, -3], "bl": a[-3, 2], "br": a[-3, -3]}[which]


# ── crop ──────────────────────────────────────────────────────────────────────

def test_neutral_params_change_nothing():
    img = _photo()
    assert frame.FrameParams().is_identity()
    assert frame.apply(img, frame.FrameParams()).tobytes() == img.tobytes()
    assert not frame.FrameParams(border=1).is_identity()


@pytest.mark.parametrize("name", [n for n in frame.ASPECTS if frame.ASPECTS[n]])
def test_named_aspects_land_on_their_ratio(name):
    img = _photo()
    out = frame.apply(img, frame.FrameParams(aspect=name))
    rw, rh = frame.ASPECTS[name]
    assert out.width / out.height == pytest.approx(rw / rh, rel=0.01)
    assert out.width <= img.width and out.height <= img.height     # only ever removes


def test_custom_ratio_forms_and_bad_input():
    assert frame.parse_aspect("16:10") == frame.parse_aspect("16/10") == (16000, 10000)
    assert frame.parse_aspect("1200x800") == (1200000, 800000)
    assert frame.parse_aspect("3 : 2") == (3000, 2000)
    for bad in ("", "garbage", "16:", "0:5", "-3:2", "1:2:3"):
        assert frame.parse_aspect(bad) is None
    out = frame.apply(_photo(), frame.FrameParams(aspect=frame.CUSTOM_ASPECT,
                                                  custom_aspect="16:10"))
    assert out.width / out.height == pytest.approx(1.6, rel=0.01)
    with pytest.raises(ValueError):
        frame.apply(_photo(), frame.FrameParams(aspect=frame.CUSTOM_ASPECT,
                                                custom_aspect="nope"))


def test_position_chooses_what_survives():
    img = _photo()                       # 800×600, wider than 1:1 → sides are trimmed
    left = frame.apply(img, frame.FrameParams(aspect="Square · 1:1", position_x=0))
    right = frame.apply(img, frame.FrameParams(aspect="Square · 1:1", position_x=100))
    assert np.allclose(_corner(left, "tl"), (220, 30, 30), atol=3)     # kept the red side
    assert np.allclose(_corner(right, "tr"), (30, 220, 30), atol=3)    # kept the green side
    assert not np.array_equal(_arr(left), _arr(right))


def test_zoom_crops_in_further():
    img = _photo()
    one = frame.apply(img, frame.FrameParams(aspect="Square · 1:1", zoom=1))
    two = frame.apply(img, frame.FrameParams(aspect="Square · 1:1", zoom=2))
    assert two.width == pytest.approx(one.width / 2, abs=2)
    assert frame.apply(img, frame.FrameParams(zoom=99)).width >= 1      # clamped, no crash


def test_fit_mode_keeps_the_whole_photo():
    img = _photo(800, 500)
    out = frame.apply(img, frame.FrameParams(aspect="Square · 1:1", crop_mode="fit",
                                             border_color="#00ff00"))
    assert out.size == (800, 800)
    top = (out.height - img.height) // 2
    band = _arr(out)[top:top + img.height, :]
    assert np.abs(band - _arr(img)).max() < 2            # nothing was cropped
    assert np.allclose(_arr(out)[2, 400], (0, 255, 0), atol=2)   # margin took the fill
    blurred = frame.apply(img, frame.FrameParams(aspect="Square · 1:1", crop_mode="fit",
                                                 border_style="blurred photo"))
    assert blurred.size == (800, 800)
    assert not np.allclose(_arr(blurred)[2, 400], (0, 0, 0), atol=2)   # not empty


# ── orientation, straighten, lean ─────────────────────────────────────────────

def test_rotation_is_clockwise_and_flips_are_right():
    img = _photo()
    r90 = frame.apply(img, frame.FrameParams(rotate=90))
    assert r90.size == (600, 800)
    # turning clockwise brings the bottom-left corner up to the top-left
    assert np.allclose(_corner(r90, "tl"), (30, 30, 220), atol=3)
    assert frame.apply(img, frame.FrameParams(rotate=180)).size == img.size
    mirrored = frame.apply(img, frame.FrameParams(flip_h=True))
    assert np.allclose(_corner(mirrored, "tl"), (30, 220, 30), atol=3)   # green now left
    flipped = frame.apply(img, frame.FrameParams(flip_v=True))
    assert np.allclose(_corner(flipped, "tl"), (30, 30, 220), atol=3)    # blue now top


@pytest.mark.parametrize("angle", [3, -8, 15, -20])
def test_straighten_never_leaves_empty_corners(angle):
    img = _photo()
    out = frame.apply(img, frame.FrameParams(straighten=angle))
    assert out.size == frame.inscribed_rect(img.width, img.height, angle)
    assert out.width < img.width and out.height < img.height
    for which in ("tl", "tr", "bl", "br"):
        assert _corner(out, which).sum() > 30      # a real pixel, not the empty rotation


def test_inscribed_rect_edges():
    assert frame.inscribed_rect(800, 600, 0) == (800, 600)
    assert frame.inscribed_rect(800, 600, 90)[0] <= 600
    assert frame.inscribed_rect(0, 100, 10) == (0, 0)
    square = frame.inscribed_rect(500, 500, 45)
    assert square[0] == square[1] and square[0] < 500


@pytest.mark.parametrize("kw", [dict(keystone_v=70), dict(keystone_v=-70),
                                dict(keystone_h=70), dict(keystone_h=-70),
                                dict(keystone_h=40, keystone_v=40)])
def test_lean_correction_trims_back_to_real_pixels(kw):
    img = _photo()
    out = frame.apply(img, frame.FrameParams(**kw))
    a = _arr(out)
    # every edge of the result must be real picture, never the empty wedge
    for edge in (a[0], a[-1], a[:, 0], a[:, -1]):
        assert edge.mean() > 20
    assert out.size != img.size            # something was actually trimmed


# ── frame ─────────────────────────────────────────────────────────────────────

def test_border_grows_the_canvas_and_takes_its_colour():
    img = _photo()
    out = frame.apply(img, frame.FrameParams(border=10, border_color="#ff00ff"))
    pad = round(0.10 * min(img.size))
    assert out.size == (img.width + 2 * pad, img.height + 2 * pad)
    assert np.allclose(_corner(out, "tl"), (255, 0, 255), atol=2)
    # the photo itself is untouched inside the margin
    inner = _arr(out)[pad:pad + img.height, pad:pad + img.width]
    assert np.abs(inner - _arr(img)).max() < 2


def test_rounded_corners_and_shadow():
    img = _photo()
    rounded = frame.apply(img, frame.FrameParams(corner_radius=25))
    assert rounded.mode == "RGBA"
    assert rounded.getchannel("A").getpixel((1, 1)) == 0          # corner cut away
    assert rounded.getchannel("A").getpixel((img.width // 2, img.height // 2)) == 255
    framed = frame.apply(img, frame.FrameParams(corner_radius=25, border=8))
    assert framed.mode == "RGBA" and framed.size != img.size
    # a shadow darkens the border just under the photo
    plain = _arr(frame.apply(img, frame.FrameParams(border=12, border_color="#ffffff")))
    shaded = _arr(frame.apply(img, frame.FrameParams(border=12, border_color="#ffffff",
                                                     shadow=90)))
    pad = round(0.12 * min(img.size))
    strip = (slice(img.height + pad, img.height + pad + 6), slice(pad, img.width))
    assert shaded[strip].mean() < plain[strip].mean() - 10


def test_exact_output_size_and_presets():
    img = _photo()
    assert frame.apply(img, frame.FrameParams(out_size="1920x1080")).size == (1920, 1080)
    assert frame.apply(img, frame.FrameParams(out_size="nonsense")).size == img.size
    assert frame.PRESET_NAMES[0] == frame.PRESET_NONE and len(frame.PRESETS) >= 8
    for name in frame.PRESETS:
        out = frame.apply(img, frame.preset(name))
        assert out.width > 0 and out.height > 0, name
    assert frame.preset(frame.PRESET_NONE).is_identity()
    assert frame.preset("Polaroid") is not frame.PRESETS["Polaroid"]


def test_result_size_matches_apply_without_doing_the_work():
    img = _photo()
    for p in (frame.FrameParams(aspect="Square · 1:1", border=8),
              frame.FrameParams(straighten=6, rotate=90),
              frame.FrameParams(aspect="Story · 9:16", crop_mode="fit"),
              frame.preset("Phone wallpaper")):
        assert frame.result_size(img.size, p) == frame.apply(img, p).size


def test_preview_and_describe():
    big = _photo(3000, 2000)
    out = frame.preview(big, frame.FrameParams(aspect="Square · 1:1"))
    assert max(out.size) <= frame.PREVIEW_EDGE
    txt = frame.describe(frame.FrameParams(rotate=90, straighten=2, aspect="Square · 1:1",
                                           border=6, shadow=30), (800, 600))
    assert "rotate 90°" in txt and "straighten 2°" in txt and "crop to" in txt
    assert "border 6%" in txt and "shadow 30" in txt and "800×600 →" in txt
    assert "fit into" in frame.describe(frame.FrameParams(aspect="Square · 1:1",
                                                          crop_mode="fit"))
    assert frame.describe(frame.FrameParams()) == "no changes"


def test_alpha_is_carried_through():
    img = _photo(200, 200).convert("RGBA")
    img.putalpha(Image.new("L", img.size, 90))
    out = frame.apply(img, frame.FrameParams(aspect="Square · 1:1", zoom=1.5))
    assert out.mode == "RGBA" and out.getchannel("A").getpixel((5, 5)) == 90


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_crop_aspect_border_and_preset(tmp_path):
    src = tmp_path / "p.png"
    _photo().save(src)
    assert main(["crop", str(src), "--aspect", "1:1"]) == 0
    out = tmp_path / "p_framed.png"
    assert Image.open(out).size == (600, 600)
    dst = tmp_path / "polaroid.jpg"
    assert main(["crop", str(src), "-o", str(dst), "--preset", "Polaroid"]) == 0
    assert Image.open(dst).format == "JPEG"
    named = tmp_path / "named.png"
    assert main(["crop", str(src), "-o", str(named), "--aspect", "Widescreen · 16:9"]) == 0
    w, h = Image.open(named).size
    assert w / h == pytest.approx(16 / 9, rel=0.01)


def test_cli_crop_transforms_and_folder(tmp_path):
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    for n in ("a", "b"):
        _photo().save(src_dir / f"{n}.png")
    out = tmp_path / "out"
    assert main(["crop", str(src_dir), "-o", str(out), "--rotate", "90"]) == 0
    assert sorted(p.name for p in out.glob("*.png")) == ["a_framed.png", "b_framed.png"]
    assert Image.open(out / "a_framed.png").size == (600, 800)
    exact = tmp_path / "wall.png"
    assert main(["crop", str(src_dir / "a.png"), "-o", str(exact), "--size", "1600x900",
                 "--straighten", "4", "--lean-v", "30"]) == 0
    assert Image.open(exact).size == (1600, 900)
    pos = tmp_path / "pos.png"
    assert main(["crop", str(src_dir / "a.png"), "-o", str(pos), "--aspect", "1:1",
                 "--position", "0,0"]) == 0
    assert np.allclose(_corner(Image.open(pos), "tl"), (220, 30, 30), atol=3)


def test_cli_crop_errors(tmp_path, capsys):
    assert main(["crop", str(tmp_path / "nope.png")]) == 2
    src = tmp_path / "p.png"
    _photo().save(src)
    assert main(["crop", str(src), "--aspect", "nonsense"]) == 2
    assert "can't read the ratio" in capsys.readouterr().err
    assert main(["crop", str(src), "--size", "huge"]) == 2
    assert "can't read the size" in capsys.readouterr().err
    assert main(["crop", str(src), "--position", "bad"]) == 2
