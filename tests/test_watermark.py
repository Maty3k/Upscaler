"""Watermarks: placement, relative sizing, tiling, logos and the CLI."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from upscaler import watermark as wm
from upscaler.cli import main


def _photo(w=900, h=600, shade=120):
    return Image.new("RGB", (w, h), (shade, shade, shade))


def _logo(w=200, h=80):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([0, 0, w - 1, h - 1], fill=(255, 60, 60, 255))
    return img


def _arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _changed_box(before, after, threshold=12):
    """(x0, x1, y0, y1) of what the watermark touched."""
    d = np.abs(_arr(after) - _arr(before)).max(axis=2)
    ys, xs = np.where(d > threshold)
    assert len(xs), "nothing was drawn"
    return xs.min(), xs.max(), ys.min(), ys.max()


# ── nothing to draw ───────────────────────────────────────────────────────────

def test_empty_text_or_zero_opacity_is_a_no_op():
    img = _photo()
    assert wm.WatermarkParams(text="").is_identity()
    assert wm.WatermarkParams(text="   ").is_identity()
    assert wm.WatermarkParams(opacity=0).is_identity()
    assert not wm.WatermarkParams().is_identity()
    assert wm.apply(img, wm.WatermarkParams(text="")).tobytes() == img.tobytes()
    assert wm.apply(img, wm.WatermarkParams(opacity=0)).tobytes() == img.tobytes()


# ── placement ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("position", [p for p in wm.POSITIONS if p != wm.TILED])
def test_each_position_lands_in_its_own_corner(position):
    img = _photo()
    out = wm.apply(img, wm.WatermarkParams(position=position, opacity=100))
    x0, x1, y0, y1 = _changed_box(img, out)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    row, _, col = position.partition(" ")
    if position == "center":
        row = col = "center"
    if col == "left":
        assert cx < img.width * 0.34
    elif col == "right":
        assert cx > img.width * 0.66
    else:
        assert img.width * 0.34 < cx < img.width * 0.66
    if row == "top":
        assert cy < img.height * 0.34
    elif row == "bottom":
        assert cy > img.height * 0.66
    else:
        assert img.height * 0.34 < cy < img.height * 0.66


def test_margin_pushes_the_mark_away_from_the_edge():
    img = _photo()
    tight = wm.apply(img, wm.WatermarkParams(position="top left", margin=0, opacity=100))
    loose = wm.apply(img, wm.WatermarkParams(position="top left", margin=20, opacity=100))
    assert _changed_box(img, loose)[0] > _changed_box(img, tight)[0]
    assert _changed_box(img, loose)[2] > _changed_box(img, tight)[2]


def test_tiled_covers_the_whole_frame():
    img = _photo()
    out = wm.apply(img, wm.WatermarkParams(position=wm.TILED, text="PROOF", opacity=100))
    x0, x1, y0, y1 = _changed_box(img, out)
    assert x0 < img.width * 0.1 and x1 > img.width * 0.9
    assert y0 < img.height * 0.1 and y1 > img.height * 0.9
    # a wider gap means fewer marks on the page
    dense = np.abs(_arr(wm.apply(img, wm.WatermarkParams(position=wm.TILED, tile_gap=5,
                                                         opacity=100))) - _arr(img)).max(axis=2)
    sparse = np.abs(_arr(wm.apply(img, wm.WatermarkParams(position=wm.TILED, tile_gap=40,
                                                          opacity=100))) - _arr(img)).max(axis=2)
    assert (dense > 12).mean() > (sparse > 12).mean()


# ── sizing ────────────────────────────────────────────────────────────────────

def test_size_is_relative_so_a_preview_matches_the_export():
    """The same settings must cover the same share of a small and a large
    frame — that is what makes the downscaled preview honest."""
    coverage = []
    for size in ((450, 300), (900, 600), (1800, 1200)):
        img = _photo(*size)
        out = wm.apply(img, wm.WatermarkParams(position="center", size=10, opacity=100))
        d = np.abs(_arr(out) - _arr(img)).max(axis=2) > 12
        coverage.append(d.mean())
    assert max(coverage) - min(coverage) < 0.004


def test_text_size_and_opacity_scale():
    img = _photo()
    small = np.abs(_arr(wm.apply(img, wm.WatermarkParams(size=3, opacity=100))) - _arr(img))
    big = np.abs(_arr(wm.apply(img, wm.WatermarkParams(size=9, opacity=100))) - _arr(img))
    assert (big > 12).mean() > (small > 12).mean() * 3
    faint = np.abs(_arr(wm.apply(img, wm.WatermarkParams(opacity=20, position="center",
                                                         size=10))) - _arr(img)).max()
    solid = np.abs(_arr(wm.apply(img, wm.WatermarkParams(opacity=100, position="center",
                                                         size=10))) - _arr(img)).max()
    assert faint < solid / 2


def test_outline_keeps_white_text_readable_on_white():
    """The point of the outline default: without it a white mark on a white
    sky is invisible."""
    white = _photo(600, 400, shade=250)
    with_outline = np.abs(_arr(wm.apply(white, wm.WatermarkParams(
        position="center", size=8, opacity=100))) - _arr(white)).max()
    without = np.abs(_arr(wm.apply(white, wm.WatermarkParams(
        position="center", size=8, opacity=100, outline_width=0, shadow=0))) - _arr(white)).max()
    assert with_outline > 100 and without < 30


# ── logo ──────────────────────────────────────────────────────────────────────

def test_logo_scales_to_a_share_of_the_width():
    img = _photo()
    layer = wm._logo_layer(_logo(), wm.WatermarkParams(logo_scale=18), img.size)
    assert layer.width == round(0.18 * img.width)
    assert layer.height == round(layer.width * 80 / 200)      # aspect kept
    out = wm.apply(img, wm.WatermarkParams(kind="logo", position="top left"), logo=_logo())
    assert _changed_box(img, out)[0] < img.width * 0.34


def test_logo_without_an_image_is_an_error():
    with pytest.raises(ValueError, match="logo"):
        wm.apply(_photo(), wm.WatermarkParams(kind="logo"))


def test_unknown_kind_or_position_is_rejected():
    with pytest.raises(ValueError):
        wm.apply(_photo(), wm.WatermarkParams(kind="hologram"))
    with pytest.raises(ValueError):
        wm.apply(_photo(), wm.WatermarkParams(position="nowhere"))


# ── odds and ends ─────────────────────────────────────────────────────────────

def test_alpha_and_mode_are_preserved():
    rgb = _photo(200, 200)
    assert wm.apply(rgb, wm.WatermarkParams()).mode == "RGB"
    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.new("L", rgba.size, 123))
    out = wm.apply(rgba, wm.WatermarkParams())
    assert out.mode == "RGBA" and out.getchannel("A").getpixel((5, 5)) == 123


def test_rotation_changes_the_mark():
    img = _photo()
    flat = wm.apply(img, wm.WatermarkParams(position="center", size=10, opacity=100))
    tilted = wm.apply(img, wm.WatermarkParams(position="center", size=10, opacity=100,
                                              rotation=45))
    assert not np.array_equal(_arr(flat), _arr(tilted))
    _x0, _x1, y0, y1 = _changed_box(img, tilted)
    assert (y1 - y0) > (_changed_box(img, flat)[3] - _changed_box(img, flat)[2])


def test_presets_and_describe():
    img = _photo()
    assert wm.PRESET_NAMES[0] == wm.PRESET_NONE and len(wm.PRESETS) >= 6
    for name in wm.PRESETS:
        p = wm.preset(name)
        out = wm.apply(img, p, logo=_logo())
        assert np.abs(_arr(out) - _arr(img)).max() > 5, name
    assert wm.preset(wm.PRESET_NONE).is_identity()
    assert wm.preset("nonsense").is_identity()
    assert wm.preset("Proof (tiled)") is not wm.PRESETS["Proof (tiled)"]
    txt = wm.describe(wm.WatermarkParams(text="© Me", position=wm.TILED, rotation=20))
    assert '"© Me"' in txt and "tiled" in txt and "rotated 20°" in txt
    assert wm.describe(wm.WatermarkParams(opacity=0)) == "no watermark"


def test_preview_is_relative_and_survives_a_missing_logo():
    big = _photo(3000, 2000)
    out = wm.preview(big, wm.WatermarkParams())
    assert max(out.size) <= 1000
    # a logo mark with no logo yet returns the photo rather than raising
    assert wm.preview(big, wm.WatermarkParams(kind="logo")).size == out.size


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_watermark_text_and_preset(tmp_path):
    src = tmp_path / "p.png"
    _photo().save(src)
    assert main(["watermark", str(src), "--text", "© Test"]) == 0
    out = tmp_path / "p_wm.png"
    assert out.exists() and np.abs(_arr(Image.open(out)) - _arr(_photo())).max() > 20
    dst = tmp_path / "proof.jpg"
    assert main(["watermark", str(src), "-o", str(dst), "--preset", "Proof (tiled)"]) == 0
    assert Image.open(dst).format == "JPEG"
    # an explicit flag wins over the preset
    solid = tmp_path / "solid.png"
    assert main(["watermark", str(src), "-o", str(solid), "--preset", "Proof (tiled)",
                 "--opacity", "100"]) == 0
    faint = np.abs(_arr(Image.open(dst)) - _arr(_photo())).max()
    assert np.abs(_arr(Image.open(solid)) - _arr(_photo())).max() > faint


def test_cli_watermark_logo_and_folder(tmp_path):
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    for n in ("a", "b"):
        _photo().save(src_dir / f"{n}.png")
    logo_path = tmp_path / "logo.png"
    _logo().save(logo_path)
    out = tmp_path / "out"
    assert main(["watermark", str(src_dir), "-o", str(out), "--logo", str(logo_path),
                 "--position", "top left"]) == 0
    assert sorted(p.name for p in out.glob("*.png")) == ["a_wm.png", "b_wm.png"]
    box = _changed_box(_photo(), Image.open(out / "a_wm.png"))
    assert box[0] < 900 * 0.34 and box[2] < 600 * 0.34      # it really is top-left


def test_cli_watermark_fonts_and_errors(tmp_path, capsys):
    assert main(["watermark", "--list-fonts"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) >= 1
    assert main(["watermark", str(tmp_path / "nope.png")]) == 2
    src = tmp_path / "p.png"
    _photo().save(src)
    assert main(["watermark", str(src), "--font", "NoSuchFontXYZ"]) == 2
    assert "see --list-fonts" in capsys.readouterr().err
    assert main(["watermark", str(src), "--preset", "Logo corner"]) == 2
    assert "--logo is required" in capsys.readouterr().err
    assert main(["watermark", str(src), "--logo", str(tmp_path / "missing.png")]) == 2
